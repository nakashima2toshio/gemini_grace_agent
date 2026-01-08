# 非同期・並列化 詳細設計書

## csv_to_chunks_text_para.py 並列処理アップグレード

**作成日**: 2025-01-08  
**プログラム名**: `csv_to_chunks_text_para.py`  
**配置場所**: `gemini_grace_agent/chunking/`

---

## 1. 概要

### 1.1 目的
現在のテキストチャンキング処理（約3時間）を、非同期・並列処理により大幅に高速化する。

### 1.2 対象コンポーネント
| 現在の名前 | 新しい名前 | 役割 |
|-----------|-----------|------|
| `LargeTextProcessor` | `LargeTextProcessorPara` | バッチ処理の並列化 |
| `chunk_overlap` | `chunk_overlap_para` | オーバーラップ判定の並列化 |

### 1.3 処理フロー（変更なし）
```
Step1 (階層分割) → Step2 (意味分割) → Step3 (オーバーラップ付与)
```
※ 各ステップ内で並列化を行い、ステップ間は順次実行を維持

### 1.4 制約条件
- **Gemini API制限**: 4000文字/リクエスト
- **処理単位**: 2000文字（安全マージン確保）
- **デフォルト並列数**: 8

---

## 2. ディレクトリ構成

```
gemini_grace_agent/
├── chunking/                           # チャンキング処理専用ディレクトリ
│   ├── csv_to_chunks_text_para.py      # メインスクリプト（並列版）
│   ├── async_api_client.py             # 非同期APIクライアント
│   ├── models.py                       # Pydanticモデル定義
│   ├── prompts.py                      # プロンプト定義
│   ├── utils.py                        # ユーティリティ関数
│   ├── checkpoint_manager.py           # チェックポイント管理
│   └── __init__.py
├── doc/
│   └── csv_to_chunks_text.md           # 本設計書
├── checkpoints/                        # チェックポイント保存先（自動生成）
├── logs/                               # ログ保存先（自動生成）
└── ... (他のプログラム)
```

---

## 3. 現状分析と課題

### 3.1 ボトルネック箇所

```
LargeTextProcessor.process()
├── Step1: recursive_character_text_splitter() × N バッチ  ← 直列
├── Step2: semantic_chunking() × M パラグラフ             ← 直列
└── Step3: chunk_overlap()
    └── check_continuity() × (チャンク数-1)              ← 直列
```

### 3.2 API呼び出し回数の見積もり（例）
- 10万文字のテキスト → 50バッチ（2000文字単位）
- Step1: 50回のAPI呼び出し
- Step2: 仮に200パラグラフ → 200回のAPI呼び出し
- Step3: 仮に300チャンク → 299回のAPI呼び出し
- **合計: 約550回のAPI呼び出し（直列実行）**

### 3.3 期待される改善効果

| 並列数 | 理論上の速度向上 | 予想処理時間（3時間基準） |
|-------|-----------------|------------------------|
| 1 | 1x (基準) | 3時間 |
| 4 | 4x | 45分 |
| 8 | 8x（デフォルト） | 22分 |
| 16 | 10-12x | 15-18分（レート制限考慮） |

---

## 4. アーキテクチャ設計

### 4.1 全体構成図

```
┌──────────────────────────────────────────────────────────────┐
│                        main()                                 │
│  - argparse で並列数を取得（デフォルト: 8）                     │
│  - asyncio.run() でイベントループ開始                          │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                 LargeTextProcessorPara                        │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  async process(text, model, max_workers)               │  │
│  │  ├── split_into_batches(text) [同期: 変更なし]          │  │
│  │  ├── await step1_parallel(batches)                     │  │
│  │  │   └── CheckpointManager.save("step1", results)      │  │
│  │  ├── await step2_parallel(paragraphs)                  │  │
│  │  │   └── CheckpointManager.save("step2", results)      │  │
│  │  └── await step3_parallel(chunks)                      │  │
│  │       └── CheckpointManager.save("step3", results)     │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    AsyncAPIClient                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  - AdaptiveSemaphore による動的並列数制御                │  │
│  │  - リトライロジック内蔵（3回、指数バックオフ）             │  │
│  │  - asyncio.to_thread() で同期APIをラップ                │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                   進捗表示 (tqdm)                             │
│  Step1: 100%|████████████████████| 50/50 [02:30<00:00]       │
│  Step2: 100%|████████████████████| 200/200 [05:00<00:00]     │
│  Step3: 100%|████████████████████| 299/299 [03:45<00:00]     │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 クラス・関数一覧

| ファイル | 名前 | 種別 | 役割 |
|---------|------|------|------|
| `async_api_client.py` | `AsyncAPIClient` | class | 非同期APIクライアント |
| `async_api_client.py` | `AdaptiveSemaphore` | class | 動的並列数制御 |
| `csv_to_chunks_text_para.py` | `LargeTextProcessorPara` | class | 並列化メイン処理 |
| `csv_to_chunks_text_para.py` | `chunk_overlap_para` | async func | 並列オーバーラップ処理 |
| `csv_to_chunks_text_para.py` | `async_recursive_character_text_splitter` | async func | Step1非同期版 |
| `csv_to_chunks_text_para.py` | `async_semantic_chunking` | async func | Step2非同期版 |
| `csv_to_chunks_text_para.py` | `async_check_continuity` | async func | Step3連続性判定非同期版 |
| `checkpoint_manager.py` | `CheckpointManager` | class | チェックポイント保存・復元 |
| `models.py` | `SentenceUnit`, etc. | class | Pydanticモデル |
| `prompts.py` | `PARAGRAPH_SEPARATION_PROMPT`, etc. | const | プロンプト定義 |
| `utils.py` | `show_paragraphs`, etc. | func | ユーティリティ |

---

## 5. 詳細設計

### 5.1 AdaptiveSemaphore クラス（動的並列数制御）

```python
# async_api_client.py

class AdaptiveSemaphore:
    """
    動的に並列数を調整するセマフォ
    - 429エラー（レート制限）発生時に並列数を減少
    - 連続成功時に並列数を増加
    """
    
    def __init__(
        self,
        initial_limit: int = 8,
        min_limit: int = 1,
        max_limit: int = 16,
        success_threshold: int = 10  # この回数連続成功で増加
    ):
        self._limit = initial_limit
        self._min_limit = min_limit
        self._max_limit = max_limit
        self._success_threshold = success_threshold
        self._semaphore = asyncio.Semaphore(initial_limit)
        self._consecutive_successes = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        await self._semaphore.acquire()
    
    def release(self):
        self._semaphore.release()
    
    async def on_success(self):
        """成功時に呼び出し"""
        async with self._lock:
            self._consecutive_successes += 1
            if self._consecutive_successes >= self._success_threshold:
                await self._increase_limit()
                self._consecutive_successes = 0
    
    async def on_rate_limit_error(self):
        """429エラー時に呼び出し"""
        async with self._lock:
            self._consecutive_successes = 0
            await self._decrease_limit()
    
    async def _decrease_limit(self):
        """並列数を減少"""
        new_limit = max(self._min_limit, self._limit - 2)
        if new_limit != self._limit:
            logger.warning(f"Decreasing concurrency: {self._limit} -> {new_limit}")
            self._limit = new_limit
            # セマフォを再作成
            self._semaphore = asyncio.Semaphore(new_limit)
    
    async def _increase_limit(self):
        """並列数を増加"""
        new_limit = min(self._max_limit, self._limit + 1)
        if new_limit != self._limit:
            logger.info(f"Increasing concurrency: {self._limit} -> {new_limit}")
            self._limit = new_limit
            self._semaphore = asyncio.Semaphore(new_limit)
    
    @property
    def current_limit(self) -> int:
        return self._limit
```

### 5.2 AsyncAPIClient クラス

```python
# async_api_client.py

import asyncio
import json
import logging
from typing import Type, Optional
from pydantic import BaseModel
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class AsyncAPIClient:
    """
    非同期APIクライアント
    - asyncio.to_thread() で同期APIをラップ
    - AdaptiveSemaphore で動的並列数制御
    - リトライロジック（3回、指数バックオフ）
    - 不完全JSONの検出とリトライ
    """
    
    def __init__(
        self,
        api_key: str,
        max_workers: int = 8,
        max_retries: int = 3,
        max_output_tokens: int = 4096  # 【追加】出力トークン制限
    ):
        self.client = genai.Client(api_key=api_key)
        self.semaphore = AdaptiveSemaphore(
            initial_limit=max_workers,
            min_limit=1,
            max_limit=max_workers * 2
        )
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens
        self._total_requests = 0
        self._failed_requests = 0
        self._truncated_responses = 0  # 【追加】切断カウント
    
    def _is_valid_json(self, text: str) -> bool:
        """JSONが完全かどうかチェック"""
        if not text:
            return False
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False
    
    def _is_truncated_response(self, response) -> bool:
        """レスポンスが切断されたかチェック"""
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, 'finish_reason', None)
                # 文字列の場合
                if isinstance(finish_reason, str):
                    return finish_reason not in ['STOP', 'stop']
                # Enum の場合（値が1 = STOP）
                if hasattr(finish_reason, 'value'):
                    return finish_reason.value != 1
        except Exception:
            pass
        return False
    
    async def generate_content(
        self,
        model: str,
        contents: str,
        response_schema: Type[BaseModel],
        task_id: Optional[str] = None
    ) -> Optional[str]:
        """
        セマフォで並列数を制御しながらAPI呼び出し
        失敗時は指数バックオフでリトライ
        """
        await self.semaphore.acquire()
        try:
            return await self._execute_with_retry(
                model, contents, response_schema, task_id
            )
        finally:
            self.semaphore.release()
    
    async def _execute_with_retry(
        self,
        model: str,
        contents: str,
        response_schema: Type[BaseModel],
        task_id: Optional[str]
    ) -> Optional[str]:
        """リトライロジック（不完全JSON対策含む）"""
        
        for attempt in range(self.max_retries):
            try:
                self._total_requests += 1
                
                # asyncio.to_thread で同期APIを非同期実行
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        max_output_tokens=self.max_output_tokens,  # 【重要】明示的に設定
                    ),
                )
                
                # 【追加】レスポンス切断チェック
                if self._is_truncated_response(response):
                    self._truncated_responses += 1
                    finish_reason = getattr(response.candidates[0], 'finish_reason', 'unknown')
                    raise ValueError(f"Response truncated (finish_reason: {finish_reason})")
                
                # 【追加】JSON完全性チェック
                if response.text and not self._is_valid_json(response.text):
                    self._truncated_responses += 1
                    raise ValueError(
                        f"Incomplete JSON detected. "
                        f"Length: {len(response.text)}, "
                        f"Preview: {response.text[-50:]}..."
                    )
                
                await self.semaphore.on_success()
                return response.text
                
            except ValueError as e:
                # 不完全レスポンス → リトライ
                wait_time = 2 ** attempt
                logger.warning(
                    f"[{task_id}] {e}. "
                    f"Retrying in {wait_time}s (attempt {attempt+1}/{self.max_retries})"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(wait_time)
                    
            except Exception as e:
                error_str = str(e).lower()
                
                # レート制限エラーの判定
                if "429" in error_str or "rate" in error_str or "quota" in error_str:
                    await self.semaphore.on_rate_limit_error()
                    wait_time = 30 * (attempt + 1)
                    logger.warning(
                        f"[{task_id}] Rate limit hit. "
                        f"Waiting {wait_time}s (attempt {attempt+1}/{self.max_retries})"
                    )
                else:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"[{task_id}] Error: {e}. "
                        f"Retrying in {wait_time}s (attempt {attempt+1}/{self.max_retries})"
                    )
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(wait_time)
        
        # 全リトライ失敗
        self._failed_requests += 1
        logger.error(f"[{task_id}] Failed after {self.max_retries} retries. Using fallback.")
        return None
    
    def get_stats(self) -> dict:
        """統計情報を取得"""
        return {
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "truncated_responses": self._truncated_responses,  # 【追加】
            "success_rate": (
                (self._total_requests - self._failed_requests) / self._total_requests * 100
                if self._total_requests > 0 else 0
            ),
            "current_concurrency": self.semaphore.current_limit
        }
```

### 5.3 CheckpointManager クラス

```python
# checkpoint_manager.py

import json
import os
from datetime import datetime
from typing import List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    チェックポイント管理
    - 各ステップ完了時に中間結果を保存
    - クラッシュ時に途中から再開可能
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints", job_id: Optional[str] = None):
        self.checkpoint_dir = checkpoint_dir
        self.job_id = job_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.job_dir = os.path.join(checkpoint_dir, self.job_id)
        os.makedirs(self.job_dir, exist_ok=True)
        logger.info(f"Checkpoint directory: {self.job_dir}")
    
    def save(self, step_name: str, data: List[str], metadata: Optional[dict] = None) -> str:
        """
        ステップの結果を保存
        
        Args:
            step_name: ステップ名 (step1, step2, step3)
            data: 保存するデータ（文字列リスト）
            metadata: 追加メタデータ
        
        Returns:
            保存したファイルパス
        """
        checkpoint_data = {
            "step": step_name,
            "timestamp": datetime.now().isoformat(),
            "count": len(data),
            "data": data,
            "metadata": metadata or {}
        }
        
        filepath = os.path.join(self.job_dir, f"{step_name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Checkpoint saved: {filepath} ({len(data)} items)")
        return filepath
    
    def load(self, step_name: str) -> Optional[List[str]]:
        """
        ステップの結果を読み込み
        
        Returns:
            保存されていたデータ、またはNone
        """
        filepath = os.path.join(self.job_dir, f"{step_name}.json")
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            
            logger.info(
                f"Checkpoint loaded: {filepath} "
                f"({checkpoint_data['count']} items, saved at {checkpoint_data['timestamp']})"
            )
            return checkpoint_data["data"]
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def exists(self, step_name: str) -> bool:
        """チェックポイントが存在するか確認"""
        filepath = os.path.join(self.job_dir, f"{step_name}.json")
        return os.path.exists(filepath)
    
    def get_latest_completed_step(self) -> Optional[str]:
        """最後に完了したステップを取得"""
        for step in ["step3", "step2", "step1"]:
            if self.exists(step):
                return step
        return None
    
    def clear(self):
        """このジョブのチェックポイントを削除"""
        import shutil
        if os.path.exists(self.job_dir):
            shutil.rmtree(self.job_dir)
            logger.info(f"Checkpoints cleared: {self.job_dir}")
    
    @classmethod
    def list_jobs(cls, checkpoint_dir: str = "./checkpoints") -> List[str]:
        """保存されているジョブIDの一覧を取得"""
        if not os.path.exists(checkpoint_dir):
            return []
        return sorted(os.listdir(checkpoint_dir), reverse=True)
```

### 5.4 LargeTextProcessorPara クラス

```python
# csv_to_chunks_text_para.py

import asyncio
import re
from typing import List, Optional
from datetime import datetime
from tqdm.asyncio import tqdm_asyncio
import logging

from .async_api_client import AsyncAPIClient
from .checkpoint_manager import CheckpointManager
from .models import StructuralResult, ContinuityResult
from .prompts import PARAGRAPH_SEPARATION_PROMPT, SEMANTIC_CHUNKING_PROMPT, CONTINUITY_CHECK_PROMPT

logger = logging.getLogger(__name__)


class LargeTextProcessorPara:
    """
    並列処理対応の大規模テキストプロセッサ
    
    Features:
        - 非同期・並列API呼び出し
        - 動的並列数調整
        - チェックポイント機能
        - プログレスバー表示
    """
    
    def __init__(
        self,
        block_size: int = 2000,
        max_workers: int = 8,
        checkpoint_dir: str = "./checkpoints",
        resume_job_id: Optional[str] = None
    ):
        """
        Args:
            block_size: バッチサイズ（デフォルト: 2000文字）
            max_workers: 並列数（デフォルト: 8）
            checkpoint_dir: チェックポイント保存ディレクトリ
            resume_job_id: 再開するジョブID（指定時は途中から再開）
        """
        self.block_size = block_size
        self.max_workers = max_workers
        self.api_client: Optional[AsyncAPIClient] = None
        self.checkpoint = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            job_id=resume_job_id
        )
    
    def split_into_batches(self, text: str) -> List[str]:
        """
        テキストを block_size 以下に分割（既存ロジック）
        """
        # 改行コードを統一
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        batches = []
        current_batch = []
        current_length = 0
        
        raw_lines = text.split('\n')
        
        for raw_line in raw_lines:
            # 行自体が block_size を超えている場合
            if len(raw_line) > self.block_size:
                if current_batch:
                    batches.append("\n".join(current_batch))
                    current_batch = []
                    current_length = 0
                
                # 長い行を block_size ずつスライス
                for i in range(0, len(raw_line), self.block_size):
                    chunk = raw_line[i: i + self.block_size]
                    batches.append(chunk)
                continue
            
            # 通常の積み上げ処理
            line_len = len(raw_line) + 1  # 改行分
            
            if current_length + line_len > self.block_size:
                if current_batch:
                    batches.append("\n".join(current_batch))
                current_batch = [raw_line]
                current_length = line_len
            else:
                current_batch.append(raw_line)
                current_length += line_len
        
        if current_batch:
            batches.append("\n".join(current_batch))
        
        return batches
    
    async def process(
        self,
        text: str,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None
    ) -> List[str]:
        """
        メイン処理（非同期版）
        
        Args:
            text: 処理対象テキスト
            model: Geminiモデル名
            api_key: APIキー（省略時は環境変数から取得）
        
        Returns:
            チャンク化されたテキストのリスト
        """
        import os
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set")
        
        self.api_client = AsyncAPIClient(
            api_key=api_key,
            max_workers=self.max_workers
        )
        
        start_time = datetime.now()
        logger.info(f"Processing started at {start_time}")
        logger.info(f"Text length: {len(text)} chars, Block size: {self.block_size}")
        logger.info(f"Max workers: {self.max_workers}")
        
        # チェックポイントから再開確認
        latest_step = self.checkpoint.get_latest_completed_step()
        if latest_step:
            logger.info(f"Resuming from checkpoint: {latest_step}")
        
        # Step1: 階層分割
        if latest_step in [None]:
            batches = self.split_into_batches(text)
            logger.info(f"Total batches: {len(batches)}")
            
            print("\n=== Step1: Hierarchical Splitting (Parallel) ===")
            step1_results = await self._step1_parallel(batches, model)
            self.checkpoint.save("step1", step1_results, {"batch_count": len(batches)})
        else:
            step1_results = self.checkpoint.load("step1")
            logger.info(f"Step1 loaded from checkpoint: {len(step1_results)} paragraphs")
        
        # Step2: 意味分割
        if latest_step in [None, "step1"]:
            print("\n=== Step2: Semantic Chunking (Parallel) ===")
            step2_results = await self._step2_parallel(step1_results, model)
            self.checkpoint.save("step2", step2_results)
        else:
            step2_results = self.checkpoint.load("step2")
            logger.info(f"Step2 loaded from checkpoint: {len(step2_results)} chunks")
        
        # Step3: オーバーラップ付与
        if latest_step in [None, "step1", "step2"]:
            print("\n=== Step3: Smart Overlap (Parallel) ===")
            step3_results = await self._step3_parallel(step2_results, model)
            self.checkpoint.save("step3", step3_results)
        else:
            step3_results = self.checkpoint.load("step3")
            logger.info(f"Step3 loaded from checkpoint: {len(step3_results)} final chunks")
        
        # 統計情報
        elapsed = (datetime.now() - start_time).total_seconds()
        stats = self.api_client.get_stats()
        
        print(f"\n=== Processing Complete ===")
        print(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print(f"Total chunks: {len(step3_results)}")
        print(f"API stats: {stats}")
        
        return step3_results
    
    async def _step1_parallel(self, batches: List[str], model: str) -> List[str]:
        """Step1: 各バッチを並列処理"""
        
        async def process_one(idx: int, batch_text: str) -> List[str]:
            if not batch_text.strip():
                return []
            
            if len(batch_text) > 10000:
                logger.warning(f"Batch {idx} too large ({len(batch_text)} chars), skipping")
                return []
            
            response = await self.api_client.generate_content(
                model=model,
                contents=f"{PARAGRAPH_SEPARATION_PROMPT}\n\n【対象テキスト】\n{batch_text}",
                response_schema=StructuralResult,
                task_id=f"step1-{idx}"
            )
            
            if response:
                try:
                    result = StructuralResult.model_validate_json(response)
                    return [p.full_text.replace('\n', ' ') for p in result.paragraphs]
                except Exception as e:
                    logger.error(f"Step1 parse error at batch {idx}: {e}")
            
            return [batch_text]  # フォールバック: 元テキストをそのまま返す
        
        # タスク生成
        tasks = [process_one(i, batch) for i, batch in enumerate(batches)]
        
        # 並列実行（tqdmでプログレス表示）
        results = []
        for coro in tqdm_asyncio.as_completed(tasks, desc="Step1", total=len(tasks)):
            result = await coro
            results.append(result)
        
        # 結果をフラット化（順序はas_completedで崩れるので、順序が必要なら別途対応）
        # ここでは順序を維持するためgatherを使用
        ordered_results = await asyncio.gather(*[process_one(i, b) for i, b in enumerate(batches)])
        
        paragraphs = []
        for batch_result in ordered_results:
            paragraphs.extend(batch_result)
        
        logger.info(f"Step1 complete: {len(paragraphs)} paragraphs")
        return paragraphs
    
    async def _step2_parallel(self, paragraphs: List[str], model: str) -> List[str]:
        """Step2: 各パラグラフを並列処理"""
        
        async def process_one(idx: int, para_text: str) -> List[str]:
            if not para_text.strip():
                return []
            
            response = await self.api_client.generate_content(
                model=model,
                contents=f"{SEMANTIC_CHUNKING_PROMPT}\n\n【対象テキスト】\n{para_text}",
                response_schema=StructuralResult,
                task_id=f"step2-{idx}"
            )
            
            if response:
                try:
                    result = StructuralResult.model_validate_json(response)
                    chunks = []
                    for p in result.paragraphs:
                        clean_text = p.full_text.replace('\n', ' ')
                        if not clean_text.endswith('\n'):
                            clean_text += '\n'
                        chunks.append(clean_text)
                    return chunks
                except Exception as e:
                    logger.error(f"Step2 parse error at para {idx}: {e}")
            
            return [para_text]  # フォールバック
        
        # タスク生成＆並列実行
        tasks = [process_one(i, para) for i, para in enumerate(paragraphs)]
        
        # プログレス表示付きで実行
        ordered_results = []
        for f in tqdm_asyncio.as_completed(
            [asyncio.create_task(t) for t in tasks],
            desc="Step2",
            total=len(tasks)
        ):
            await f
        
        # 順序維持のため再度gather
        ordered_results = await asyncio.gather(*[process_one(i, p) for i, p in enumerate(paragraphs)])
        
        chunks = []
        for para_result in ordered_results:
            chunks.extend(para_result)
        
        logger.info(f"Step2 complete: {len(chunks)} chunks")
        return chunks
    
    async def _step3_parallel(self, chunks: List[str], model: str) -> List[str]:
        """Step3: 連続性判定を並列実行"""
        return await chunk_overlap_para(chunks, self.api_client, model)


async def chunk_overlap_para(
    paragraphs: List[str],
    api_client: AsyncAPIClient,
    model: str = "gemini-2.0-flash"
) -> List[str]:
    """
    並列版オーバーラップ処理
    
    連続性判定(check_continuity)は独立して並列実行可能。
    結果の適用は元の順序を維持する。
    """
    if not paragraphs:
        return []
    
    if len(paragraphs) == 1:
        return paragraphs
    
    async def check_one(idx: int, prev_text: str, next_text: str) -> tuple[int, bool]:
        """1ペアの連続性を判定"""
        response = await api_client.generate_content(
            model=model,
            contents=(
                f"{CONTINUITY_CHECK_PROMPT}\n\n"
                f"【前のテキスト】\n{prev_text}\n\n"
                f"【次のテキスト】\n{next_text}"
            ),
            response_schema=ContinuityResult,
            task_id=f"step3-{idx}"
        )
        
        if response:
            try:
                result = ContinuityResult.model_validate_json(response)
                return (idx, result.is_connected)
            except Exception as e:
                logger.error(f"Step3 parse error at pair {idx}: {e}")
        
        return (idx, False)  # エラー時は安全側（分割）
    
    # 全ペアのタスクを生成
    tasks = []
    for i in range(1, len(paragraphs)):
        prev_text = paragraphs[i - 1]
        current_text = paragraphs[i]
        tasks.append(check_one(i, prev_text, current_text))
    
    logger.info(f"Checking continuity for {len(tasks)} pairs...")
    
    # 並列実行（プログレス表示付き）
    continuity_map = {}
    for coro in tqdm_asyncio.as_completed(tasks, desc="Step3", total=len(tasks)):
        idx, is_connected = await coro
        continuity_map[idx] = is_connected
    
    # オーバーラップ適用（順序通りに処理）
    overlapped_result = [paragraphs[0]]
    
    for i in range(1, len(paragraphs)):
        current_text = paragraphs[i]
        is_connected = continuity_map.get(i, False)
        
        if not is_connected:
            overlapped_result.append(current_text)
            continue
        
        # オーバーラップ処理: 前のチャンクの最後の1文を追加
        prev_text = paragraphs[i - 1]
        sentences = re.split(r'(?<=[。．！!？?])', prev_text)
        sentences = [s for s in sentences if s.strip()]
        
        overlap_part = sentences[-1] if sentences else prev_text
        combined_text = overlap_part + current_text
        overlapped_result.append(combined_text)
    
    logger.info(f"Step3 complete: {len(overlapped_result)} final chunks")
    return overlapped_result
```

### 5.5 コマンドライン引数

```python
# csv_to_chunks_text_para.py (続き)

import argparse
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="テキストチャンキング処理（並列版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使用方法
  python csv_to_chunks_text_para.py -i input.txt -o output.txt
  
  # 並列数を16に指定
  python csv_to_chunks_text_para.py -i input.txt -o output.txt -w 16
  
  # 中断したジョブを再開
  python csv_to_chunks_text_para.py -i input.txt -o output.txt --resume 20250108_143022
  
  # 保存済みジョブの一覧表示
  python csv_to_chunks_text_para.py --list-jobs
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="入力ファイルパス"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="出力ファイルパス（省略時は標準出力）"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=8,
        help="並列処理数（デフォルト: 8）"
    )
    parser.add_argument(
        "-b", "--block-size",
        type=int,
        default=2000,
        help="バッチサイズ（デフォルト: 2000文字）"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="gemini-2.0-flash",
        help="使用するGeminiモデル（デフォルト: gemini-2.0-flash）"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="チェックポイント保存ディレクトリ（デフォルト: ./checkpoints）"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="JOB_ID",
        help="指定したジョブIDから処理を再開"
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="保存済みジョブの一覧を表示"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="詳細ログを出力"
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """ロギングの設定"""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # ログディレクトリ作成
    os.makedirs("./logs", exist_ok=True)
    
    # ログフォーマット
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # ファイルハンドラ
    file_handler = logging.FileHandler(
        f"./logs/chunking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # コンソールハンドラ
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    
    # ルートロガー設定
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


async def async_main():
    args = parse_args()
    
    # ジョブ一覧表示
    if args.list_jobs:
        jobs = CheckpointManager.list_jobs(args.checkpoint_dir)
        if jobs:
            print("Saved jobs:")
            for job_id in jobs:
                print(f"  - {job_id}")
        else:
            print("No saved jobs found.")
        return
    
    # 入力ファイルチェック
    if not args.input:
        print("Error: --input is required")
        return
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # ロギング設定
    setup_logging(args.verbose)
    
    logger.info("=" * 60)
    logger.info("Text Chunking Processor (Parallel Version)")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output or 'stdout'}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Block size: {args.block_size}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Resume job: {args.resume or 'None'}")
    
    # ファイル読み込み
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()
    logger.info(f"Text length: {len(text)} chars")
    
    # プロセッサ初期化
    processor = LargeTextProcessorPara(
        block_size=args.block_size,
        max_workers=args.workers,
        checkpoint_dir=args.checkpoint_dir,
        resume_job_id=args.resume
    )
    
    # 処理実行
    start_time = time.time()
    results = await processor.process(text, model=args.model)
    elapsed = time.time() - start_time
    
    logger.info(f"Processing completed in {elapsed:.2f} seconds")
    logger.info(f"Total chunks: {len(results)}")
    
    # 出力
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(results):
                f.write(f"--- Chunk {i+1} ---\n")
                f.write(chunk)
                f.write("\n")
        logger.info(f"Results saved to: {args.output}")
    else:
        print("\n=== Results (first 10 chunks) ===")
        for i, chunk in enumerate(results[:10]):
            print(f"[{i+1}] {chunk[:100]}...")


def main():
    """エントリーポイント"""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
```

---

## 6. エラーハンドリング詳細

### 6.1 リトライ戦略

| エラー種別 | リトライ | 待機時間 | 理由 |
|-----------|---------|---------|------|
| レート制限 (429) | ○ | 30秒 × 試行回数 | 時間を置けば解消 |
| タイムアウト | ○ | 指数バックオフ | 一時的な問題 |
| サーバーエラー (5xx) | ○ | 指数バックオフ | 一時的な問題 |
| **不完全JSON** | ○ | 指数バックオフ | 出力切断、再試行で解消の可能性 |
| 認証エラー (401) | × | - | 設定の問題 |
| 不正リクエスト (400) | × | - | 入力データの問題 |

### 6.2 不完全JSONレスポンスへの対策

#### 6.2.1 問題の概要

Gemini APIからのレスポンスが途中で切れ、不完全なJSONが返される場合がある。

```
Error: EOF while parsing a string at line 10 column 25
input_value='{\n  "paragraphs": [\n  ...        "text": "三年'
```

**発生頻度**: 約1%（100回に1回程度）
**影響**: 該当バッチのデータが欠損

#### 6.2.2 原因

1. **出力トークン制限**: `max_output_tokens` のデフォルト値に達した
2. **ネットワーク切断**: 通信途中でタイムアウト
3. **APIサーバー負荷**: サーバー側でレスポンスが切断

#### 6.2.3 対策実装

**対策A: max_output_tokens の明示的設定**

```python
config=types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=StructuralResult,
    max_output_tokens=4096,  # 明示的に設定（重要）
)
```

**対策B: 不完全JSON検出とリトライ**

```python
class AsyncAPIClient:
    
    def _is_valid_json(self, text: str) -> bool:
        """JSONが完全かどうかチェック"""
        try:
            import json
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False
    
    def _is_truncated_response(self, response) -> bool:
        """レスポンスが切断されたかチェック"""
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            finish_reason = getattr(candidate, 'finish_reason', None)
            # STOP以外は異常終了
            return finish_reason not in [None, 'STOP', 1]  # 1 = STOP (enum値)
        return False
    
    async def _execute_with_retry(self, ...):
        for attempt in range(self.max_retries):
            try:
                response = await asyncio.to_thread(...)
                
                # 切断チェック
                if self._is_truncated_response(response):
                    raise ValueError(
                        f"Response truncated (finish_reason: {response.candidates[0].finish_reason})"
                    )
                
                # JSON完全性チェック
                if response.text and not self._is_valid_json(response.text):
                    raise ValueError("Incomplete JSON response detected")
                
                await self.semaphore.on_success()
                return response.text
                
            except ValueError as e:
                # 不完全レスポンスはリトライ
                logger.warning(f"[{task_id}] {e}, retrying...")
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
                continue
            except Exception as e:
                # その他のエラー処理...
```

**対策C: フォールバック時のデータ保持**

```python
# エラー時も元データを保持（データ欠損防止）
async def process_one(idx: int, batch_text: str) -> List[str]:
    response = await self.api_client.generate_content(...)
    
    if response:
        try:
            result = StructuralResult.model_validate_json(response)
            return [p.full_text.replace('\n', ' ') for p in result.paragraphs]
        except Exception as e:
            logger.error(f"Parse error at batch {idx}: {e}")
    
    # 【重要】フォールバック: 元テキストを保持してデータ欠損を防ぐ
    logger.warning(f"Batch {idx}: Using original text as fallback")
    return [batch_text]
```

**対策D: バッチサイズの調整オプション**

```python
# コマンドラインオプション
parser.add_argument(
    "-b", "--block-size",
    type=int,
    default=2000,
    help="バッチサイズ（不完全JSON多発時は1500に縮小推奨）"
)
```

#### 6.2.4 エラー発生時のログ出力

```python
# 詳細なエラーログを出力
logger.error(
    f"[Batch {idx}] JSON Parse Error\n"
    f"  Error: {e}\n"
    f"  Input length: {len(batch_text)} chars\n"
    f"  Response length: {len(response) if response else 0} chars\n"
    f"  Response preview: {response[:200] if response else 'None'}..."
)
```

### 6.3 リトライフロー

```
試行1: 即時実行
  ↓ 失敗（通常エラー or 不完全JSON）
試行2: 2秒待機後に実行
  ↓ 失敗（通常エラー or 不完全JSON）
試行3: 4秒待機後に実行
  ↓ 失敗
フォールバック → 元テキストを保持 → 次の処理へ

※ 429エラーの場合:
試行1 → 30秒待機 → 試行2 → 60秒待機 → 試行3 → 90秒待機

※ 不完全JSONの場合:
試行1 → 2秒待機 → 試行2 → 4秒待機 → 試行3 → フォールバック
```

### 6.4 フォールバック動作（データ欠損防止）

| ステップ | フォールバック | データ保持 |
|---------|--------------|-----------|
| Step1 | 元のバッチテキストをそのまま返す | ○ |
| Step2 | 元のパラグラフをそのまま返す | ○ |
| Step3 | `is_connected = False`（分割側、安全） | ○ |

**重要**: 従来は `return []` でデータが欠損していたが、  
新設計では `return [original_text]` で元データを保持する。

---

## 7. 使用方法

### 7.1 基本的な使用方法

```bash
# 基本実行
python -m chunking.csv_to_chunks_text_para -i input.txt -o output.txt

# 並列数を変更（デフォルト: 8）
python -m chunking.csv_to_chunks_text_para -i input.txt -o output.txt -w 16

# 詳細ログを出力
python -m chunking.csv_to_chunks_text_para -i input.txt -o output.txt -v
```

### 7.2 ジョブの再開

```bash
# 保存済みジョブの一覧表示
python -m chunking.csv_to_chunks_text_para --list-jobs

# 特定のジョブから再開
python -m chunking.csv_to_chunks_text_para -i input.txt -o output.txt --resume 20250108_143022
```

### 7.3 モジュールとして使用

```python
import asyncio
from chunking import LargeTextProcessorPara

async def process_text():
    processor = LargeTextProcessorPara(
        block_size=2000,
        max_workers=8
    )
    
    with open("input.txt", "r") as f:
        text = f.read()
    
    results = await processor.process(text)
    return results

# 実行
results = asyncio.run(process_text())
```

---

## 8. 依存パッケージ

```txt
# requirements.txt
google-genai>=0.1.0
pydantic>=2.0.0
tqdm>=4.65.0
```

---

## 9. 変更履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|---------|
| 2025-01-08 | 1.0 | 初版作成 |
| 2025-01-08 | 1.1 | 確認事項の回答を反映、デフォルト並列数を8に変更 |
