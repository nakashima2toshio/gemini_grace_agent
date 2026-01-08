

# Gemini API非同期・並列処理 詳細設計書

## 1. 設計概要

### 1.1 目的
- Gemini API呼び出しの並列化による処理時間の大幅短縮（目標: 3時間 → 30分程度）
- 大規模テキストの効率的な分散処理
- エラー耐性の向上（リトライ機構）

### 1.2 対象コンポーネント
- `LargeTextProcessor` → `LargeTextProcessorPara`（並列版）
- `chunk_overlap` → `chunk_overlap_para`（並列版）
- 関連するAPI呼び出し関数の非同期化

---

## 2. 疑問点・要確認事項

### 🔴 重要な確認事項

#### Q1: Step1の並列化粒度
- **Option A**: バッチ単位で並列処理（推奨）
  - 各バッチを独立して並列処理
  - 実装が簡潔
- **Option B**: バッチ内のparagraphも並列処理
  - より細かい粒度での並列化
  - 複雑だが最大限の高速化

**推奨**: Option A（バッチ単位）

#### Q2: Step2の並列化粒度
- Step1の結果（各paragraph）を並列でsemantic_chunkingに投入
- 順序を保持する必要があるため、結果の並べ替えが必要

#### Q3: Step3の並列化戦略
**問題**: `chunk_overlap`は前のチャンクに依存する逐次処理
- **Option A**: スライディングウィンドウ方式
  - 複数の連続性チェックを並列実行
  - 例: chunks[0-1], [1-2], [2-3]を同時にチェック
- **Option B**: バッチ分割方式
  - チャンクを複数グループに分割し、各グループ内を並列処理
  - グループ境界は後で処理

**推奨**: Option A（スライディングウィンドウ）

#### Q4: エラー時のデータ処理
- リトライ3回後も失敗した場合:
  - **Option A**: 空の結果を返す（推奨）
  - **Option B**: 元のテキストをそのまま返す
  - **Option C**: エラーマークを付けて返す

**推奨**: Option A + ログに記録

#### Q5: yieldの使用目的
- **Option A**: async generator（進捗報告用）
  ```python
  async for progress in processor.process_async(...):
      print(f"Progress: {progress}%")
  ```
- **Option B**: 最終結果のみ返却（通常のasync関数）

**推奨**: Option B（シンプルさ優先）、必要なら進捗コールバック追加

---

## 3. アーキテクチャ設計

### 3.1 非同期化する関数

```python
# 現在（同期版）                          # 新規（非同期版）
recursive_character_text_splitter()  →  recursive_character_text_splitter_async()
semantic_chunking()                  →  semantic_chunking_async()
check_continuity()                   →  check_continuity_async()
```

### 3.2 並列制御機構

```python
import asyncio
from typing import List, Optional, Callable
import argparse

class AsyncConfig:
    """非同期処理の設定"""
    def __init__(self, max_concurrent: int = 8, max_retries: int = 3):
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent)
```

### 3.3 リトライ機構

```python
async def retry_async(
    func: Callable,
    *args,
    max_retries: int = 3,
    **kwargs
) -> Optional[any]:
    """
    非同期関数を実行し、失敗時は最大max_retries回リトライ
    """
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error after {max_retries} retries: {e}")
                return None
            await asyncio.sleep(2 ** attempt)  # 指数バックオフ
```

---

## 4. 処理フロー詳細

### 4.1 全体フロー

```
Input Text
    ↓
┌─────────────────────────────────────┐
│ Step0: テキスト分割（同期処理）      │
│  - split_into_batches()             │
│  - バッチリスト生成                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Step1: 階層分割（並列処理）          │
│  - 各バッチを並列で処理              │
│  - recursive_character_text_splitter│
│  - 結果: List[List[str]]            │
└─────────────────────────────────────┘
    ↓（全バッチ完了を待つ）
┌─────────────────────────────────────┐
│ Step2: 意味的分割（並列処理）        │
│  - 各paragraphを並列で処理           │
│  - semantic_chunking                │
│  - 結果: List[str]                  │
└─────────────────────────────────────┘
    ↓（全paragraph完了を待つ）
┌─────────────────────────────────────┐
│ Step3: オーバーラップ処理（並列処理）│
│  - 連続性チェックを並列実行          │
│  - check_continuity                 │
│  - 結果: List[str]                  │
└─────────────────────────────────────┘
    ↓
Final Result
```

### 4.2 Step1: 階層分割（並列化）

```python
async def process_step1_async(
    batches: List[str],
    model: str,
    config: AsyncConfig
) -> List[List[str]]:
    """
    各バッチを並列で階層分割
    """
    async def process_single_batch(batch_idx: int, batch_text: str):
        async with config.semaphore:
            print(f"[Step1] Processing batch {batch_idx+1}/{len(batches)}")
            result = await retry_async(
                recursive_character_text_splitter_async,
                batch_text,
                model,
                max_retries=config.max_retries
            )
            return (batch_idx, result or [])

    # 並列実行
    tasks = [
        process_single_batch(i, batch)
        for i, batch in enumerate(batches)
    ]
    results = await asyncio.gather(*tasks)

    # 順序を保持してマージ
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]
```

### 4.3 Step2: 意味的分割（並列化）

```python
async def process_step2_async(
    all_paragraphs: List[str],
    model: str,
    config: AsyncConfig
) -> List[str]:
    """
    各paragraphを並列で意味的分割
    """
    async def process_single_paragraph(p_idx: int, p_text: str):
        async with config.semaphore:
            result = await retry_async(
                semantic_chunking_async,
                p_text,
                model,
                max_retries=config.max_retries
            )
            return (p_idx, result or [p_text])

    # 並列実行
    tasks = [
        process_single_paragraph(i, p)
        for i, p in enumerate(all_paragraphs)
    ]
    results = await asyncio.gather(*tasks)

    # 順序を保持してフラット化
    results.sort(key=lambda x: x[0])
    flat_results = []
    for _, chunks in results:
        flat_results.extend(chunks)

    return flat_results
```

### 4.4 Step3: オーバーラップ処理（並列化）

```python
async def chunk_overlap_para(
    paragraphs: List[str],
    model: str,
    config: AsyncConfig
) -> List[str]:
    """
    連続性チェックを並列実行してオーバーラップ処理
    """
    if not paragraphs:
        return []

    # 全ての連続性チェックを並列実行
    async def check_single_continuity(idx: int):
        if idx == 0:
            return (idx, None)  # 最初のチャンクは前がない

        async with config.semaphore:
            prev_text = paragraphs[idx - 1]
            curr_text = paragraphs[idx]
            is_connected = await retry_async(
                check_continuity_async,
                prev_text,
                curr_text,
                model,
                max_retries=config.max_retries
            )
            return (idx, is_connected if is_connected is not None else False)

    # 並列で連続性チェック
    tasks = [check_single_continuity(i) for i in range(len(paragraphs))]
    continuity_results = await asyncio.gather(*tasks)
    continuity_results.sort(key=lambda x: x[0])

    # 結果に基づいてオーバーラップ処理
    overlapped_result = []
    for idx, (_, is_connected) in enumerate(continuity_results):
        current_text = paragraphs[idx]

        if idx == 0 or not is_connected:
            overlapped_result.append(current_text)
        else:
            prev_text = paragraphs[idx - 1]
            sentences = re.split(r'(?<=[。．！!？?])', prev_text)
            sentences = [s for s in sentences if s.strip()]
            overlap_part = sentences[-1] if sentences else prev_text
            overlapped_result.append(overlap_part + current_text)

    return overlapped_result
```

---

## 5. クラス実装

### 5.1 LargeTextProcessorPara

```python
class LargeTextProcessorPara:
    """大規模テキスト処理（並列版）"""

    def __init__(
        self,
        block_size: int = 2000,
        max_concurrent: int = 8,
        max_retries: int = 3
    ):
        self.block_size = block_size
        self.config = AsyncConfig(max_concurrent, max_retries)

    def split_into_batches(self, text: str) -> List[str]:
        """テキストをバッチに分割（同期処理）"""
        # 既存のロジックをそのまま使用
        pass

    async def process_async(
        self,
        text: str,
        model: str = "gemini-2.0-flash"
    ) -> List[str]:
        """
        非同期並列処理のメインエントリーポイント
        """
        # Step0: バッチ分割（同期）
        batches = self.split_into_batches(text)
        print(f"Total Batches: {len(batches)}")

        # Step1: 階層分割（並列）
        print("Starting Step1: Hierarchical splitting...")
        step1_results = await self.process_step1_async(batches, model)
        all_paragraphs = [p for batch in step1_results for p in batch]
        print(f"Step1 completed. Paragraphs: {len(all_paragraphs)}")

        # Step2: 意味的分割（並列）
        print("Starting Step2: Semantic chunking...")
        step2_results = await self.process_step2_async(all_paragraphs, model)
        print(f"Step2 completed. Chunks: {len(step2_results)}")

        # Step3: オーバーラップ（並列）
        print("Starting Step3: Overlap processing...")
        final_results = await chunk_overlap_para(
            step2_results,
            model,
            self.config
        )
        print(f"Step3 completed. Final chunks: {len(final_results)}")

        return final_results

    async def process_step1_async(
        self,
        batches: List[str],
        model: str
    ) -> List[List[str]]:
        """Step1の実装（上記参照）"""
        pass

    async def process_step2_async(
        self,
        paragraphs: List[str],
        model: str
    ) -> List[str]:
        """Step2の実装（上記参照）"""
        pass
```

---

## 6. コマンドライン引数

```python
def parse_args():
    parser = argparse.ArgumentParser(
        description='Gemini API並列テキスト処理'
    )
    parser.add_argument(
        '--parallel',
        type=int,
        default=8,
        help='並列実行数（デフォルト: 8）'
    )
    parser.add_argument(
        '--retries',
        type=int,
        default=3,
        help='リトライ回数（デフォルト: 3）'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='入力ファイルパス'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output.txt',
        help='出力ファイルパス'
    )
    return parser.parse_args()

# 使用例
# python csv_to_chunks_text_async.py --parallel 16 --input data.txt --output result.txt
```

---

## 7. エラーハンドリング戦略

### 7.1 リトライポリシー
- 初回失敗: 2秒待機
- 2回目失敗: 4秒待機
- 3回目失敗: 8秒待機
- 3回すべて失敗: Noneを返し、ログに記録

### 7.2 ログ記録
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)
```

### 7.3 部分失敗時の対応
- Step1で失敗したバッチ: 空のリストとして扱う
- Step2で失敗したparagraph: 元のテキストをそのまま保持
- Step3で失敗した連続性チェック: False（接続なし）として扱う

---

## 8. パフォーマンス見積もり

### 8.1 現状（同期処理）
- 処理時間: 約3時間
- API呼び出し: 逐次実行
- ボトルネック: API待機時間

### 8.2 改善後（並列処理）
- 並列数8の場合: 約20-30分（理論値: 22.5分）
- 並列数16の場合: 約15-20分
- 実効率: 70-80%（API制限、ネットワーク遅延考慮）

### 8.3 推奨設定
- **小規模（~10MB）**: `--parallel 4`
- **中規模（10-50MB）**: `--parallel 8`（デフォルト）
- **大規模（50MB+）**: `--parallel 16`

---

## 9. 実装の優先順位

### Phase 1: 基本非同期化（必須）
1. API呼び出し関数の非同期化
2. リトライ機構の実装
3. セマフォによる並列数制御

### Phase 2: 並列処理の実装（必須）
1. Step1の並列化
2. Step2の並列化
3. Step3の並列化

### Phase 3: 拡張機能（オプション）
1. 進捗表示の改善
2. 中断・再開機能
3. 結果のキャッシング

---

## 10. テスト戦略

### 10.1 単体テスト
- 各非同期関数の動作確認
- リトライ機構のテスト
- エラーハンドリングのテスト

### 10.2 統合テスト
- 小規模テキストでの全体動作確認
- 並列数を変えた性能比較
- エラー発生時の挙動確認

### 10.3 負荷テスト
- 大規模テキストでの処理時間計測
- メモリ使用量の監視
- API制限との兼ね合い確認

---

## 11. 注意事項・制約

### 11.1 API制限
- Gemini APIのレート制限を確認
- 並列数が多すぎると429エラーの可能性
- 推奨: 初回は`--parallel 4`でテスト

### 11.2 メモリ使用量
- 大量のタスクを同時に生成するとメモリ圧迫
- 必要に応じてバッチ処理を導入

### 11.3 順序保証
- Step1とStep2は元の順序を保持する必要あり
- タプル `(index, result)` で管理

---

## 付録: 実装チェックリスト

- [ ] `recursive_character_text_splitter_async()` 実装
- [ ] `semantic_chunking_async()` 実装
- [ ] `check_continuity_async()` 実装
- [ ] `retry_async()` 実装
- [ ] `AsyncConfig` クラス実装
- [ ] `LargeTextProcessorPara` クラス実装
- [ ] `chunk_overlap_para()` 実装
- [ ] コマンドライン引数パーサー実装
- [ ] ログ機構の実装
- [ ] 単体テスト作成
- [ ] 統合テスト実行
- [ ] パフォーマンス計測