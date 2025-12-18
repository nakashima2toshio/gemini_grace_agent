# celery_tasks.py - Celery非同期タスク定義

## 概要

Q/Aペア生成の並列処理のためのCeleryタスク定義モジュール。
OpenAI APIを使用してテキストチャンクからQ/Aペアを非同期に生成する。

## 依存関係

```python
from celery import Celery
from openai import OpenAI
from models import QAPairsResponse  # Pydanticモデル
from config import ModelConfig, CeleryConfig
```

## Celeryアプリケーション設定

### 基本設定

```python
app = Celery(
    'qa_generation',
    broker=os.getenv('CELERY_BROKER_URL', CeleryConfig.BROKER_URL),
    backend=os.getenv('CELERY_RESULT_BACKEND', CeleryConfig.RESULT_BACKEND)
)
```

### 設定パラメータ

| 設定項目 | 説明 |
|---------|------|
| `task_serializer` | タスクシリアライザ（CeleryConfig参照） |
| `task_time_limit` | タスクのハードタイムアウト |
| `task_soft_time_limit` | タスクのソフトタイムアウト |
| `worker_concurrency` | ワーカーの並列度 |
| `worker_prefetch_multiplier` | プリフェッチ数 |
| `task_acks_late` | 完了後にACKを送信（True） |
| `task_reject_on_worker_lost` | ワーカー喪失時にリジェクト（True） |

## ヘルパー関数

### supports_temperature(model: str) -> bool

モデルがtemperatureパラメータをサポートするかチェック。

```python
# GPT-5シリーズ、O-Seriesはtemperature=1のみサポート
if supports_temperature(model):
    completion_params["temperature"] = 0.7
```

**委譲先**: `ModelConfig.supports_temperature(model)`

---

### determine_qa_count(chunk_data: Dict, config: Dict) -> int

チャンクのトークン数に基づいてQ/A数を決定。

**引数**:
- `chunk_data`: チャンクデータ（`tokens`キーを含む）
- `config`: 設定情報（`qa_per_chunk`キーを含む）

**戻り値**: 生成するQ/A数

**ロジック**:
| トークン数 | Q/A数 |
|-----------|-------|
| < 50 | base_qa_count - 1（最小1） |
| 50-150 | base_qa_count |
| > 150 | base_qa_count + 1 |

---

### _extract_parsed_response(response, model: str) -> QAPairsResponse

`responses.parse()` APIのレスポンスから解析済みデータを抽出。

**引数**:
- `response`: OpenAI APIレスポンス
- `model`: 使用したモデル名

**戻り値**: `QAPairsResponse`オブジェクト

**例外**: `ValueError` - 解析可能なレスポンスがない場合

**解析方法（優先順）**:

1. **方法1**: `output_parsed`属性（GPT-5シリーズ対応）
2. **方法2**: `output`配列から探索（GPT-4o対応）
3. **方法3**: `text`属性からJSON直接解析
4. **方法4**: `output_text`属性からJSON解析

## Celeryタスク

### generate_qa_for_chunk_async

単一チャンクからQ/Aペアを非同期生成。

```python
@app.task(bind=True, max_retries=3)
def generate_qa_for_chunk_async(
    self,
    chunk_data: Dict,
    config: Dict,
    model: str = "gpt-4o-mini"
) -> Dict
```

**引数**:
- `chunk_data`: チャンクデータ
  - `id`: チャンクID
  - `text`: テキスト内容
  - `tokens`: トークン数
  - `doc_id`: ドキュメントID
  - `dataset_type`: データセットタイプ
  - `chunk_idx`: チャンクインデックス
- `config`: データセット設定
  - `lang`: 言語（"ja" または "en"）
  - `qa_per_chunk`: チャンクあたりのQ/A数
- `model`: 使用するモデル（デフォルト: "gpt-4o-mini"）

**戻り値**:
```python
{
    "success": True,           # 成功フラグ
    "chunk_id": "...",         # チャンクID
    "qa_pairs": [...],         # 生成されたQ/Aペア
    "error": None              # エラーメッセージ（失敗時）
}
```

**OpenAI API呼び出し**:

1. **主要API**: `client.responses.parse()` - 構造化出力API
   ```python
   response = client.responses.parse(
       input=combined_input,
       model=model,
       text_format=QAPairsResponse,
       max_output_tokens=2000
   )
   ```

2. **フォールバック**: `client.chat.completions.create()` - Chat Completions API
   - GPT-5シリーズ/O3/O4: `max_completion_tokens`を使用
   - その他: `max_tokens`を使用

**リトライ設定**: 最大3回、待機時間は5秒 × (リトライ回数 + 1)

---

### generate_qa_for_batch_async

複数チャンクからQ/Aペアを非同期バッチ生成。

```python
@app.task(bind=True, max_retries=3)
def generate_qa_for_batch_async(
    self,
    chunks: List[Dict],
    config: Dict,
    model: str = "gpt-4o-mini"
) -> Dict
```

**引数**:
- `chunks`: チャンクデータのリスト（1-5個）
- `config`: データセット設定
- `model`: 使用するモデル

**戻り値**:
```python
{
    "success": True,
    "chunk_ids": [...],        # チャンクIDリスト
    "qa_pairs": [...],         # 生成されたQ/Aペア
    "error": None
}
```

**特徴**:
- 単一チャンクの場合は`generate_qa_for_chunk_async`に委譲
- チャンクテキストは1000文字で切り詰め
- `max_output_tokens=4000`（バッチ処理用に増加）

## ユーティリティ関数

### submit_parallel_qa_generation

並列Q/A生成ジョブを投入。

```python
def submit_parallel_qa_generation(
    chunks: List[Dict],
    config: Dict,
    model: str = "gpt-4o-mini",
    batch_size: int = 3
) -> List
```

**引数**:
- `chunks`: チャンクのリスト
- `config`: データセット設定
- `model`: 使用するモデル
- `batch_size`: バッチサイズ（1-5）

**戻り値**: Celeryタスクのリスト（`AsyncResult`オブジェクト）

**処理フロー**:
- `batch_size > 1`: `generate_qa_for_batch_async`を使用
- `batch_size == 1`: `generate_qa_for_chunk_async`を使用
- キュー: `qa_generation`

---

### collect_results

並列処理の結果を収集（Redis直接アクセスによる確実な取得）。

```python
def collect_results(
    tasks: List,
    timeout: int = 300
) -> List[Dict]
```

**引数**:
- `tasks`: Celeryタスクのリスト
- `timeout`: タイムアウト（秒、デフォルト: 300秒）

**戻り値**: Q/Aペアのリスト

**機能**:
- Redis直接アクセスによる確実な結果取得
- 5秒ごとの進捗表示
- 30秒ごとの停滞チェック
- 3分間進捗なしの場合、状態を強制リフレッシュ
- ループ終了後の最終確認処理

**結果サマリー出力**:
```
=====================================
結果収集完了:
- 成功: X/Yタスク
- 失敗: Zタスク
- 未完了: Wタスク
- 生成Q/Aペア: N個
- 所要時間: T秒
=====================================
```

## Q/Aペア出力形式

生成されるQ/Aペアの形式:

```python
{
    "question": "質問文",
    "answer": "回答文",
    "question_type": "fact|reason|comparison|application",
    "source_chunk_id": "チャンクID",
    "doc_id": "ドキュメントID",
    "dataset_type": "データセットタイプ",
    "chunk_idx": 0
}
```

**質問タイプ**:
| タイプ | 日本語 | 英語 |
|--------|--------|------|
| fact | 事実確認型（〜は何ですか？） | What is...? |
| reason | 理由説明型（なぜ〜ですか？） | Why...? |
| comparison | 比較型（〜と〜の違いは？） | What's the difference...? |
| application | 応用型（〜はどのように活用されますか？） | How is... used? |

## 使用例

### Celeryワーカー起動

```bash
celery -A celery_tasks worker --loglevel=info --concurrency=4 -Q qa_generation
```

### タスク投入と結果収集

```python
from celery_tasks import submit_parallel_qa_generation, collect_results

# チャンクデータの準備
chunks = [
    {"id": "chunk_1", "text": "テキスト1", "tokens": 100, "doc_id": "doc_1"},
    {"id": "chunk_2", "text": "テキスト2", "tokens": 120, "doc_id": "doc_1"},
]

config = {
    "lang": "ja",
    "qa_per_chunk": 2
}

# タスク投入
tasks = submit_parallel_qa_generation(chunks, config, model="gpt-4o-mini", batch_size=3)

# 結果収集
qa_pairs = collect_results(tasks, timeout=300)

print(f"生成されたQ/Aペア数: {len(qa_pairs)}")
```

## 環境変数

| 変数名 | 説明 | デフォルト |
|--------|------|-----------|
| `CELERY_BROKER_URL` | Celeryブローカー URL | CeleryConfig.BROKER_URL |
| `CELERY_RESULT_BACKEND` | 結果バックエンド URL | CeleryConfig.RESULT_BACKEND |
| `OPENAI_API_KEY` | OpenAI APIキー | 必須 |
| `REDIS_HOST` | Redisホスト | localhost |
| `REDIS_PORT` | Redisポート | 6379 |
| `REDIS_DB` | Redisデータベース番号 | 0 |

## 注意事項

1. **モデル名**: config.pyで定義されたモデル名をそのまま使用すること。マッピングを作成しないこと。

2. **API選択**:
   - 構造化出力が必要: `client.responses.parse()`
   - 通常のテキスト生成: `client.responses.create()`

3. **temperatureパラメータ**: GPT-5シリーズ、O-Seriesはtemperature=1のみサポート。`supports_temperature()`で確認すること。

4. **タイムアウト**: 長時間実行タスクでは適切なタイムアウト値を設定すること（デフォルト: 300秒）。

## 関連ファイル

- `models.py`: `QAPairsResponse` Pydanticモデル定義
- `config.py`: `ModelConfig`, `CeleryConfig` 設定クラス
- `helper_api.py`: OpenAI API統合ヘルパー