# make_qa_register_qdrant.py 完全ガイド

## 📋 目次

1. [概要](#概要)
2. [システムアーキテクチャ](#システムアーキテクチャ)
3. [データ処理フロー](#データ処理フロー)
4. [機能詳細](#機能詳細)
5. [入力形式と対応](#入力形式と対応)
6. [使用方法（パターン別）](#使用方法パターン別)
7. [コマンドラインオプション](#コマンドラインオプション)
8. [実行例とワークフロー](#実行例とワークフロー)
9. [関数リファレンス](#関数リファレンス)
10. [トラブルシューティング](#トラブルシューティング)
11. [ベストプラクティス](#ベストプラクティス)

---

## 📖 概要

`make_qa_register_qdrant.py` は、**Q/Aペアの自動生成からQdrantベクトルデータベースへの登録までを一貫して実行する統合CLIツール**です。

### ファイル情報

- **ファイル名**: `make_qa_register_qdrant.py`
- **説明**: Q/A生成からQdrant登録までを完結する統合ツール（改修版）
- **改修内容**: チャンクCSV読み込み機能を追加
- **バージョン**: 最新版（2025-01-17更新）

### 主な特徴

✅ **3つの入力形式に対応**

- **事前定義データセット**（`config.py`で定義）
- **テキスト/CSV形式の生データ**
- **チャンクCSV形式**（事前作成済みチャンクの再利用） ← 新機能

✅ **2フェーズの自動実行**

- **Phase 1**: Q/Aペア生成（`QAPipeline`を使用）
- **Phase 2**: Qdrant登録（Embedding + インデックス化）

✅ **高速並列処理**

- Celeryによる非同期タスク実行
- 最大24ワーカーでの並列処理に対応

✅ **柔軟な処理制御**

- Q/Aペアが既に存在する場合は生成をスキップ
- チャンクが既に作成済みの場合は再利用
- 段階的な処理の確認・デバッグが可能

✅ **UI統合対応**

- ファイル名の正規化（タイムスタンプ除去）
- `agent_rag.py`での参照を容易にする

---

## 🏗️ システムアーキテクチャ

### コンポーネント構成

```
┌─────────────────────────────────────────────────────────────┐
│          make_qa_register_qdrant.py (統合CLIツール)           │
│                                                             │
│  ┌──────────────────────┐      ┌──────────────────────┐    │
│  │   Phase 1: QA生成     │      │   Phase 2: Qdrant登録 │    │
│  │                      │      │                      │    │
│  │  QAPipeline          │      │  Qdrant Service      │    │
│  │  ├─ データ読み込み     │       │  ├─ Embedding生成    │    │
│  │  ├─ チャンク作成       │      │  ├─ ポイント構築       │    │
│  │  ├─ Q/A生成(LLM)      │       │  └─ バッチ登録         │    │
│  │  └─ CSV出力           │       │                      │    │
│  └──────────────────────┘      └──────────────────────┘    │
│           │                              │                  │
│           v                              v                  │
│  qa_pairs_*.csv                   Qdrantコレクション         │
└─────────────────────────────────────────────────────────────┘
```

### 依存モジュール

```
make_qa_register_qdrant.py
├── qa_generation/
│   ├── pipeline.py           # メインパイプライン制御
│   ├── generation.py         # LLMによるQ/A生成
│   ├── structure.py          # チャンク作成ロジック
│   ├── semantic.py           # セマンティック処理
│   ├── evaluation.py         # カバレージ分析
│   └── data_io.py            # データ入出力
│
├── services/
│   └── qdrant_service.py     # Qdrant操作（CRUD）
│       ├── create_or_recreate_collection_for_qdrant()
│       ├── embed_texts_for_qdrant()
│       ├── build_points_for_qdrant()
│       └── upsert_points_to_qdrant()
│
├── qdrant_client_wrapper.py  # Qdrantクライアント初期化
│   └── create_qdrant_client()
│
└── config.py                 # 設定（データセット定義等）
    └── DATASET_CONFIGS
```

### レイヤー構成

```
┌──────────────────────────────────────────┐
│  CLI Layer (make_qa_register_qdrant.py)   │  ← ユーザーインターフェース
├───────────────────────────────────────────┤
│  Business Logic Layer                     │
│  ├─ QAPipeline (qa_generation/)           │  ← Q/A生成ロジック
│  └─ QdrantService (services/)             │  ← Qdrant操作ロジック
├───────────────────────────────────────────┤
│  Infrastructure Layer                     │
│  ├─ Google Gemini API (LLM)               │  ← 外部API
│  ├─ Qdrant Vector DB                      │  ← ベクトルDB
│  └─ Celery (タスクキュー)                   │  ← 並列処理基盤
└───────────────────────────────────────────┘
```

---

## 🔄 データ処理フロー

### 全体フロー図

```
入力データ
   │
   ├─ --dataset         → Hugging Face等のデータセット
   ├─ --input-csv       → テキストCSV or Q/AペアCSV
   └─ --input-chunks    → チャンクCSV（事前作成済み）✨ NEW
   │
   v
┌──────────────────────────────────────────────┐
│ Phase 1: Q/Aペア生成 (QAPipeline)             │
├──────────────────────────────────────────────┤
│                                              │
│  ┌─────────────────────────────────┐         │
│  │ 1. データソース判定               │         │
│  │   - input_chunks?               │         │
│  │   - input_csv?                  │         │
│  │   - dataset?                    │         │
│  └──────────┬──────────────────────┘         │
│             │                                │
│             v                                │
│  ┌─────────────────────────────────┐         │
│  │ 2. チャンク作成/読み込み          │         │
│  │   ・テキスト分割                 │         │
│  │   ・トークン数計算                │         │
│  │   ・メタデータ付与                │         │
│  │   ・チャンクCSV読み込み（NEW）     │         │
│  └──────────┬──────────────────────┘         │
│             │                                │
│             v                                │
│  ┌─────────────────────────────────┐         │
│  │ 3. チャンクの前処理              │         │
│  │   ・マージ（小さいチャンク統合）   │         │
│  │   ・バッチ化（API呼び出し最適化）  │         │
│  └──────────┬──────────────────────┘         │
│             │                                │
│             v                                │
│  ┌─────────────────────────────────┐         │
│  │ 4. Q/A生成（LLM呼び出し）         │         │
│  │   ・Gemini API呼び出し            │         │
│  │   ・プロンプト生成                │         │
│  │   ・応答のパース                  │         │
│  │   ・Celery並列処理（オプション）   │         │
│  └──────────┬──────────────────────┘         │
│             │                                │
│             v                                │
│  ┌─────────────────────────────────┐         │
│  │ 5. CSV出力                       │         │
│  │   qa_pairs_{dataset}_{ts}.csv   │         │
│  └─────────────────────────────────┘         │
│                                              │
└──────────────┬───────────────────────────────┘
               │
               v
       qa_pairs_{dataset}_{timestamp}.csv
               │
               v
┌──────────────────────────────────────────────┐
│ Phase 2: Qdrant登録                          │
├──────────────────────────────────────────────┤
│                                              │
│  ┌─────────────────────────────────┐         │
│  │ 1. CSV読み込み                   │         │
│  │   ・必須カラム確認                │         │
│  │   ・データ検証                    │         │
│  └──────────┬──────────────────────┘         │
│             │                                │
│             v                                │
│  ┌─────────────────────────────────┐         │
│  │ 2. ベクトル化テキスト準備         │         │
│  │   question + "\n" + answer       │         │
│  └──────────┬──────────────────────┘         │
│             │                                │
│             v                                │
│  ┌─────────────────────────────────┐         │
│  │ 3. Qdrantコレクション準備         │         │
│  │   ・コレクション作成/再作成       │         │
│  │   ・ベクトル次元設定（768次元）   │         │
│  │   ・距離メトリック設定（Cosine）  │         │
│  └──────────┬──────────────────────┘         │
│             │                                │
│             v                                │
│  ┌─────────────────────────────────┐         │
│  │ 4. バッチ処理ループ              │         │
│  │   ┌───────────────────┐          │         │
│  │   │ Batch 1: 0-99     │          │         │
│  │   ├───────────────────┤          │         │
│  │   │ Batch 2: 100-199  │          │         │
│  │   ├───────────────────┤          │         │
│  │   │ Batch 3: 200-299  │          │         │
│  │   └───────────────────┘          │         │
│  │                                  │         │
│  │   各バッチで:                     │         │
│  │   ├─ Embedding生成 (Gemini)      │         │
│  │   ├─ ポイント構築（メタデータ付与）│         │
│  │   └─ Qdrantへアップサート         │         │
│  └──────────┬──────────────────────┘         │
│             │                                │
│             v                                │
│  ┌─────────────────────────────────┐         │
│  │ 5. UI用正規化CSV作成             │         │
│  │   ・ファイル名から日時除去        │         │
│  │   ・qa_output/ に保存            │         │
│  │   → agent_rag.pyで参照           │         │
│  └─────────────────────────────────┘         │
│                                              │
└──────────────┬───────────────────────────────┘
               │
               v
      Qdrantコレクション
      ├─ ベクトルインデックス（768次元）
      ├─ メタデータ（question, answer, source等）
      └─ 高速検索が可能
```

### 入力判定ロジック

```python
# 入力ソースの優先順位と処理分岐

if args.input_chunks:
    # ========================================
    # パターンA: チャンクCSV → Q/A生成 → Qdrant
    # ========================================
    logger.info(f"📁 チャンクCSVを使用: {args.input_chunks}")

    pipeline = QAPipeline(
        input_chunks=args.input_chunks,  # ✅ 新機能
        model=args.model,
        max_docs=args.max_docs
    )

    result = pipeline.run(
        use_celery=args.use_celery,
        celery_workers=args.celery_workers,
        batch_chunks=args.batch_chunks,
        merge_chunks=args.merge_chunks,
        analyze_coverage=True
    )

    処理 = [
        "チャンク読み込み（作成スキップ）",
        "Q/A生成",
        "Qdrant登録"
    ]

elif args.input_csv:
    # CSVの中身を確認
    df = pd.read_csv(args.input_csv)

    if 'question' in df.columns and 'answer' in df.columns:
        # ====================================
        # パターンB: Q/AペアCSV → Qdrant
        # ====================================
        logger.info("✅ Q/Aカラムが存在します - Q/A生成をスキップして登録へ")

        generated_csv = args.input_csv
        qa_count = len(df)

        処理 = [
            "Q/A生成スキップ",
            "Qdrant登録のみ"
        ]

    elif 'text' in df.columns or 'Combined_Text' in df.columns:
        # ====================================
        # パターンC: テキストCSV → チャンク作成 → Q/A生成 → Qdrant
        # ====================================
        logger.info("📝 テキストカラムのみ検出 - Q/A生成を実行します")

        pipeline = QAPipeline(
            input_file=args.input_csv,
            model=args.model,
            max_docs=args.max_docs
        )

        result = pipeline.run(
            use_celery=args.use_celery,
            celery_workers=args.celery_workers,
            batch_chunks=args.batch_chunks,
            merge_chunks=args.merge_chunks,
            analyze_coverage=True,
            overlap_tokens=args.overlap_tokens,
            use_similarity=args.use_similarity,
            similarity_threshold=args.similarity_threshold
        )

        処理 = [
            "チャンク作成",
            "Q/A生成",
            "Qdrant登録"
        ]

    else:
        logger.error("❌ CSVファイルに必要なカラムが見つかりません")
        logger.error("   必要なカラム: (question + answer) または (text または Combined_Text)")
        sys.exit(1)

elif args.dataset:
    # ========================================
    # パターンD: データセット → チャンク作成 → Q/A生成 → Qdrant
    # ========================================
    pipeline = QAPipeline(
        dataset_name=args.dataset,
        model=args.model,
        max_docs=args.max_docs
    )

    result = pipeline.run(
        use_celery=args.use_celery,
        celery_workers=args.celery_workers,
        batch_chunks=args.batch_chunks,
        merge_chunks=args.merge_chunks,
        analyze_coverage=True,
        overlap_tokens=args.overlap_tokens,
        use_similarity=args.use_similarity,
        similarity_threshold=args.similarity_threshold
    )

    処理 = [
        "データセット読み込み",
        "チャンク作成",
        "Q/A生成",
        "Qdrant登録"
    ]
```

---

## 🔧 機能詳細

### 1. normalize_source_filename() 関数

**目的**: UI（agent_rag.py）での参照を安定させるため、ファイル名からタイムスタンプを除去

```python
def normalize_source_filename(filename: str) -> str:
    """
    ファイル名から日時サフィックス（例: _20251230_232641）を除去して正規化する。
    UI(agent_rag.py)での参照を安定させるための処理。
    """
    normalized = re.sub(r'_\d{8}_\d{6}', '', filename)
    return normalized
```

**処理例**:

```python
# 入力
filename = "qa_pairs_wikipedia_ja_20250117_143025.csv"

# 出力
normalized = "qa_pairs_wikipedia_ja.csv"
```

**使用箇所**:

- Phase 2のQdrant登録時
- ペイロードの`source`フィールドに正規化名を設定
- UI用CSVの保存時

---

### 2. run_registration() 関数

**目的**: Phase 2のQdrant登録処理を実行

**処理フロー**:

```
1. CSV読み込み
   ↓
2. 必須カラム確認（question, answer）
   ↓
3. ベクトル化テキスト準備（question + "\n" + answer）
   ↓
4. Qdrantクライアント作成
   ↓
5. コレクション作成/再作成
   ↓
6. バッチ処理ループ
   ├─ Embedding生成
   ├─ ポイント構築
   └─ Qdrantへアップサート
   ↓
7. UI用正規化CSV作成
```

**パラメータ**:

```python
def run_registration(
    csv_path: str,          # Q/AペアCSVのパス
    collection_name: str,   # Qdrantコレクション名
    recreate: bool,         # コレクションを再作成するか
    batch_size: int,        # バッチサイズ（デフォルト: 100）
    provider: str           # Embeddingプロバイダー（デフォルト: "gemini"）
) -> bool:                  # 成功時True、失敗時False
```

**実装詳細**:

```python
def run_registration(csv_path: str, collection_name: str, recreate: bool,
                     batch_size: int, provider: str):
    logger.info(f"\n" + "=" * 60)
    logger.info(f"Phase 2: Qdrant Registration")
    logger.info(f"=" * 60)

    # 1. CSV読み込み
    df = pd.read_csv(csv_path)

    # 2. ベクトル化対象テキスト準備
    texts = (df['question'].astype(str) + "\n" + df['answer'].astype(str)).tolist()

    # 3. Qdrantクライアント作成
    client = create_qdrant_client()

    # 4. コレクション作成/再作成
    create_or_recreate_collection_for_qdrant(client, collection_name, recreate=recreate)

    # 5. ファイル名正規化
    source_filename = os.path.basename(csv_path)
    normalized_filename = normalize_source_filename(source_filename)

    # 6. バッチ処理
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]

        # Embedding生成
        vectors = embed_texts_for_qdrant(batch_texts)

        # ポイント構築
        points = build_points_for_qdrant(
            batch_df,
            vectors,
            domain=collection_name,
            source_file=normalized_filename,
            start_index=i  # グローバルインデックス
        )

        # source情報を確実に正規化名で設定
        for point in points:
            point.payload["source"] = normalized_filename

        # Qdrantへアップサート
        upsert_points_to_qdrant(client, collection_name, points)

    # 7. UI用正規化CSV作成
    output_path = os.path.join("qa_output", normalized_filename)
    df[['question', 'answer']].to_csv(output_path, index=False, encoding='utf-8')

    return True
```

---

### 3. main() 関数

**目的**: CLIエントリーポイント、引数解析と処理実行

**処理フロー**:

```
1. 引数解析
   ↓
2. 入力検証（排他制御）
   ↓
3. APIキー確認
   ↓
4. Phase 1実行（Q/A生成）
   ↓
5. Phase 2実行（Qdrant登録）
   ↓
6. 完了メッセージ
```

**引数グループ**:

1. **Input Source Options** (排他的)

   - `--dataset`
   - `--input-csv`
   - `--input-chunks` ✨ NEW
2. **QA Generation Options**

   - `--model`
   - `--max-docs`
   - `--use-celery`
   - `--celery-workers`
   - `--batch-chunks`
   - `--merge-chunks`
   - `--overlap-tokens`
   - `--use-similarity`
   - `--similarity-threshold`
3. **Qdrant Registration Options**

   - `--collection` (必須)
   - `--recreate`
   - `--batch-size`
   - `--provider`

**入力検証ロジック**:

```python
# 排他制御（いずれか1つのみ）
input_count = sum([
    args.dataset is not None,
    args.input_csv is not None,
    args.input_chunks is not None  # ✅ 新規追加
])

if input_count == 0:
    logger.error("--dataset, --input-csv, --input-chunks のいずれか1つを指定してください")
    sys.exit(1)

if input_count > 1:
    logger.error("--dataset, --input-csv, --input-chunks は同時に指定できません")
    sys.exit(1)
```

---

## 📥 入力形式と対応

### 入力形式マトリクス


| 入力オプション           | 形式     | 必須カラム                                | チャンク作成 | Q/A生成 | Qdrant登録 |
| ------------------------ | -------- | ----------------------------------------- | ------------ | ------- | ---------- |
| `--dataset`              | 事前定義 | -                                         | ✅           | ✅      | ✅         |
| `--input-csv` (テキスト) | CSV      | `text` or `Combined_Text`                 | ✅           | ✅      | ✅         |
| `--input-csv` (Q/Aペア)  | CSV      | `question`, `answer`                      | ❌           | ❌      | ✅         |
| `--input-chunks`         | CSV      | `chunk_id`, `text`, `tokens`, `chunk_idx` | ❌           | ✅      | ✅         |

### パターン別詳細

#### パターンA: チャンクCSVから（NEW）

```bash
python qa_qdrant/make_qa_register_qdrant.py \
  --input-chunks wiki_chunks.csv \
  --collection qa_wikipedia \
  --use-celery \
  --celery-workers 16 \
  --recreate
```

**必須カラム**:

```csv
chunk_id,text,tokens,chunk_idx,dataset_type
chunk_0,"テキスト...",150,0,wikipedia_ja
chunk_1,"テキスト...",200,1,wikipedia_ja
```

**処理**:

1. チャンクCSV読み込み ← チャンク作成スキップ
2. Q/A生成
3. Qdrant登録

**メリット**:

- チャンク作成時間の節約
- チャンク品質の事前確認・調整が可能
- 再現性の向上

---

#### パターンB: Q/AペアCSVから（生成スキップ）

```bash
python qa_qdrant/make_qa_register_qdrant.py \
  --input-csv qa_pairs.csv \
  --collection my_qa \
  --batch-size 100 \
  --recreate
```

**必須カラム**:

```csv
question,answer
"質問1","回答1"
"質問2","回答2"
```

**処理**:

1. Q/A生成スキップ
2. Qdrant登録のみ

**メリット**:

- 既存のQ/Aペアを直接登録
- 処理時間の大幅短縮

---

#### パターンC: テキストCSVから

```bash
python qa_qdrant/make_qa_register_qdrant.py \
  --input-csv data.csv \
  --collection my_collection \
  --use-celery \
  --celery-workers 16 \
  --merge-chunks \
  --recreate
```

**必須カラム**:

```csv
text
"長文テキスト1..."
"長文テキスト2..."
```

または

```csv
Combined_Text
"長文テキスト1..."
"長文テキスト2..."
```

**処理**:

1. チャンク作成
2. Q/A生成
3. Qdrant登録

---

#### パターンD: データセットから

```bash
python qa_qdrant/make_qa_register_qdrant.py \
  --dataset wikipedia_ja \
  --collection qa_wikipedia_ja \
  --use-celery \
  --celery-workers 24 \
  --recreate
```

**データセット定義** (config.py):

```python
DATASET_CONFIGS = {
    "wikipedia_ja": {
        "name": "Wikipedia日本語",
        "file": "data/wikipedia_ja.csv",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "ja",
        "chunk_size": 300,
        "qa_per_chunk": 3
    }
}
```

**処理**:

1. データセット読み込み
2. チャンク作成
3. Q/A生成
4. Qdrant登録

---

## 🚀 使用方法（パターン別）

### パターン1: チャンクCSVからQ/A生成 + Qdrant登録（推奨）

**2段階ワークフロー**:

```bash
# Step 1: チャンク作成（別ツール）
python -m chunking.csv_to_chunks_text_para \
  -i wiki_data.txt \
  -o wiki_chunks.csv \
  -w 8

# Step 2: Q/A生成 + Qdrant登録
python qa_qdrant/make_qa_register_qdrant.py \
  --input-chunks wiki_chunks.csv \
  --collection qa_wikipedia \
  --use-celery \
  --celery-workers 16 \
  --merge-chunks \
  --recreate
```

**メリット**:

- チャンク品質を事前確認可能
- チャンク作成とQ/A生成を分離
- デバッグしやすい

---

### パターン2: データセットから一貫処理

```bash
python qa_qdrant/make_qa_register_qdrant.py \
  --dataset wikipedia_ja \
  --collection qa_wikipedia_ja \
  --use-celery \
  --celery-workers 24 \
  --recreate
```

**メリット**:

- 1コマンドで完結
- 設定が`config.py`に集約

---

### パターン3: Q/AペアCSVから直接登録

```bash
# 既存のQ/AペアCSVを登録
python qa_qdrant/make_qa_register_qdrant.py \
  --input-csv qa_output/pipeline/qa_pairs_wiki.csv \
  --collection qa_wikipedia \
  --batch-size 100 \
  --recreate
```

**メリット**:

- Phase 1をスキップ
- 最も高速

---

### パターン4: テキストCSVからフル実行

```bash
python qa_qdrant/make_qa_register_qdrant.py \
  --input-csv raw_data.csv \
  --collection my_collection \
  --use-celery \
  --celery-workers 16 \
  --overlap-tokens 50 \
  --use-similarity \
  --similarity-threshold 0.7 \
  --recreate
```

**メリット**:

- 生データから直接処理
- 高度なチャンク作成オプション使用可能

---

## 🎛️ コマンドラインオプション

### 入力ソースオプション（排他的、いずれか1つ必須）


| オプション       | 型  | 説明                         | 例             |
| ---------------- | --- | ---------------------------- | -------------- |
| `--dataset`      | str | 事前定義データセット名       | `wikipedia_ja` |
| `--input-csv`    | str | 入力CSVファイルのパス        | `data.csv`     |
| `--input-chunks` | str | チャンクCSVファイルのパス ✨ | `chunks.csv`   |

---

### Q/A生成オプション


| オプション               | 型    | デフォルト         | 説明                                 |
| ------------------------ | ----- | ------------------ | ------------------------------------ |
| `--model`                | str   | `gemini-2.0-flash` | 使用するGeminiモデル                 |
| `--max-docs`             | int   | `None`             | 処理する最大文書数（デバッグ用）     |
| `--use-celery`           | flag  | False              | Celery並列処理を使用                 |
| `--celery-workers`       | int   | 8                  | Celeryワーカー数                     |
| `--batch-chunks`         | int   | 3                  | 1回のAPI呼び出しで処理するチャンク数 |
| `--merge-chunks`         | flag  | True               | 小さいチャンクを統合                 |
| `--overlap-tokens`       | int   | 0                  | チャンク間の重複トークン数           |
| `--use-similarity`       | flag  | False              | ベクトル類似度分割を使用             |
| `--similarity-threshold` | float | 0.7                | 類似度分割の閾値                     |

---

### Qdrant登録オプション


| オプション     | 型   | デフォルト | 必須 | 説明                       |
| -------------- | ---- | ---------- | ---- | -------------------------- |
| `--collection` | str  | -          | ✅   | 登録先Qdrantコレクション名 |
| `--recreate`   | flag | False      | -    | コレクションを再作成       |
| `--batch-size` | int  | 100        | -    | Embeddingバッチサイズ      |
| `--provider`   | str  | `gemini`   | -    | Embeddingプロバイダー      |

---

## 📝 実行例とワークフロー

### ワークフロー1: 小規模テスト（10文書）

```bash
# チャンク作成
python -m chunking.csv_to_chunks_text_para \
  -i test_data.txt \
  -o test_chunks.csv \
  --max-rows 10

# Q/A生成 + Qdrant登録（Celeryなし）
python qa_qdrant/make_qa_register_qdrant.py \
  --input-chunks test_chunks.csv \
  --collection test_collection \
  --batch-size 10 \
  --recreate

# 確認
curl http://localhost:6333/collections/test_collection
```

**所要時間**: 約3-5分

---

### ワークフロー2: 中規模処理（1,000文書）

```bash
# Celeryワーカー起動
./start_celery.sh restart -w 16

# チャンク作成
python -m chunking.csv_to_chunks_text_para \
  -i data.txt \
  -o chunks.csv \
  -w 16

# Q/A生成 + Qdrant登録
python qa_qdrant/make_qa_register_qdrant.py \
  --input-chunks chunks.csv \
  --collection my_collection \
  --use-celery \
  --celery-workers 16 \
  --merge-chunks \
  --batch-size 100 \
  --recreate
```

**所要時間**: 約30-60分

---

### ワークフロー3: 大規模処理（10,000文書以上）

```bash
# Celeryワーカー起動（最大並列）
./start_celery.sh restart -w 24

# チャンク作成
python -m chunking.csv_to_chunks_text_para \
  -i large_data.txt \
  -o chunks.csv \
  -w 24

# Q/A生成 + Qdrant登録
python qa_qdrant/make_qa_register_qdrant.py \
  --input-chunks chunks.csv \
  --collection large_collection \
  --use-celery \
  --celery-workers 24 \
  --merge-chunks \
  --batch-size 200 \
  --recreate

# Flower（モニタリング）
celery -A celery_config flower --port=5555
# http://localhost:5555 でアクセス
```

**所要時間**: 数時間～数日

---

### ワークフロー4: 既存Q/AペアをQdrantに登録

```bash
# Q/Aペアが既に存在する場合
python qa_qdrant/make_qa_register_qdrant.py \
  --input-csv qa_output/pipeline/qa_pairs_wiki_20250117.csv \
  --collection qa_wikipedia \
  --batch-size 100 \
  --recreate
```

**所要時間**: 数分～数十分

---

## 🔬 関数リファレンス

### normalize_source_filename()

```python
def normalize_source_filename(filename: str) -> str:
    """
    ファイル名から日時サフィックスを除去して正規化する。

    Args:
        filename (str): 正規化対象のファイル名

    Returns:
        str: 正規化されたファイル名

    Examples:
        >>> normalize_source_filename("qa_pairs_wiki_20250117_143025.csv")
        "qa_pairs_wiki.csv"

        >>> normalize_source_filename("qa_pairs_news.csv")
        "qa_pairs_news.csv"
    """
    normalized = re.sub(r'_\d{8}_\d{6}', '', filename)
    return normalized
```

**使用箇所**:

- Qdrantポイントのペイロードに`source`フィールドとして保存
- UI用正規化CSVのファイル名として使用

---

### run_registration()

```python
def run_registration(
    csv_path: str,
    collection_name: str,
    recreate: bool,
    batch_size: int,
    provider: str
) -> bool:
    """
    Phase 2のQdrant登録処理を実行する。

    Args:
        csv_path (str): Q/AペアCSVのパス
        collection_name (str): Qdrantコレクション名
        recreate (bool): コレクションを再作成するか
        batch_size (int): Embeddingバッチサイズ
        provider (str): Embeddingプロバイダー

    Returns:
        bool: 成功時True、失敗時False

    Raises:
        FileNotFoundError: CSVファイルが見つからない
        ValueError: 必須カラムが不足
        Exception: Qdrant接続エラー、登録エラー

    Process:
        1. CSV読み込み
        2. ベクトル化テキスト準備（question + answer）
        3. Qdrantクライアント作成
        4. コレクション作成/再作成
        5. バッチ処理ループ
           - Embedding生成
           - ポイント構築
           - Qdrantへアップサート
        6. UI用正規化CSV作成
    """
```

---

### main()

```python
def main():
    """
    CLIエントリーポイント。

    Process:
        1. コマンドライン引数解析
        2. 入力検証（排他制御）
        3. APIキー確認
        4. Phase 1実行（Q/A生成）
        5. Phase 2実行（Qdrant登録）
        6. 完了メッセージ

    Exits:
        1: 入力エラー、APIキーエラー、処理エラー
        0: 正常終了
    """
```

---

## ⚠️ トラブルシューティング

### 問題1: 入力検証エラー

**症状**:

```
❌ --dataset, --input-csv, --input-chunks のいずれか1つを指定してください
```

**原因**: 入力オプションが指定されていない

**対処法**:

```bash
# いずれか1つを指定
python qa_qdrant/make_qa_register_qdrant.py \
  --dataset wikipedia_ja \
  --collection qa_wikipedia \
  --recreate
```

---

### 問題2: 複数の入力オプションエラー

**症状**:

```
❌ --dataset, --input-csv, --input-chunks は同時に指定できません
```

**原因**: 複数の入力オプションを同時に指定

**対処法**:

```bash
# ❌ 誤り
--dataset wikipedia_ja --input-csv data.csv

# ✅ 正しい
--dataset wikipedia_ja
# または
--input-csv data.csv
```

---

### 問題3: APIキーエラー

**症状**:

```
❌ GOOGLE_API_KEYが設定されていません
```

**対処法**:

```bash
export GOOGLE_API_KEY="your-api-key-here"

# または .env ファイルに設定
echo "GOOGLE_API_KEY=your-api-key-here" >> .env
```

---

### 問題4: チャンクCSVの必須カラム不足

**症状**:

```
ValueError: 必須カラムが不足しています: ['chunk_id', 'text']
```

**原因**: チャンクCSVに必須カラムが含まれていない

**必須カラム**:

- `chunk_id`
- `text`
- `tokens`
- `chunk_idx`

**対処法**:

```bash
# 正しいツールでチャンクCSVを作成
python -m chunking.csv_to_chunks_text_para \
  -i data.txt \
  -o chunks.csv
```

---

### 問題5: Q/AペアCSVの必須カラム不足

**症状**:

```
❌ Q/Aカラムが見つかりません。
```

**原因**: CSVに`question`または`answer`カラムがない

**対処法**:

```csv
# CSVファイルの形式を確認
question,answer
"質問1","回答1"
"質問2","回答2"
```

---

### 問題6: Qdrant接続エラー

**症状**:

```
❌ Qdrant接続エラー: Connection refused
```

**対処法**:

```bash
# Qdrantコンテナの起動確認
docker ps | grep qdrant

# Qdrantを起動
docker-compose up -d qdrant

# 接続確認
curl http://localhost:6333/
```

---

### 問題7: Celeryワーカーが起動していない

**症状**:

```
RuntimeError: Celery workers are not running
```

**対処法**:

```bash
# Celeryワーカーを起動
./start_celery.sh restart -w 16

# ステータス確認
./start_celery.sh status

# または、Celeryを使わない
python qa_qdrant/make_qa_register_qdrant.py \
  --dataset wikipedia_ja \
  --collection qa_wikipedia \
  --recreate
  # --use-celery を指定しない
```

---

### 問題8: メモリ不足

**症状**:

```
MemoryError: Unable to allocate array
```

**対処法**:

```bash
# バッチサイズを小さくする
--batch-size 50

# 処理文書数を制限
--max-docs 100

# Celeryワーカー数を減らす
--celery-workers 8
```

---

### 問題9: Q/A生成フェーズでCSVファイルが作成されない

**症状**:

```
❌ Q/A生成フェーズでCSVファイルが作成されませんでした。
```

**原因**:

- Q/A生成が失敗
- ファイル書き込み権限エラー

**対処法**:

```bash
# ログを確認
# qa_output/pipeline/ ディレクトリの権限確認
ls -la qa_output/pipeline/

# ディレクトリ作成
mkdir -p qa_output/pipeline

# 権限変更
chmod 755 qa_output/pipeline
```

---

### 問題10: UI用ファイル作成失敗

**症状**:

```
⚠️ UI用ファイル作成失敗: Permission denied
```

**対処法**:

```bash
# qa_output/ ディレクトリの権限確認
ls -la qa_output/

# ディレクトリ作成
mkdir -p qa_output

# 権限変更
chmod 755 qa_output
```

---

## 💡 ベストプラクティス

### 1. 段階的なテスト

```bash
# Step 1: 小規模テスト（10文書）
--max-docs 10

# Step 2: 中規模テスト（100文書）
--max-docs 100

# Step 3: 本番実行（全データ）
# --max-docs を指定しない
```

---

### 2. チャンク品質の確認

```bash
# チャンク作成のみ実行
python -m chunking.csv_to_chunks_text_para \
  -i data.txt \
  -o chunks.csv

# チャンクCSVを確認
head -20 chunks.csv

# 統計確認
python -c "
import pandas as pd
df = pd.read_csv('chunks.csv')
print(df['tokens'].describe())
print(f'Total chunks: {len(df)}')
"

# 確認後、Q/A生成
python qa_qdrant/make_qa_register_qdrant.py \
  --input-chunks chunks.csv \
  --collection my_collection \
  --recreate
```

---

### 3. Celeryの効果的な使用

```bash
# 小規模（< 100文書）: Celeryなし
python qa_qdrant/make_qa_register_qdrant.py \
  --input-chunks chunks.csv \
  --collection small_collection \
  --recreate

# 中規模（100-1,000文書）: Celery 8-16ワーカー
./start_celery.sh restart -w 16
python qa_qdrant/make_qa_register_qdrant.py \
  --input-chunks chunks.csv \
  --collection medium_collection \
  --use-celery \
  --celery-workers 16 \
  --recreate

# 大規模（> 1,000文書）: Celery 24ワーカー
./start_celery.sh restart -w 24
python qa_qdrant/make_qa_register_qdrant.py \
  --input-chunks chunks.csv \
  --collection large_collection \
  --use-celery \
  --celery-workers 24 \
  --recreate
```

---

### 4. バッチサイズの最適化

```bash
# メモリ潤沢（32GB以上）
--batch-size 200

# 標準（16GB）
--batch-size 100

# メモリ制限あり（8GB以下）
--batch-size 50
```

---

### 5. ログの活用

```bash
# ログをファイルに保存
python qa_qdrant/make_qa_register_qdrant.py \
  --input-chunks chunks.csv \
  --collection my_collection \
  --use-celery \
  --celery-workers 16 \
  --recreate \
  2>&1 | tee qa_generation.log

# エラーのみ抽出
grep "❌" qa_generation.log

# 進捗確認
grep "進捗" qa_generation.log
```

---

### 6. コレクション名の規則

```bash
# 推奨: qa_{データセット名}
--collection qa_wikipedia_ja
--collection qa_news
--collection qa_fineweb

# 避ける: 一般的すぎる名前
--collection data
--collection test
```

---

### 7. 並列処理の監視

```bash
# Flowerでモニタリング
celery -A celery_config flower --port=5555

# ブラウザでアクセス
# http://localhost:5555

# タスク数、成功率、失敗数などを確認
```

---

### 8. 再実行時の注意

```bash
# 既存コレクションを削除して再作成
--recreate

# 既存コレクションに追記（非推奨）
# --recreate を指定しない
```

---

### 9. エラーハンドリング

```bash
# エラー発生時は自動的に終了（sys.exit(1)）
# 安全に中断できる（Ctrl+C）

# 中断後の再開
# 同じコマンドを実行すれば最初から再実行される
# （中間状態の保存機能はないため）
```

---

### 10. UI統合の活用

```python
# agent_rag.pyでの参照

# 正規化されたファイル名で安定的に参照可能
source_file = "qa_pairs_wikipedia_ja.csv"  # タイムスタンプなし

# qa_output/ ディレクトリから読み込み
csv_path = os.path.join("qa_output", source_file)
df = pd.read_csv(csv_path)
```

---

## 📊 パフォーマンス指標

### チャンク作成


| 文書数 | ワーカー数 | 処理時間 | チャンク数 |
| ------ | ---------- | -------- | ---------- |
| 100    | 8          | ~2分     | ~300       |
| 1,000  | 16         | ~15分    | ~3,000     |
| 10,000 | 24         | ~2時間   | ~30,000    |

### Q/A生成


| チャンク数 | Celery | ワーカー数 | 処理時間 | Q/A数   |
| ---------- | ------ | ---------- | -------- | ------- |
| 100        | No     | -          | ~15分    | ~300    |
| 100        | Yes    | 8          | ~5分     | ~300    |
| 1,000      | Yes    | 16         | ~30分    | ~3,000  |
| 10,000     | Yes    | 24         | ~4時間   | ~30,000 |

### Qdrant登録


| Q/A数   | バッチサイズ | 処理時間 | 登録速度 |
| ------- | ------------ | -------- | -------- |
| 1,000   | 100          | ~5分     | ~200/分  |
| 10,000  | 100          | ~30分    | ~333/分  |
| 100,000 | 200          | ~4時間   | ~416/分  |

---

## 📚 関連ドキュメント

- `qa_generation/doc/qa_generation_module_guide.md` - qa_generationモジュールの詳細
- `qa_qdrant/doc/qa_generation.md` - モジュール依存関係分析
- `chunking/SKILL.md` - チャンク処理の技術詳細
- `services/qdrant_service.py` - Qdrant操作関数の実装
- `qdrant_client_wrapper.py` - Qdrantクライアントの実装

---

**作成日**: 2025-01-17
**最終更新**: 2025-01-17
**対象ファイル**: `make_qa_register_qdrant.py`
**ドキュメントバージョン**: 2.0.0
