# make_qa_register_qdrant.py 完全ガイド

**Q/A生成からQdrantベクトルデータベース登録までを一貫して実行する統合CLIツール**

**✨ 最新版の特徴**:
- **スマートQ/A生成がデフォルト**：`SmartQAGenerator`による動的Q/A数決定
- **CSV行結合オプション追加**：`--combine-rows` + `--block-size` で複数行をチャンク化
- **テキストカラム指定**：`--text-column` で任意のカラムを指定可能
- LLMが内容の重要度・複雑さを分析し、最適なQ/A数（0-5個）を自動決定
- 従来方式（トークン数ベース）への切り替えも可能

---

## 📋 目次

1. [概要](#概要)
2. [Q/A生成方式](#qa生成方式)
3. [プログラム構成](#プログラム構成)
4. [処理の流れ](#処理の流れ)
5. [データ処理の流れ](#データ処理の流れ)
6. [入力形式と対応](#入力形式と対応)
7. [使用方法](#使用方法)
8. [コマンドラインオプション](#コマンドラインオプション)
9. [実行例](#実行例)
10. [推奨される使い分け](#推奨される使い分け)
11. [トラブルシューティング](#トラブルシューティング)

---

## 📖 概要

### ファイル情報

- **ファイル名**: `make_qa_register_qdrant.py`
- **配置場所**: `qa_qdrant/make_qa_register_qdrant.py`
- **バージョン**: v2.2（CSV処理オプション追加版）
- **説明**: Q/A生成からQdrant登録までを完結する統合ツール

### 改修履歴

**v2.2（最新）- CSV処理オプション追加**:
- ✨ `--text-column` オプション追加（テキストカラム名指定、デフォルト: text）
- ✨ `--combine-rows` オプション追加（複数行を結合してチャンク化）
- ✨ `--block-size` オプション追加（結合する行数、デフォルト: 400）
- ✨ `combine_rows_to_chunks()` 関数を追加
- CSV入力時の柔軟なデータ前処理が可能に

**v2.1 - スマート生成デフォルト化**:
- ✨ `SmartQAGenerator`をデフォルトで使用
- ✨ LLMによる動的Q/A数決定（0-5個）
- ✨ 内容の重要度・複雑さを考慮した高品質生成
- ✨ `--use-smart-generation`オプション追加（デフォルト有効）
- ✨ `--no-smart-generation`オプション追加（従来方式切り替え）

**v2.0 - 入力処理の統一**:
- `--input-chunks`を廃止（チャンク処理の統一）
- `--input-csv`を`--input-file`に変更（テキスト/CSV両対応）
- `--output`オプションを追加（出力先の柔軟化）
- `--ui-output`オプションを追加（UI用CSV出力先の柔軟化）
- CSVファイルのカラム自動判定機能を追加

### 主な特徴

✅ **インテリジェントなQ/A生成（デフォルト）**

- **スマート生成モード**：`SmartQAGenerator`クラス使用
  - LLMがチャンク内容を分析
  - 情報密度、重要度、複雑さを評価
  - 最適なQ/A数を動的決定（0-5個）
  - 主要トピックを明示的にカバー
  - 不要なチャンクは0個生成（スキップ可能）

- **従来モード**：トークン数ベースの固定計算
  - `--no-smart-generation`で切り替え
  - 高速処理・低コスト
  - 大規模データセット向け

✅ **柔軟なCSV入力処理** 🆕

- `--text-column`：テキストを含むカラム名を指定（デフォルト: text）
- `--combine-rows`：複数行を結合してチャンク化
- `--block-size`：結合する行数を指定（デフォルト: 400）
- 小さなレコードが多いCSVを効率的にQ/A生成可能

✅ **2つの入力形式に対応**

- 事前定義データセット（`config.py`で定義済み）
- ローカルファイル（`.txt`テキストファイル、`.csv`ファイル）

✅ **インテリジェントな入力処理**

- テキストファイル → 自動的にチャンク作成 + Q/A生成
- CSV（テキストのみ）→ チャンク作成 + Q/A生成（行結合オプション対応）
- CSV（Q/Aペアあり）→ Phase 1をスキップして直接登録

✅ **2フェーズの自動実行**

- **Phase 0**: CSV行結合処理（`--combine-rows`指定時のみ）🆕
- **Phase 1**: Q/Aペア生成（`QAPipeline` + `SmartQAGenerator`）
- **Phase 2**: Qdrant登録（Embedding + インデックス化）

✅ **高度な並列処理**

- Celeryによる非同期タスク実行
- 最大24ワーカーでの並列処理対応

✅ **UI統合対応**

- ファイル名の正規化（タイムスタンプ除去）
- `agent_rag.py`での参照を容易化

---

## 🎯 Q/A生成方式

### 2つの生成方式の比較

| 項目 | スマート生成（デフォルト） | 従来方式 |
|------|--------------------------|---------|
| **使用クラス** | `SmartQAGenerator` | `QAGenerator` (legacy mode) |
| **Q/A数決定** | LLMによる動的判断 | トークン数ベース固定計算 |
| **Q/A数範囲** | 0-5個（柔軟） | 2-8個（固定ルール） |
| **LLM呼び出し** | 2回/チャンク（分析+生成） | 1回/チャンク |
| **処理速度** | 🐢 約2倍の時間 | ⚡ 高速 |
| **API コスト** | 💰💰 約2倍 | 💰 低コスト |
| **品質** | ⭐⭐⭐ 高品質 | ⭐⭐ 標準 |
| **トピック明示** | ✅ あり | ❌ なし |
| **重要度考慮** | ✅ あり | ❌ なし |
| **0個生成** | ✅ 可能（不要時スキップ） | ❌ 不可能 |
| **メタデータ** | 豊富（topic, importance, complexity） | 基本のみ |

### スマート生成の処理フロー

```
チャンク入力
    ↓
【ステップ1】LLMによる分析
    ├─ 情報密度の評価
    ├─ 重要度スコア計算（0.0-1.0）
    ├─ 複雑さ判定（low/medium/high）
    ├─ 主要トピックの抽出
    └─ 適切なQ/A数決定（0-5個）
    ↓
【ステップ2】動的Q/A生成
    ├─ 分析結果に基づくプロンプト構築
    ├─ 主要トピックを優先的にカバー
    ├─ 重要度に応じた品質指示
    └─ トピック付きQ/A生成
    ↓
Q/Aペア出力（最適数）
```

### 従来方式の処理フロー

```
チャンク入力
    ↓
【ステップ1】トークン数カウント
    └─ 固定ルールでQ/A数決定
        • < 50 tokens  → 2個
        • < 100 tokens → 3個
        • < 200 tokens → 4個
        • < 300 tokens → 5個
        • >= 300 tokens → 6個
        • 後半チャンク → +1個
    ↓
【ステップ2】Q/A生成
    └─ 固定数のQ/A生成
    ↓
Q/Aペア出力（固定数）
```

### 生成例の比較

#### テストケース: 技術文書

**入力チャンク**:
```
AES-256暗号化アルゴリズムは、対称鍵暗号方式の一種で、
256ビットの鍵長を持ちます。NIST（米国国立標準技術研究所）
により承認されており、機密情報の保護に広く使用されています。
ブロック暗号として動作し、128ビットのブロックサイズで
データを処理します。CBC、GCM、CTRなど複数のモードが利用可能で、
用途に応じて選択できます。
```

#### スマート生成の出力（5個）

```json
[
  {
    "question": "AES-256の暗号化方式の特徴は何ですか？",
    "answer": "対称鍵暗号方式の一種で、256ビットの鍵長を持ち、高いセキュリティを提供します",
    "topic": "暗号化方式",
    "generation_method": "smart",
    "importance_score": 0.9,
    "complexity": "high"
  },
  {
    "question": "AES-256の鍵長は何ビットですか？",
    "answer": "256ビットの鍵長を持ちます",
    "topic": "鍵長"
  },
  {
    "question": "AES-256のブロックサイズはどのくらいですか？",
    "answer": "128ビットのブロックサイズでデータを処理します",
    "topic": "ブロックサイズ"
  },
  {
    "question": "AES-256で利用可能な動作モードにはどのようなものがありますか？",
    "answer": "CBC、GCM、CTRなど複数のモードが利用可能で、用途に応じて選択できます",
    "topic": "利用モード"
  },
  {
    "question": "AES-256はどの機関により承認されていますか？",
    "answer": "NIST（米国国立標準技術研究所）により承認されており、機密情報の保護に広く使用されています",
    "topic": "承認機関"
  }
]
```

**特徴**:
- ✅ 主要トピックを全てカバー
- ✅ 各Q/Aにトピックラベル付き
- ✅ 重要度・複雑さのメタデータ
- ✅ 技術的な詳細を漏らさない

#### 従来方式の出力（4個）

```json
[
  {
    "question": "AES-256とは何ですか？",
    "answer": "対称鍵暗号方式の一種で、256ビットの鍵長を持つ暗号化アルゴリズムです",
    "question_type": "fact",
    "generation_method": "legacy"
  },
  {
    "question": "AES-256はどこで承認されていますか？",
    "answer": "NIST（米国国立標準技術研究所）により承認されています",
    "question_type": "fact"
  },
  {
    "question": "ブロックサイズは何ビットですか？",
    "answer": "128ビットのブロックサイズでデータを処理します",
    "question_type": "fact"
  },
  {
    "question": "利用可能なモードは？",
    "answer": "CBC、GCM、CTRなど複数のモードが利用可能です",
    "question_type": "fact"
  }
]
```

**特徴**:
- ✅ 基本的な情報はカバー
- ❌ トピックが不明確
- ❌ 質問タイプが全て"fact"
- ❌ メタデータが少ない

---

## 🏗️ プログラム構成

### プロジェクト構成

```text
実行ファイル層（4つのCLIツール）
├── make_qa.py                        ⬅️ Q/A生成専用ツール
├── register_csv_to_qdrant.py         ⬅️ 汎用CSV登録ツール
├── register_qdrant.py                ⬅️ make_qa出力専用登録ツール
└── make_qa_register_qdrant.py        ⬅️ 統合ツール（Q/A生成→Qdrant登録）

共通モジュール層（全ツールが依存）
├── qa_generation/
│   ├── pipeline.py                   ⬅️ QAPipeline（Q/A生成ロジック）
│   ├── smart_qa_generator.py         ⬅️ SmartQAGenerator（スマート生成）
│   └── generation.py                 ⬅️ QAGenerator（従来+スマート統合）
├── services/
│   └── qdrant_service.py             ⬅️ Qdrant操作ロジック
├── qdrant_client_wrapper.py          ⬅️ Qdrantクライアント作成
└── config.py                         ⬅️ DATASET_CONFIGS、QdrantConfig
```

### ディレクトリ構造

```text
プロジェクトルート/
├── qa_qdrant/                        # Q/Aペア生成&登録ツール群
│   ├── make_qa.py                    # Q/A生成専用CLI
│   ├── make_qa_register_qdrant.py    # 統合CLI（本ドキュメント対象）
│   └── register_to_qdrant.py         # Qdrant登録専用CLI
│
├── qa_generation/                    # Q/A生成コアモジュール
│   ├── __init__.py
│   ├── pipeline.py                   # QAPipelineクラス
│   ├── smart_qa_generator.py         # ✨ SmartQAGeneratorクラス
│   ├── generation.py                 # QAGeneratorクラス（統合版）
│   ├── structure.py                  # チャンク作成・統合
│   ├── evaluation.py                 # カバレッジ分析
│   ├── semantic.py                   # セマンティック類似度計算
│   ├── models.py                     # Pydanticモデル
│   ├── content.py                    # コンテンツ処理
│   ├── data_io.py                    # データ入出力
│   └── config.py                     # データセット設定
│
├── services/                         # Qdrant関連サービス
│   └── qdrant_service.py             # Qdrant操作関数群
│
├── helper/                           # ヘルパーモジュール
│   └── helper_llm.py                 # LLMクライアント
│
├── qdrant_client_wrapper.py          # Qdrantクライアント生成
├── config.py                         # グローバル設定
├── celery_tasks.py                   # Celeryタスク定義
│
├── qa_output/                        # Q/A生成出力先
│   ├── pipeline/                     # QAPipeline出力（タイムスタンプ付き）
│   │   ├── qa_pairs_*.csv
│   │   ├── qa_pairs_*.json
│   │   ├── coverage_*.json
│   │   └── summary_*.json
│   └── *.csv                         # UI用正規化CSV（タイムスタンプなし）
│
└── data/                             # データセットファイル
    ├── wikipedia_ja.csv
    ├── cc_news.csv
    └── ...
```

### モジュール依存関係

```text
make_qa_register_qdrant.py
    ├─ Phase 0: CSV行結合処理（--combine-rows指定時）🆕
    │   └─ combine_rows_to_chunks()
    │       ├─ pandas.read_csv()
    │       ├─ block_size行ごとに結合
    │       └─ combined_chunks_YYYYMMDD_HHMMSS.csv 出力
    │
    ├─ Phase 1: Q/A生成
    │   ├─ qa_generation/pipeline.py (QAPipeline)
    │   │   ├─ qa_generation/structure.py (create_document_chunks)
    │   │   ├─ qa_generation/generation.py (generate_qa_dataset)
    │   │   │   └─ qa_generation/smart_qa_generator.py (SmartQAGenerator) ✨
    │   │   ├─ qa_generation/evaluation.py (analyze_coverage)
    │   │   └─ qa_generation/data_io.py (save_results)
    │   └─ config.py (DATASET_CONFIGS)
    │
    └─ Phase 2: Qdrant登録
        ├─ services/qdrant_service.py
        │   ├─ create_or_recreate_collection_for_qdrant()
        │   ├─ embed_texts_for_qdrant()
        │   ├─ build_points_for_qdrant()
        │   └─ upsert_points_to_qdrant()
        └─ qdrant_client_wrapper.py (create_qdrant_client)
```

### 主要な関数・クラス

#### `make_qa_register_qdrant.py`

| 関数/クラス | 説明 |
|-----------|------|
| `normalize_source_filename()` | ファイル名から日時サフィックスを除去 |
| `combine_rows_to_chunks()` 🆕 | CSVの複数行を結合してチャンクCSVを作成 |
| `run_registration()` | Phase 2（Qdrant登録）の実行 |
| `main()` | CLIエントリーポイント |

#### `qa_generation/pipeline.py`

| クラス/メソッド | 説明 |
|---------------|------|
| `QAPipeline.__init__()` | パイプライン初期化 |
| `QAPipeline.load_data()` | データ読み込み（.txt, .csv対応） |
| `QAPipeline.create_chunks()` | チャンク作成 |
| `QAPipeline.generate_qa()` | Q/A生成（スマート/従来切り替え） |
| `QAPipeline.evaluate_coverage()` | カバレッジ分析 |
| `QAPipeline.save()` | 結果保存 |
| `QAPipeline.run()` | パイプライン実行 |

#### `qa_generation/smart_qa_generator.py` ✨

| クラス/メソッド | 説明 |
|---------------|------|
| `SmartQAGenerator.__init__()` | 初期化（Geminiモデル設定） |
| `SmartQAGenerator.analyze_chunk()` | チャンク分析（Q/A数・重要度決定） |
| `SmartQAGenerator.generate_qa_pairs()` | 分析結果に基づくQ/A生成 |
| `SmartQAGenerator.process_chunk()` | 分析+生成の一括実行 |

#### `qa_generation/generation.py`

| クラス/メソッド | 説明 |
|---------------|------|
| `QAGenerator.__init__()` | 初期化（use_smart_generationフラグ） |
| `QAGenerator.determine_qa_count()` | Q/A数決定（スマート/従来分岐） |
| `QAGenerator.generate_for_chunk()` | 単一チャンクQ/A生成 |
| `QAGenerator._generate_smart()` | スマート生成実行 |
| `QAGenerator._generate_legacy()` | 従来方式実行 |
| `generate_qa_dataset()` | データセット全体のQ/A生成 |

#### `services/qdrant_service.py`

| 関数 | 説明 |
|-----|------|
| `create_or_recreate_collection_for_qdrant()` | コレクション作成/再作成 |
| `embed_texts_for_qdrant()` | テキストのEmbedding生成 |
| `build_points_for_qdrant()` | Qdrantポイント構築 |
| `upsert_points_to_qdrant()` | Qdrantへアップロード |

---

## 🔄 処理の流れ

### 全体フロー

```
┌─────────────────────────────────────────────────────────┐
│            make_qa_register_qdrant.py                   │
│                                                         │
│  ┌───────────────────────────────────────────────┐    │
│  │  Phase 0: CSV行結合処理（--combine-rows時のみ）🆕│    │
│  │                                                 │    │
│  │  --combine-rows 指定時:                         │    │
│  │    ├─ --text-column で指定カラムを読み込み     │    │
│  │    ├─ --block-size 行ごとに結合               │    │
│  │    └─ combined_chunks_YYYYMMDD_HHMMSS.csv 作成 │    │
│  └───────────────────────────────────────────────┘    │
│                          ↓                             │
│  ┌───────────────────────────────────────────────┐    │
│  │  Phase 1: Q/A Generation Pipeline              │    │
│  │                                                 │    │
│  │  入力ファイル判定                                │    │
│  │    ├─ .txt file → チャンク作成 + Q/A生成       │    │
│  │    ├─ .csv (text only) → チャンク作成 + Q/A生成│    │
│  │    ├─ .csv (Q/A pairs) → Phase 1スキップ       │    │
│  │    └─ dataset → チャンク作成 + Q/A生成         │    │
│  │         ↓                                       │    │
│  │  ✨ スマート生成（デフォルト）                   │    │
│  │    ├─ LLMがチャンク分析                         │    │
│  │    ├─ 最適Q/A数決定（0-5個）                   │    │
│  │    └─ トピック付きQ/A生成                      │    │
│  │         ↓                                       │    │
│  │  出力: qa_pairs_YYYYMMDD_HHMMSS.csv            │    │
│  └───────────────────────────────────────────────┘    │
│                          ↓                             │
│  ┌───────────────────────────────────────────────┐    │
│  │  Phase 2: Qdrant Registration                  │    │
│  │                                                 │    │
│  │  1. CSVファイルロード                           │    │
│  │  2. Qdrantコレクション準備                      │    │
│  │  3. バッチ処理ループ:                           │    │
│  │     ├─ Embedding生成（question + answer）      │    │
│  │     ├─ Qdrantポイント構築                       │    │
│  │     └─ Qdrantへアップロード                     │    │
│  │  4. UI用正規化CSVの作成                         │    │
│  └───────────────────────────────────────────────┘    │
│                          ↓                             │
│                  ✅ 完了                                │
└─────────────────────────────────────────────────────────┘
```

### Phase 0: CSV行結合処理の詳細フロー 🆕

```
--combine-rows 指定時のみ実行

入力CSV
    ↓
【1】CSVロード
    └─ pd.read_csv()
    ↓
【2】カラム確認
    ├─ --text-column で指定されたカラムを使用
    └─ カラムが存在しない場合はエラー
    ↓
【3】行結合処理
    ├─ --block-size 行ごとに結合
    ├─ 空行をフィルタリング
    └─ 結合したテキストを "\n\n" で連結
    ↓
【4】チャンクCSV出力
    ├─ chunk_id: 連番
    ├─ text: 結合されたテキスト
    ├─ start_row: 開始行番号
    ├─ end_row: 終了行番号
    └─ row_count: 結合した行数
    ↓
出力: combined_chunks_YYYYMMDD_HHMMSS.csv
    ↓
Phase 1へ（この一時CSVを入力として使用）
```

### Phase 1: Q/A生成の詳細フロー（スマート生成）

```
入力ファイル
    ↓
【1】データ読み込み
    ├─ .txt → DataFrameに変換
    ├─ .csv → カラム判定
    │   ├─ question & answer あり → Phase 2へスキップ
    │   └─ text/Combined_Text → チャンク作成へ
    └─ dataset → config.pyから読み込み
    ↓
【2】チャンク作成
    ├─ セマンティック分割（オプション）
    ├─ パラグラフ優先分割
    ├─ 文境界保持
    └─ オーバーラップ設定（オプション）
    ↓
【3】Q/A生成（スマート生成モード） ✨
    ├─ チャンクごとに:
    │   ├─ 【3-1】LLMでチャンク分析
    │   │   ├─ 情報密度評価
    │   │   ├─ 重要度スコア計算（0.0-1.0）
    │   │   ├─ 複雑さ判定（low/medium/high）
    │   │   ├─ 主要トピック抽出
    │   │   └─ Q/A数決定（0-5個）
    │   │
    │   └─ 【3-2】Q/A生成
    │       ├─ 分析結果に基づくプロンプト
    │       ├─ トピック優先指示
    │       ├─ 重要度に応じた品質指示
    │       └─ トピック付きQ/A生成
    │
    ├─ バッチ処理（batch_chunks=3）
    └─ Celery並列処理（オプション）
    ↓
【4】カバレッジ分析
    ├─ セマンティック類似度計算
    ├─ カバレッジ率算出
    └─ 未カバーチャンク検出
    ↓
【5】結果保存
    ├─ qa_pairs_YYYYMMDD_HHMMSS.csv（タイムスタンプ付き）
    ├─ qa_pairs_YYYYMMDD_HHMMSS.json
    ├─ coverage_YYYYMMDD_HHMMSS.json
    └─ summary_YYYYMMDD_HHMMSS.json
    ↓
Phase 2へ
```

### Phase 2: Qdrant登録の詳細フロー

```
Q/A CSV
    ↓
【1】CSVロード
    └─ pd.read_csv()
    ↓
【2】Qdrantコレクション準備
    ├─ --recreate指定時:
    │   ├─ 既存コレクション削除
    │   └─ 新規作成
    └─ 未指定時:
        └─ コレクション存在確認
    ↓
【3】バッチ処理ループ（batch_size=100）
    ├─ Embedding生成
    │   └─ embed_texts_for_qdrant(question + "\n" + answer)
    ├─ Qdrantポイント構築
    │   ├─ UUID生成
    │   ├─ ベクトル設定
    │   └─ Payload設定:
    │       ├─ question, answer
    │       ├─ domain, source
    │       ├─ question_type, difficulty
    │       ├─ ✨ topic（スマート生成の場合）
    │       ├─ ✨ importance_score（スマート生成の場合）
    │       ├─ ✨ complexity（スマート生成の場合）
    │       └─ chunk_id, generation_method
    └─ Qdrantアップロード
    ↓
【4】UI用正規化CSV作成
    ├─ ファイル名からタイムスタンプ除去
    │   例: qa_pairs_20250120_123456.csv
    │       → qa_pairs.csv
    └─ qa_output/に保存
    ↓
✅ 完了
```

---

## 📊 データ処理の流れ

### データ構造の変換

```
1. 入力データ（テキスト/CSV）
    ↓
1.5 行結合処理（--combine-rows時のみ）🆕
    └─ 複数行を block_size ごとに結合
    ↓
2. DataFrame
    columns: [Combined_Text, title, ...]
    ↓
3. チャンク（Dictのリスト）
    {
        'id': 'chunk_0',
        'text': 'チャンク本文...',
        'tokens': 250,
        'type': 'paragraph',
        'doc_id': 'doc_0',
        'chunk_idx': 0,
        'dataset_type': 'local_file'
    }
    ↓
4. Q/Aペア（Dictのリスト）- スマート生成の場合 ✨
    {
        'question': '質問文',
        'answer': '回答文',
        'question_type': 'fact',
        'topic': 'トピック名',              # ✨ スマート生成
        'source_chunk_id': 'chunk_0',
        'doc_id': 'doc_0',
        'dataset_type': 'local_file',
        'chunk_idx': 0,
        'generation_method': 'smart',       # ✨ スマート生成
        'importance_score': 0.85,           # ✨ スマート生成
        'complexity': 'high'                # ✨ スマート生成
    }
    ↓
5. CSV/JSONファイル
    qa_pairs_YYYYMMDD_HHMMSS.csv
    ↓
6. Qdrantポイント
    PointStruct(
        id=UUID,
        vector=[768次元ベクトル],
        payload={
            'question': '質問文',
            'answer': '回答文',
            'topic': 'トピック名',           # ✨ スマート生成
            'importance_score': 0.85,        # ✨ スマート生成
            'complexity': 'high',            # ✨ スマート生成
            'domain': 'local_file',
            'source': 'normalized_filename.csv',
            ...
        }
    )
```

### データフロー図

```
┌────────────────┐
│  Input Source  │
└────────────────┘
        │
        ├─ .txt file
        ├─ .csv (text)  ─────┐
        ├─ .csv (Q/A)        │ --combine-rows 🆕
        └─ dataset           ↓
        │            ┌────────────────┐
        │            │ 行結合処理      │
        │            │ (block_size行) │
        │            └────────────────┘
        │                    │
        ↓                    ↓
┌────────────────┐
│   DataFrame    │
└────────────────┘
        │
        ↓
┌────────────────┐
│    Chunks      │  ← create_document_chunks()
└────────────────┘
        │
        ↓
┌────────────────┐
│ ✨ Smart       │  ← SmartQAGenerator.analyze_chunk()
│   Analysis     │     (情報密度、重要度、複雑さ、トピック)
└────────────────┘
        │
        ↓
┌────────────────┐
│   Q/A Pairs    │  ← SmartQAGenerator.generate_qa_pairs()
│  (with topics) │     (トピック付き、メタデータ豊富)
└────────────────┘
        │
        ↓
┌────────────────┐
│  CSV/JSON      │  ← save_results()
│   Files        │
└────────────────┘
        │
        ↓
┌────────────────┐
│  Embeddings    │  ← embed_texts_for_qdrant()
└────────────────┘
        │
        ↓
┌────────────────┐
│ Qdrant Points  │  ← build_points_for_qdrant()
└────────────────┘
        │
        ↓
┌────────────────┐
│ Qdrant DB      │  ← upsert_points_to_qdrant()
└────────────────┘
```

---

## 📂 入力形式と対応

### 1. テキストファイル (.txt)

**形式**:

```
任意のプレーンテキスト
複数行対応
```

**処理**:

- 自動的に`Combined_Text`カラムを持つDataFrameに変換
- チャンク作成
- Q/A生成（スマート生成）

**使用例**:

```bash
python make_qa_register_qdrant.py \
  --input-file document.txt \
  --collection my_docs \
  --recreate
```

### 2. CSVファイル（テキストのみ）

**形式**:

```csv
text,title
"テキスト内容1","タイトル1"
"テキスト内容2","タイトル2"
```

または

```csv
Combined_Text,title
"テキスト内容1","タイトル1"
"テキスト内容2","タイトル2"
```

**処理**:

- `text`または`Combined_Text`カラムを検出
- チャンク作成
- Q/A生成（スマート生成）

**使用例**:

```bash
python make_qa_register_qdrant.py \
  --input-file documents.csv \
  --collection my_collection \
  --recreate
```

#### オプション: 行結合処理 🆕

小さなレコードが多いCSV（例: ニュース記事1行=1レコード）を効率的に処理する場合：

```bash
python make_qa_register_qdrant.py \
  --input-file news.csv \
  --collection news \
  --text-column text \
  --combine-rows \
  --block-size 400 \
  --recreate
```

**処理内容**:
1. CSVの `text` カラムを読み込み
2. 400行ごとに結合して1つのチャンクに
3. 結合されたチャンクからQ/A生成

**行結合の効果**:

| 状態 | 行数 | チャンク数 | Q/A生成効率 |
|------|------|-----------|------------|
| 結合前 | 10,000行 | 10,000チャンク | 低い（1行=1チャンクで内容が薄い） |
| 結合後（block_size=400） | 10,000行 | 25チャンク | 高い（400行=1チャンクで十分な内容） |

### 3. CSVファイル（Q/Aペア既存）

**形式**:

```csv
question,answer,topic,importance_score,complexity
"質問1","回答1","トピック1",0.85,"high"
"質問2","回答2","トピック2",0.60,"medium"
```

**処理**:

- `question`と`answer`カラムを検出
- **Phase 1をスキップ**
- 直接Phase 2（Qdrant登録）へ

**使用例**:

```bash
python make_qa_register_qdrant.py \
  --input-file qa_pairs.csv \
  --collection my_qa \
  --recreate
```

### 4. 事前定義データセット

**形式**:

```python
# config.pyで定義
DATASET_CONFIGS = {
    "wikipedia_ja": {
        "name": "Wikipedia日本語",
        "file": "data/wikipedia_ja.csv",
        "text_column": "Combined_Text",
        "title_column": "title",
        "lang": "ja",
        "chunk_size": 300,
        "qa_per_chunk": 3
    },
    ...
}
```

**使用例**:

```bash
python make_qa_register_qdrant.py \
  --dataset wikipedia_ja \
  --collection wiki_ja \
  --max-docs 100 \
  --recreate
```

### 入力判定ロジック

```
入力ファイル種別判定フローチャート:

┌─ --input-file 指定? ──┐
│                       │
YES                    NO
│                       │
↓                       ↓
拡張子チェック         --dataset指定
│                       │
├─ .txt               ↓
│  └→ チャンク+Q/A（スマート生成）
│
├─ .csv
│  ├─ --combine-rows 指定? 🆕
│  │  └→ 行結合処理 → チャンク+Q/A
│  │
│  ├─ カラムチェック
│  │  ├─ question & answer あり
│  │  │  └→ Phase 1スキップ
│  │  │
│  │  ├─ text or Combined_Text のみ
│  │  │  └→ チャンク+Q/A（スマート生成）
│  │  │
│  │  └─ 必要カラムなし
│  │     └→ エラー
│
└─ その他
   └→ エラー
```

---

## 💻 使用方法

### 基本的な使い方

#### パターン1: テキストファイルからQ/A生成&登録（スマート生成）

```bash
python make_qa_register_qdrant.py \
  --input-file document.txt \
  --collection my_docs \
  --recreate
```

**実行内容**:

1. テキストファイルを読み込み
2. セマンティックチャンク作成
3. ✨ **スマート生成**: LLMがチャンク分析 → 最適Q/A数決定 → トピック付きQ/A生成
4. Qdrantにベクトル登録

**実行ログ**:

```
==============================================================
🆕 Q/A生成モード: スマート生成（デフォルト）
   - LLMによる動的Q/A数決定（0-5個）
   - 内容の重要度・複雑さを考慮
   - 主要トピックを明示的にカバー
   ※ 従来方式に戻す場合: --no-smart-generation
==============================================================

Phase 1: QA Generation Pipeline
==============================================================

[1/4] データ読み込み...
  📄 テキストファイル: document.txt
  ✅ 読み込み完了: テキスト長 5,234 文字

[2/4] チャンク作成...
  ✅ 12 chunks created

[3/4] Q/Aペア生成...
  生成モード: スマート生成

  分析完了: Q/A数=5, 重要度=0.90
  Q/A生成完了: 5個
  分析完了: Q/A数=3, 重要度=0.65
  Q/A生成完了: 3個
  分析完了: Q/A数=0, 重要度=0.20
  Q/A生成スキップ（qa_count=0）
  ...

[4/4] カバレッジ分析...
  ✅ Coverage: 85.7%

Phase 2: Qdrant Registration
==============================================================
  ✅ Collection 'my_docs' created
  ✅ 42 Q/A pairs loaded
  ✅ Batch 1/1 uploaded (42 points)
  ✅ UI CSV saved: qa_output/document.csv

✅ Complete!
```

#### パターン2: CSV行結合オプションの使用 🆕

```bash
python make_qa_register_qdrant.py \
  --input-file OUTPUT/cc_news_5per.csv \
  --collection cc_news_5per \
  --use-celery \
  --model gemini-2.5-flash \
  --concurrency 4 \
  --text-column text \
  --combine-rows \
  --block-size 400 \
  --recreate
```

**実行ログ**:

```
==============================================================
🆕 Q/A生成モード: スマート生成（デフォルト）
   - LLMによる動的Q/A数決定（0-5個）
   - 内容の重要度・複雑さを考慮
   - 主要トピックを明示的にカバー
   ※ 従来方式に戻す場合: --no-smart-generation
==============================================================

📦 CSV行結合設定:
   - テキストカラム: text
   - ブロックサイズ: 400 行
==============================================================

🔧 Celery並列処理設定:
   - 並列タスク数 (concurrency): 4
   - ワーカープロセス数チェック: 1
   ※ start_celery.sh -c と同じ値を推奨
==============================================================

Phase 1: QA Generation Pipeline
==============================================================

📁 入力ファイル: OUTPUT/cc_news_5per.csv
✅ CSVファイル確認: 5000 行
   カラム: ['text', 'title', 'date']
📦 --combine-rows が指定されました - 行結合処理を実行
📦 行結合処理を開始
   - テキストカラム: text
   - ブロックサイズ: 400 行
   - 入力行数: 5000
   - 生成チャンク数: 13
   - 出力ファイル: qa_output/pipeline/combined_chunks_20250124_123456.csv
...
```

#### パターン3: 従来方式で処理（大規模データ向け）

```bash
python make_qa_register_qdrant.py \
  --input-file large_dataset.csv \
  --collection large_docs \
  --max-docs 10000 \
  --no-smart-generation \
  --recreate
```

**実行ログ**:

```
==============================================================
🔧 Q/A生成モード: 従来方式（トークン数ベース）
   - 固定的なQ/A数決定（2-8個）
   ※ スマート生成に切り替える場合: --use-smart-generation
==============================================================

Phase 1: QA Generation Pipeline
==============================================================

[3/4] Q/Aペア生成...
  生成モード: 従来方式
  ...
```

### 高度な使用例

#### Celery並列処理（スマート生成）

```bash
# 1. Celeryワーカー起動（別ターミナル）
./start_celery.sh restart -c 8 --flower

# 2. 並列処理で実行
python make_qa_register_qdrant.py \
  --input-file large_doc.txt \
  --collection large_docs \
  --use-celery \
  --concurrency 8 \
  --recreate
```

**注意**: スマート生成 + Celery並列処理の場合、処理時間は短縮されますが、API呼び出し数は2倍のままです。

#### セマンティック分割を使用（スマート生成）

```bash
python make_qa_register_qdrant.py \
  --input-file document.txt \
  --collection my_docs \
  --use-similarity \
  --similarity-threshold 0.75 \
  --overlap-tokens 50 \
  --recreate
```

**効果**:

- より自然な文脈境界でチャンク分割
- スマート生成が各チャンクの内容を分析
- 高品質なQ/A生成

#### 出力先のカスタマイズ

```bash
python make_qa_register_qdrant.py \
  --input-file document.txt \
  --collection my_docs \
  --output custom_output/qa_results \
  --ui-output custom_output/ui_files \
  --recreate
```

---

## 🎛️ コマンドラインオプション

### 入力ソースオプション（いずれか1つ必須）

| オプション | 型 | 説明 |
|-----------|---|------|
| `--dataset` | str | 事前定義されたデータセット名（`config.py`参照） |
| `--input-file` | str | 入力ファイルのパス（`.txt`, `.csv`） |

### CSV処理パラメータ（CSVファイル入力時）🆕

| オプション | 型 | デフォルト | 説明 |
|----------|---|----------|------|
| `--text-column` | str | `text` | テキストを含むカラム名 |
| `--combine-rows` | flag | False | 複数行を結合してチャンク化する |
| `--block-size` | int | 400 | 結合する行数（`--combine-rows`指定時に有効） |

### Q/A生成パラメータ

| オプション | 型 | デフォルト | 説明 |
|----------|---|----------|------|
| `--model` | str | `gemini-2.0-flash` | 使用するGeminiモデル |
| `--max-docs` | int | None | 処理する最大文書数 |
| **✨ `--use-smart-generation`** | **flag** | **True** | **スマートQ/A生成を使用（デフォルト有効）** |
| **✨ `--no-smart-generation`** | **flag** | **-** | **従来方式に切り替え（トークン数ベース）** |
| `--use-celery` | flag | False | Celery並列処理を使用 |
| `-c`, `--concurrency` | int | 8 | 並列タスク数（start_celery.sh -c と同じ値を推奨） |
| `--celery-workers` | int | 1 | (非推奨) Celeryワーカープロセス数チェック用 |
| `--batch-chunks` | int | 3 | 1回のAPIで処理するチャンク数 |
| `--merge-chunks` | flag | True | 小さいチャンクを統合（デフォルト有効） |
| `--overlap-tokens` | int | 0 | チャンク間の重複トークン数 |
| `--use-similarity` | flag | False | ベクトル類似度によるセマンティック分割 |
| `--similarity-threshold` | float | 0.7 | セマンティック分割の類似度閾値 |

### Qdrant登録パラメータ

| オプション | 型 | デフォルト | 説明 |
|----------|---|----------|------|
| `--collection` | str | **必須** | 登録先コレクション名 |
| `--recreate` | flag | False | コレクションを再作成 |
| `--batch-size` | int | 100 | Embeddingバッチサイズ |
| `--provider` | str | `gemini` | Embeddingプロバイダー |

### 出力パラメータ

| オプション | 型 | デフォルト | 説明 |
|----------|---|----------|------|
| `--output` | str | `qa_output/pipeline` | Q/AペアCSVの出力ディレクトリ |
| `--ui-output` | str | `qa_output` | UI用正規化CSVの出力ディレクトリ |

---

## 📝 実行例

### 例1: 小規模テキストファイルの処理（スマート生成）

```bash
python make_qa_register_qdrant.py \
  --input-file sample.txt \
  --collection sample_docs \
  --recreate
```

**期待される動作**:
- ✅ スマート生成モードで実行
- ✅ 技術的なチャンクから4-5個のQ/A
- ✅ 単純なチャンクから1-2個のQ/A
- ✅ メタ情報のみのチャンクは0個（スキップ）

### 例2: 大規模CSVファイルの処理（従来方式）

```bash
python make_qa_register_qdrant.py \
  --input-file large_data.csv \
  --collection large_docs \
  --max-docs 10000 \
  --no-smart-generation \
  --use-celery \
  --concurrency 16 \
  --recreate
```

**期待される動作**:
- ✅ 従来方式（高速・低コスト）
- ✅ Celery並列処理で高速化
- ✅ 全チャンクから固定数（2-8個）のQ/A生成

### 例3: Q/Aペア既存CSVの登録（Phase 1スキップ）

```bash
python make_qa_register_qdrant.py \
  --input-file existing_qa_pairs.csv \
  --collection qa_collection \
  --recreate
```

**期待される動作**:
- ✅ Phase 1をスキップ
- ✅ 直接Qdrant登録
- ✅ 高速処理

### 例4: データセット指定（スマート生成 + Celery）

```bash
python make_qa_register_qdrant.py \
  --dataset wikipedia_ja \
  --collection wiki_ja \
  --max-docs 500 \
  --use-celery \
  --concurrency 16 \
  --recreate
```

**期待される動作**:
- ✅ スマート生成で高品質Q/A
- ✅ Celery並列処理で処理時間短縮
- ✅ Wikipedia記事の主要トピックを確実にカバー

### 例5: セマンティック分割 + スマート生成

```bash
python make_qa_register_qdrant.py \
  --input-file technical_doc.txt \
  --collection tech_docs \
  --use-similarity \
  --similarity-threshold 0.8 \
  --overlap-tokens 100 \
  --recreate
```

**期待される動作**:
- ✅ 意味的に自然なチャンク境界
- ✅ スマート生成で各チャンクの特性に応じたQ/A数
- ✅ オーバーラップで文脈保持

### 例6: 従来方式への切り替え例

```bash
# スマート生成（デフォルト）
python make_qa_register_qdrant.py \
  --input-file doc.txt \
  --collection docs \
  --recreate

# 従来方式に切り替え
python make_qa_register_qdrant.py \
  --input-file doc.txt \
  --collection docs \
  --no-smart-generation \
  --recreate
```

### 例7: CSV行結合オプションの使用 🆕

```bash
python make_qa_register_qdrant.py \
  --input-file OUTPUT/cc_news_5per.csv \
  --collection cc_news_5per \
  --use-celery \
  --model gemini-2.5-flash \
  --concurrency 4 \
  --text-column text \
  --combine-rows \
  --block-size 400 \
  --recreate
```

**期待される動作**:
- ✅ CSVの `text` カラムから400行ずつ結合
- ✅ 結合されたテキストからQ/A生成
- ✅ 小さなレコードが多いCSVを効率的に処理

### 例8: カスタムテキストカラムの指定 🆕

```bash
python make_qa_register_qdrant.py \
  --input-file articles.csv \
  --collection articles \
  --text-column content \
  --combine-rows \
  --block-size 200 \
  --recreate
```

**期待される動作**:
- ✅ `content` カラムをテキストとして使用
- ✅ 200行ごとに結合してチャンク化
- ✅ デフォルトの `text` カラム以外にも対応

---

## 🎯 推奨される使い分け

### スマート生成を使うべき場合（デフォルト）

#### シナリオ1: 品質重視のプロジェクト

- **使用例**: 顧客向けFAQシステム、技術文書検索
- **理由**: トピックカバレッジが確実、高品質な回答

```bash
python make_qa_register_qdrant.py \
  --input-file faq_documents.txt \
  --collection customer_faq \
  --recreate
```

#### シナリオ2: 少〜中量データ（100-1,000チャンク）

- **使用例**: 企業ドキュメント、製品マニュアル
- **理由**: コストが許容範囲内、品質向上のメリット大

```bash
python make_qa_register_qdrant.py \
  --input-file product_manual.csv \
  --collection product_docs \
  --max-docs 500 \
  --recreate
```

#### シナリオ3: 多様なコンテンツ

- **使用例**: 技術文書と一般文書が混在
- **理由**: 各チャンクの特性に応じた適応的なQ/A生成

```bash
python make_qa_register_qdrant.py \
  --input-file mixed_content.txt \
  --collection mixed_docs \
  --recreate
```

#### シナリオ4: メタデータ活用が必要

- **使用例**: トピック別フィルタリング、重要度ソート
- **理由**: topic, importance_score, complexityのメタデータ

```bash
python make_qa_register_qdrant.py \
  --input-file research_papers.csv \
  --collection research \
  --recreate
```

### 従来方式を使うべき場合（`--no-smart-generation`）

#### シナリオ1: 大規模データセット（10,000+チャンク）

- **使用例**: Wikipediaデータセット、ニュース記事コレクション
- **理由**: API コスト最小化、処理時間短縮

```bash
python make_qa_register_qdrant.py \
  --dataset wikipedia_ja \
  --collection wiki_ja \
  --max-docs 50000 \
  --no-smart-generation \
  --use-celery \
  --concurrency 24 \
  --recreate
```

#### シナリオ2: コスト最適化が必要

- **使用例**: 予算制限のあるプロジェクト
- **理由**: API呼び出しが半分、コストも約半分

```bash
python make_qa_register_qdrant.py \
  --input-file large_corpus.csv \
  --collection corpus \
  --no-smart-generation \
  --recreate
```

#### シナリオ3: 高速処理が必要

- **使用例**: リアルタイム処理、バッチジョブ
- **理由**: 処理時間が約半分

```bash
python make_qa_register_qdrant.py \
  --input-file daily_news.csv \
  --collection news \
  --no-smart-generation \
  --use-celery \
  --concurrency 16 \
  --recreate
```

#### シナリオ4: 安定性重視（本番環境）

- **使用例**: ミッションクリティカルなシステム
- **理由**: シンプルなロジック、デバッグが容易

```bash
python make_qa_register_qdrant.py \
  --input-file production_data.csv \
  --collection production \
  --no-smart-generation \
  --recreate
```

### CSV行結合を使うべき場合 🆕

#### シナリオ: 小さなレコードが多いCSV

- **使用例**: ニュース記事、ツイート、短文データ
- **理由**: 1行=1レコードでは内容が薄すぎてQ/A生成の効率が悪い

```bash
python make_qa_register_qdrant.py \
  --input-file news_articles.csv \
  --collection news \
  --text-column text \
  --combine-rows \
  --block-size 400 \
  --recreate
```

### 使い分けのガイドライン

| 判断基準 | スマート生成 | 従来方式 |
|---------|------------|---------|
| **データ量** | < 1,000チャンク | > 10,000チャンク |
| **予算** | 余裕あり | 制約あり |
| **処理時間** | 許容可能 | 最小化したい |
| **品質** | 最優先 | 標準で十分 |
| **コンテンツ** | 多様・複雑 | 均質・単純 |
| **メタデータ** | 必要 | 不要 |
| **環境** | 開発・テスト | 本番 |

### ハイブリッドアプローチ（推奨）

重要なドキュメントのみスマート生成、その他は従来方式：

```bash
# ステップ1: 重要ドキュメント（スマート生成）
python make_qa_register_qdrant.py \
  --input-file important_docs.csv \
  --collection all_docs \
  --recreate

# ステップ2: その他ドキュメント（従来方式）
python make_qa_register_qdrant.py \
  --input-file other_docs.csv \
  --collection all_docs \
  --no-smart-generation
  # --recreateを指定しないことで追加登録
```

---

## ⚠️ 注意事項

### 1. スマート生成のコストと処理時間

- **API コスト**: 約2倍（チャンクごとに2回のLLM呼び出し）
- **処理時間**: 約2倍（分析ステップの追加）

### 2. Celery並列処理の制限

- スマート生成 + Celery の場合、`celery_tasks.py`が`use_smart_generation`パラメータに対応している必要があります
- 現在の実装では対応していない可能性があるため、同期処理を推奨

### 3. 0個生成の扱い

- スマート生成では、メタ情報のみのチャンクから0個のQ/Aを生成する場合があります
- これは意図的な動作であり、不要なQ/Aを生成しないための機能です

### 4. バッチサイズの推奨

- スマート生成の場合、`--batch-chunks=1`を推奨（各チャンクを個別に分析）
- 従来方式の場合、`--batch-chunks=3`（デフォルト）で問題なし

### 5. CSV行結合時の注意 🆕

- `--combine-rows` は `--input-file` が CSV の場合のみ有効
- `--text-column` で指定したカラムが存在しない場合はエラー
- `--block-size` が大きすぎると1チャンクあたりのテキスト量が多くなりすぎる可能性

---

## 🔧 トラブルシューティング

### 問題1: スマート生成が遅い

**症状**:
```
Phase 1の処理時間が非常に長い
```

**解決策**:
```bash
# 従来方式に切り替え
python make_qa_register_qdrant.py \
  --input-file doc.txt \
  --collection docs \
  --no-smart-generation \
  --recreate
```

### 問題2: API コストが高い

**症状**:
```
Gemini APIの使用量が予想以上に多い
```

**解決策**:
```bash
# 1. 従来方式に切り替え
--no-smart-generation

# 2. max-docsで制限
--max-docs 100

# 3. バッチサイズを調整
--batch-chunks 5
```

### 問題3: メモリ不足

**症状**:
```
MemoryError: Unable to allocate array
```

**解決策**:
```bash
# 1. バッチサイズを小さく
--batch-size 50

# 2. max-docsで制限
--max-docs 1000

# 3. Celery並列処理を使わない
# --use-celery を削除
```

### 問題4: Celeryワーカーが応答しない

**症状**:
```
RuntimeError: Celery workers are not running
```

**解決策**:
```bash
# 1. ワーカーを起動（別ターミナル）
./start_celery.sh restart -c 8 --flower

# 2. ワーカー数を確認
celery -A celery_tasks inspect active

# 3. 同期処理に切り替え
# --use-celery を削除
```

### 問題5: Q/A数が0個ばかり

**症状**:
```
分析完了: Q/A数=0, 重要度=0.20
Q/A生成スキップ（qa_count=0）
```

**原因**:
- チャンクの内容が補足情報のみ
- メタ情報のみのチャンク

**解決策**:
```bash
# 1. 従来方式に切り替え（固定数生成）
--no-smart-generation

# 2. チャンクサイズを大きく
# config.pyで chunk_size を調整

# 3. セマンティック分割を使用
--use-similarity --similarity-threshold 0.7
```

### 問題6: トピックが取得できない

**症状**:
```
'topic'フィールドがない
```

**原因**:
- 従来方式を使用している
- CSVに既存Q/Aがあり、topicカラムがない

**解決策**:
```bash
# スマート生成に切り替え
python make_qa_register_qdrant.py \
  --input-file doc.txt \
  --collection docs \
  --recreate
  # --no-smart-generation を削除
```

### 問題7: エラー「Column 'question' or 'answer' not found」

**症状**:
```
ValueError: CSV must have 'text' or 'Combined_Text' column,
or both 'question' and 'answer' columns
```

**原因**:
- CSVファイルに必要なカラムがない

**解決策**:
```bash
# CSVのヘッダーを確認
head -n 1 your_file.csv

# 必要なカラムのいずれかを含める:
# - text または Combined_Text （テキストのみの場合）
# - question と answer （Q/Aペアの場合）
```

### 問題8: 「カラム 'xxx' が見つかりません」エラー 🆕

**症状**:
```
ValueError: カラム 'text' が見つかりません。利用可能: ['content', 'title', ...]
```

**原因**:
- `--text-column` で指定したカラム名がCSVに存在しない

**解決策**:
```bash
# CSVのカラムを確認
head -n 1 your_file.csv

# 正しいカラム名を指定
python make_qa_register_qdrant.py \
  --input-file your_file.csv \
  --text-column content \
  --combine-rows \
  --collection my_docs \
  --recreate
```

### 問題9: 行結合後のチャンク数が少なすぎる 🆕

**症状**:
```
📦 行結合処理を開始
   - 入力行数: 10000
   - 生成チャンク数: 2
```

**原因**:
- `--block-size` が大きすぎる

**解決策**:
```bash
# block-size を小さくする
python make_qa_register_qdrant.py \
  --input-file your_file.csv \
  --text-column text \
  --combine-rows \
  --block-size 200 \  # 400 → 200 に変更
  --collection my_docs \
  --recreate
```

---

## 📚 関連ドキュメント

- `qa_generation_comparison.md` - スマート生成と従来方式の詳細比較
- `smart_generation_upgrade_summary.md` - v2.1改修内容の詳細
- `smart_qa_generator.py` - SmartQAGeneratorクラスの実装
- `generation.py` - QAGeneratorクラス（統合版）の実装
- `pipeline.py` - QAPipelineクラスの実装

---

## 📝 バージョン履歴

| バージョン | 日付 | 変更内容 |
|----------|------|---------|
| **v2.2** | 2025-01-24 | ✨ CSV処理オプション追加（`--text-column`, `--combine-rows`, `--block-size`） |
| v2.1 | 2025-01-20 | ✨ スマート生成デフォルト化、`--use-smart-generation`/`--no-smart-generation`追加 |
| v2.0 | 2025-01-19 | 入力処理統一、`--input-file`追加、CSVカラム自動判定 |
| v1.0 | 2025-01-15 | 初版リリース |

---

**作成日**: 2025-01-20
**最終更新**: 2025-01-24
**バージョン**: v2.2
**作成者**: AI Assistant
