# make_qa_register_qdrant.py - 処理フローとデータフロー詳細説明

## 📊 全体のデータフロー図

```
┌─────────────────────────────────────────────────────────────────┐
│                     入力データソース                               │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
       [オプション A]                   [オプション B]
      --dataset 指定                   --input-csv 指定
              │                               │
              │                      ┌────────┴────────┐
              │                      │  CSVファイル確認   │
              │                      │  カラムを判定     │
              │                      └────────┬────────┘
              │                               │
              │                    ┌──────────┴──────────┐
              │                    │                     │
              │              [ケース 1]            [ケース 2]
              │          question/answer       text/Combined_Text
              │           カラムあり               カラムのみ
              │                │                     │
              │                │          ┌──────────┴──────────┐
              │                │          │   QAPipeline 起動    │
              │                │          │   Q/A生成を実行      │
              │                │          └──────────┬──────────┘
              │                │                     │
      ┌───────┴───────┐        │                     │
      │  QAPipeline    │        │                     │
      │  Q/A生成を実行  │        │                     │
      └───────┬───────┘        │                     │
              │                │                     │
              └────────────────┴─────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Q/AペアCSV生成    │
                    │  (question/answer) │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Qdrant登録処理    │
                    │  1. ベクトル化     │
                    │  2. ポイント構築   │
                    │  3. アップサート   │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │  登録完了          │
                    │  + UI用CSV保存     │
                    └───────────────────┘
```

## 🔍 詳細な処理フロー

### Phase 1: Q/A生成フェーズ

#### パターン1: `--dataset` を使用する場合

```python
# コマンド例
python make_qa_register_qdrant.py \
  --dataset wikipedia_ja \
  --collection qa_wikipedia_ja \
  --use-celery \
  --celery-workers 24 \
  --recreate

# 処理フロー
1. DATASET_CONFIGS から設定を読み込み
   ├─ データソースパス
   ├─ チャンクサイズ
   └─ その他の設定

2. QAPipeline を初期化
   └─ dataset_name="wikipedia_ja"

3. QAPipeline.run() 実行
   ├─ データ読み込み
   ├─ チャンク化
   ├─ Q/Aペア生成 (Celery並列処理)
   └─ CSVファイル保存
       └─ qa_output/pipeline/qa_pairs_YYYYMMDD_HHMMSS.csv

4. 生成されたCSVパスを取得
   └─ generated_csv = result['saved_files']['qa_csv']
```

#### パターン2: `--input-csv` を使用する場合（修正版）

```python
# コマンド例
python make_qa_register_qdrant.py \
  --input-csv OUTPUT/cc_news_5per.csv \
  --collection cc_news_5per \
  --use-celery \
  --celery-workers 16 \
  --recreate

# 処理フロー
1. CSVファイルを読み込み
   └─ df_check = pd.read_csv(args.input_csv)

2. カラムを確認
   ├─ has_qa_columns = 'question' in df AND 'answer' in df
   └─ has_text_columns = 'text' in df OR 'Combined_Text' in df

3-A. question/answer カラムが存在する場合
   └─ Q/A生成をスキップ
       └─ generated_csv = args.input_csv
       └─ 直接Phase 2へ

3-B. text/Combined_Text カラムのみの場合
   └─ QAPipeline を起動
       ├─ dataset_name=None
       ├─ input_file=args.input_csv
       └─ Q/A生成を実行
           ├─ テキストを読み込み
           ├─ チャンク化
           ├─ Q/Aペア生成 (Celery並列処理)
           └─ CSVファイル保存
               └─ qa_output/pipeline/qa_pairs_YYYYMMDD_HHMMSS.csv

3-C. どちらのカラムもない場合
   └─ エラー終了
```

### Phase 2: Qdrant登録フェーズ

```python
# すべてのパターン共通
1. generated_csv を読み込み
   └─ df = pd.read_csv(csv_path)

2. question/answer カラム確認
   └─ 必須: 'question' AND 'answer'

3. ベクトル化対象テキストを準備
   └─ texts = question + "\n" + answer

4. Qdrantクライアント作成
   └─ client = create_qdrant_client()

5. コレクション作成/再作成
   └─ create_or_recreate_collection_for_qdrant()

6. バッチ処理ループ
   for batch in batches(df, batch_size):
       ├─ ベクトル化
       │   └─ vectors = embed_texts_for_qdrant(batch_texts)
       │
       ├─ ポイント構築
       │   └─ points = build_points_for_qdrant(
       │           batch_df, vectors, domain, source_file, start_index)
       │
       └─ Qdrantへアップサート
           └─ upsert_points_to_qdrant(client, collection_name, points)

7. UI用正規化CSV作成
   └─ qa_output/<正規化ファイル名>.csv
```

## 📁 データ形式の変換

### 入力データの種類

#### 種類1: テキストCSV（text_5per_2_csv.py で作成）
```csv
text,Combined_Text
"記事本文1...","記事本文1..."
"記事本文2...","記事本文2..."
```

#### 種類2: Q/AペアCSV（make_qa.py で作成）
```csv
question,answer,chunk_id,source
"質問1?","回答1...","chunk_0","cc_news.csv"
"質問2?","回答2...","chunk_1","cc_news.csv"
```

### 出力データ

#### Qdrantに登録されるポイント
```python
{
    "id": "cc_news_5per_0",  # <collection>_<index>
    "vector": [0.123, 0.456, ...],  # 768次元
    "payload": {
        "question": "質問内容",
        "answer": "回答内容",
        "domain": "cc_news_5per",
        "source": "cc_news_5per.csv",  # 正規化されたファイル名
        "chunk_id": "chunk_0"
    }
}
```

## 🎯 使用例とデータの流れ

### 例1: テキストCSVからQ/A生成 & 登録（ワンステップ）

```bash
# 元のコマンド（エラーになっていた）
python make_qa_register_qdrant.py \
  --input-csv OUTPUT/cc_news_5per.csv \  # text/Combined_Text のみ
  --collection cc_news_5per \
  --use-celery \
  --celery-workers 16 \
  --recreate

# 修正後の動作
┌─────────────────────────────┐
│ OUTPUT/cc_news_5per.csv     │
│ (text, Combined_Text)       │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ カラムチェック               │
│ → text/Combined_Text のみ   │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ QAPipeline 起動             │
│ input_file=OUTPUT/...csv    │
│ use_celery=True             │
│ celery_workers=16           │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Q/A生成実行                 │
│ 1. テキスト読み込み          │
│ 2. チャンク化               │
│ 3. Gemini APIでQ/A生成      │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Q/AペアCSV生成              │
│ qa_output/pipeline/         │
│ qa_pairs_20260110_133045.csv│
│ (question, answer)          │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Qdrant登録処理              │
│ 1. ベクトル化               │
│ 2. ポイント構築             │
│ 3. アップサート             │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ 完了                        │
│ コレクション: cc_news_5per   │
└─────────────────────────────┘
```

### 例2: 既存Q/AペアCSVから直接登録

```bash
python make_qa_register_qdrant.py \
  --input-csv qa_output/qa_pairs_fineweb_edu_ja.csv \  # question/answer あり
  --collection fineweb_edu_ja \
  --recreate

# 動作
┌─────────────────────────────┐
│ qa_pairs_fineweb_edu_ja.csv │
│ (question, answer)          │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ カラムチェック               │
│ → question/answer あり      │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Q/A生成スキップ             │
│ generated_csv = 入力ファイル │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Qdrant登録処理              │
│ （Phase 2へ直行）           │
└─────────────────────────────┘
```

## 💡 重要なポイント

### 1. QAPipeline の役割
- **make_qa.py**: スタンドアロンのQ/A生成ツール
- **make_qa_register_qdrant.py**: QAPipeline を内部で使用
  - `input_file` パラメータで任意のCSVを処理可能
  - テキストカラム（text/Combined_Text）からQ/A生成

### 2. カラムの判定ロジック
```python
# 優先順位
if 'question' in df AND 'answer' in df:
    # Q/A生成スキップ
elif 'text' in df OR 'Combined_Text' in df:
    # Q/A生成実行
else:
    # エラー
```

### 3. ファイル名の正規化
```python
# 入力: qa_pairs_20260110_133045.csv
# 出力: qa_pairs.csv（日時サフィックス削除）
# 理由: UI(agent_rag.py)での参照を安定化
```

### 4. Celery並列処理
- `--use-celery`: 有効化フラグ
- `--celery-workers`: ワーカー数（デフォルト: 8）
- Q/A生成の高速化に使用
- Qdrant登録には影響しない

## 🔧 トラブルシューティング

### エラー: "Q/Aカラムが見つかりません"
**原因**: text/Combined_Textカラムを持つCSVを指定したが、旧バージョンを使用
**解決**: 修正版のスクリプトを使用

### エラー: "CSVファイルに必要なカラムが見つかりません"
**原因**: question/answer も text/Combined_Text もないCSV
**解決**: 正しいCSVファイルを指定

### Q/A生成が実行されない
**原因**: 既にquestion/answerカラムが存在
**動作**: 正常（生成をスキップして登録へ進む）