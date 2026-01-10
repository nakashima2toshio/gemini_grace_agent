# 改修版 使用方法ガイド

## 📋 改修内容サマリー

チャンクCSV読み込み機能を追加し、2段階処理を可能にしました。

### ✅ 改修ファイル一覧

1. **csv_to_chunks_text_para_modified.py** - CSV出力機能追加
2. **pipeline_modified.py** - CSV読み込み機能追加
3. **make_qa_modified.py** - `--input-chunks` 引数追加
4. **make_qa_register_qdrant_modified.py** - `--input-chunks` 引数追加

---

## 🚀 使用方法

### **パターン1: 2段階実行（チャンクCSV経由）**

#### **Step 1: チャンク作成（LLMベース）**

```bash
# テキストファイルからチャンクCSVを作成
python -m chunking.csv_to_chunks_text_para_modified \
  -i wiki_data.txt \
  -o wiki_chunks.csv \
  -w 8
```

**出力例: wiki_chunks.csv**
```csv
chunk_id,text,tokens,chunk_idx,dataset_type,type,sentence_count,source_file
wiki_data_chunk_0,"第1章 概要...",150,0,wiki_data,llm_chunk,5,wiki_data.txt
wiki_data_chunk_1,"機械学習の基礎...",180,1,wiki_data,llm_chunk,6,wiki_data.txt
```

#### **Step 2: Q/A生成**

```bash
# チャンクCSVからQ/A生成
python make_qa_modified.py \
  --input-chunks wiki_chunks.csv \
  --model gemini-2.0-flash \
  --analyze-coverage
```

#### **Step 3: Qdrant登録（オプション）**

```bash
# Q/AペアをQdrantに登録
python make_qa_register_qdrant_modified.py \
  --input-chunks wiki_chunks.csv \
  --collection wiki_qa \
  --recreate \
  --use-celery \
  --celery-workers 16
```

---

### **パターン2: 従来の1段階実行（変更なし）**

```bash
# データセットから直接Q/A生成
python make_qa_modified.py \
  --dataset wikipedia_ja \
  --use-celery \
  --celery-workers 8

# データセットからQ/A生成 & Qdrant登録
python make_qa_register_qdrant_modified.py \
  --dataset wikipedia_ja \
  --collection wiki_qa \
  --recreate
```

---

## 📊 CSV形式の仕様

### **チャンクCSV（csv_to_chunks_text_para_modified.py の出力）**

| カラム名 | 型 | 必須 | 説明 |
|---------|---|------|------|
| chunk_id | string | ✅ | チャンク識別子 |
| text | string | ✅ | チャンクのテキスト |
| tokens | int | ✅ | トークン数 |
| chunk_idx | int | ✅ | チャンクインデックス |
| dataset_type | string | ✅ | データセット種別 |
| type | string | ⚪ | チャンク種別（llm_chunk等） |
| sentence_count | int | ⚪ | 含まれる文の数 |
| source_file | string | ⚪ | 元ファイル名 |

### **Q/AペアCSV（make_qa_modified.py の出力）**

| カラム名 | 型 | 必須 | 説明 |
|---------|---|------|------|
| question | string | ✅ | 質問文 |
| answer | string | ✅ | 回答文 |
| question_type | string | ⚪ | 質問タイプ |
| difficulty | string | ⚪ | 難易度 |

---

## 🔧 配置方法

### **1. チャンク処理スクリプト**

```bash
# 元のファイルを置き換え
cp csv_to_chunks_text_para_modified.py chunking/csv_to_chunks_text_para.py
```

### **2. パイプライン**

```bash
# 元のファイルを置き換え
cp pipeline_modified.py qa_generation/pipeline.py
```

### **3. CLIスクリプト**

```bash
# 元のファイルを置き換え
cp make_qa_modified.py make_qa.py
cp make_qa_register_qdrant_modified.py make_qa_register_qdrant.py
```

---

## 💡 実行例

### **例1: Wikipediaデータのチャンク化 → Q/A生成**

```bash
# Step 1: チャンク作成
python -m chunking.csv_to_chunks_text_para \
  -i wikipedia_ja.txt \
  -o wikipedia_chunks.csv \
  -w 8 \
  -v

# Step 2: チャンク確認（オプション）
head -n 5 wikipedia_chunks.csv

# Step 3: Q/A生成
python make_qa.py \
  --input-chunks wikipedia_chunks.csv \
  --batch-chunks 3 \
  --analyze-coverage
```

### **例2: ニュースデータの一括処理**

```bash
# チャンク作成 → Q/A生成 → Qdrant登録（1コマンド）
python make_qa_register_qdrant.py \
  --input-chunks news_chunks.csv \
  --collection news_qa \
  --use-celery \
  --celery-workers 24 \
  --recreate
```

---

## ⚠️ 注意事項

### **1. チャンク作成パラメータの扱い**

`--input-chunks` 使用時は以下のパラメータは**無視**されます:
- `--overlap-tokens`
- `--use-similarity`
- `--similarity-threshold`

これらはチャンク作成時に適用済みのため。

### **2. CSV形式の互換性**

- **必須カラム**: `chunk_id`, `text`, `tokens`, `chunk_idx`
- 不足している場合はエラーになります

### **3. エラーハンドリング**

```bash
# チェックポイントから再開
python -m chunking.csv_to_chunks_text_para \
  --resume JOB_ID \
  -i input.txt \
  -o output.csv
```

---

## 📈 パフォーマンス比較

| 処理方式 | チャンク精度 | 処理速度 | コスト |
|---------|------------|---------|--------|
| **LLMチャンク（新）** | ⭐⭐⭐⭐⭐ | ⭐⭐☆☆☆ | 高 |
| セマンティック（既存） | ⭐⭐⭐☆☆ | ⭐⭐⭐⭐⭐ | 低 |

**推奨:**
- 品質重視 → LLMチャンク（新方式）
- 速度重視 → セマンティック（既存方式）

---

## 🐛 トラブルシューティング

### **問題1: CSVが読み込めない**

```python
ValueError: 必須カラムが不足しています: ['chunk_id', 'text']
```

**対処法:**
- CSVヘッダーを確認
- csv_to_chunks_text_para_modified.py で再作成

### **問題2: チャンク数が0**

```
logger.error("チャンクが作成されませんでした")
```

**対処法:**
- 入力ファイルの内容を確認
- `-v` オプションで詳細ログを確認

### **問題3: API呼び出しエラー**

```
GOOGLE_API_KEYが設定されていません
```

**対処法:**
```bash
export GOOGLE_API_KEY="your-api-key"
```

---

## 📞 サポート

問題が発生した場合:
1. `-v` オプションで詳細ログを確認
2. `checkpoints/` ディレクトリのログを確認
3. CSVファイルの形式を確認

---

## 🔄 次のステップ

改修第二弾の候補:
- [ ] チャンク統合機能の追加
- [ ] オーバーラップ機能のLLMチャンク対応
- [ ] チャンク品質評価機能
- [ ] バッチ処理の最適化
### ===========================================================

## make_qa_register_qdrant.py - 処理フローとデータフロー詳細説明

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
              │                      │  CSVファイル確認  │
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
              │                │          │   Q/A生成を実行       │
              │                │          └──────────┬──────────┘
              │                │                     │
      ┌───────┴───────┐        │                     │
      │  QAPipeline   │        │                     │
      │  Q/A生成を実行  │        │                     │
      └───────┬───────┘        │                     │
              │                │                     │
              └────────────────┴─────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Q/AペアCSV生成    │
                    │  (question/answer)│
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    │  Qdrant登録処理    │
                    │  1. ベクトル化      │
                    │  2. ポイント構築    │
                    │  3. アップサート    │
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