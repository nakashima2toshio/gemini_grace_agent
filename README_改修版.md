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
