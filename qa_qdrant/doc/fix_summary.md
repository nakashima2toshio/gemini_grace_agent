# タイムアウトエラー診断ガイド

**日時**: 2026-01-20
**症状**: タスクが120秒でタイムアウトする（インポートテストは成功）

---

## 🎯 現状の確認

### ✅ 成功している項目

1. インポートテスト成功（`python celery_config.py`）
2. インポートテスト成功（`python celery_tasks.py`）
3. Celeryワーカー起動（1個、concurrency=8）
4. タスク投入成功

### ❌ 失敗している項目

1. タスクが120秒でタイムアウト
2. Q/Aペアが生成されない

---

## 🔍 診断ステップ

### ステップ1: ワーカーログの確認（最重要）

```bash
# ワーカーログ診断スクリプトを実行
python check_worker_logs.py
```

**確認すべきポイント**:
- `[ワーカー] タスク開始`が出力されているか
- `[ワーカー] インポート成功`が出力されているか
- `[ワーカー] Q/A生成開始`が出力されているか
- エラーメッセージがないか

**期待される出力**:
```
✅ タスク開始: 1件
   [INFO] [ワーカー] タスク開始: chunk=test_chunk_0, provider=gemini, smart=True

✅ インポート成功: 1件
   [INFO] [ワーカー] インポート成功: qa_generation.generation

❌ Q/A生成開始: 見つかりません
```

### ステップ2: Celery基本動作テスト

```bash
# 簡易タスクでCeleryの基本動作を確認
python test_simple_task.py
```

**期待される出力**:
```
[ステップ4] タスクの実行
簡易タスクを投入中...
タスクID: 12345678-1234-1234-1234-123456789abc

[ステップ5] 結果の取得（タイムアウト: 10秒）
✅ 成功!
   結果: 30
   期待値: 30

🎉 Celeryワーカーは正常に動作しています！
```

**もし失敗した場合**:
```
❌ 失敗: The operation timed out.

考えられる原因:
  1. ワーカーがタスクを受け取っていない
  2. ワーカーでエラーが発生している
  3. タイムアウトが短すぎる
```

→ この場合、Celery自体に問題があります

### ステップ3: 手動でログを確認

```bash
# 最新のログファイルを確認
tail -f logs/celery_qa_*.log

# または、全体を確認
cat logs/celery_qa_*.log | less
```

**検索すべきキーワード**:
```bash
# エラーメッセージを検索
grep -i "error" logs/celery_qa_*.log

# タスク実行のログを検索
grep "\[ワーカー\]" logs/celery_qa_*.log

# API関連のエラーを検索
grep -i "api\|gemini\|rate" logs/celery_qa_*.log
```

---

## 🔧 考えられる原因と対処法

### 原因1: ワーカーがタスクを受け取っていない

**症状**:
- ワーカーログに`[ワーカー] タスク開始`が出力されない
- `check_worker_logs.py`で「タスク開始: 見つかりません」

**対処法**:
```bash
# ワーカーを完全に停止
pkill -9 -f "celery worker"

# キューをクリア
python -c "from celery_config import app; app.control.purge()"

# ワーカーを再起動
./start_celery.sh start -w 8

# 再テスト
python test_simple_task.py
```

### 原因2: generate_qa_dataset関数でエラーが発生

**症状**:
- ワーカーログに`[ワーカー] タスク開始`は出力される
- しかし`[ワーカー] Q/A生成開始`が出力されない
- または`[ワーカー] タスクエラー`が出力される

**対処法**:
```bash
# generate_qa_datasetを直接テスト
python -c "
from qa_generation.generation import generate_qa_dataset

chunk = {
    'id': 'test',
    'text': 'This is a test.',
    'tokens': 10,
    'doc_id': 'test',
    'chunk_idx': 0
}

try:
    qa_pairs = generate_qa_dataset(
        chunks=[chunk],
        dataset_type='test',
        model='gemini-2.0-flash',
        config={'type': 'test', 'qa_per_chunk': 1},
        provider='gemini',
        use_smart_generation=True
    )
    print(f'成功: {len(qa_pairs)}個のQ/Aペア生成')
except Exception as e:
    print(f'エラー: {e}')
    import traceback
    traceback.print_exc()
"
```

### 原因3: Gemini API のエラー

**症状**:
- ワーカーログに「API error」「rate limit」「quota」などのメッセージ

**対処法**:
```bash
# APIキーの確認
python -c "import os; print('GOOGLE_API_KEY:', 'あり' if os.getenv('GOOGLE_API_KEY') else 'なし')"

# Gemini APIの直接テスト
python -c "
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash')

try:
    response = model.generate_content('Hello')
    print('✅ Gemini API動作確認OK')
    print(f'応答: {response.text[:100]}...')
except Exception as e:
    print(f'❌ Gemini APIエラー: {e}')
"
```

### 原因4: タイムアウト設定が短すぎる

**症状**:
- ワーカーログに処理中のログがあるが、タイムアウトする

**対処法**:
```bash
# テストのタイムアウトを延長
# test_celery_integration.py の120秒を300秒に変更

# または、celery_config.pyのタイムアウトを確認
python -c "from celery_config import CeleryConfig; print(f'task_time_limit: {CeleryConfig.task_time_limit}秒')"
```

---

## 📝 診断結果の記録

以下の情報を記録してください:

### 1. ワーカーログの内容

```bash
python check_worker_logs.py > diagnosis_result.txt
```

### 2. 簡易タスクのテスト結果

```bash
python test_simple_task.py >> diagnosis_result.txt
```

### 3. 環境情報

```bash
echo "=== 環境情報 ===" >> diagnosis_result.txt
python --version >> diagnosis_result.txt
pip list | grep -E "celery|redis|google" >> diagnosis_result.txt
```

---

## 🚀 次のステップ

### ケースA: 簡易タスクが成功した場合

→ Celeryは正常に動作しています
→ `generate_qa_dataset`関数に問題がある可能性が高い
→ 原因2の対処法を実行

### ケースB: 簡易タスクも失敗した場合

→ Celery自体に問題があります
→ 原因1の対処法を実行
→ Redisの再起動も試す: `brew services restart redis`

### ケースC: ワーカーログにAPIエラーがある場合

→ Gemini APIに問題があります
→ 原因3の対処法を実行
→ APIキーやクォータを確認

---

## 📞 追加の診断が必要な場合

以下の情報を提供してください:

1. `python check_worker_logs.py`の出力
2. `python test_simple_task.py`の出力
3. ワーカーログの全文（または最後の500行）
4. `cat logs/celery_qa_*.log | tail -500`

---

**作成日**: 2026-01-20
**更新日**: 2026-01-20
