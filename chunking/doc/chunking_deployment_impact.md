# csv_to_chunks_text_para_modified.py のchunking/配置 影響分析

## 📋 目次

1. [現状の構成](#現状の構成)
2. [配置による影響](#配置による影響)
3. [必要な変更一覧](#必要な変更一覧)
4. [変更手順](#変更手順)
5. [テスト項目](#テスト項目)
6. [ロールバック手順](#ロールバック手順)

---

## 📁 現状の構成

### **chunking/ ディレクトリ構造**

```
chunking/
├── __init__.py              # パッケージ初期化
├── async_api_client.py      # 非同期APIクライアント
├── checkpoint_manager.py    # チェックポイント管理
├── csv_to_chunks_text_para.py  # ← これを置き換える
├── models.py                # Pydanticモデル
├── prompts.py               # プロンプト定義
├── requirements.txt         # 依存パッケージ
└── utils.py                 # ユーティリティ関数
```

### **__init__.py の現在の公開API**

```python
# chunking/__init__.py (document index 1 から)
from .models import (
    SentenceUnit,
    ParagraphUnit,
    StructuralResult,
    ContinuityResult
)
from .prompts import (
    PARAGRAPH_SEPARATION_PROMPT,
    SEMANTIC_CHUNKING_PROMPT,
    CONTINUITY_CHECK_PROMPT
)
from .async_api_client import AsyncAPIClient
from .checkpoint_manager import CheckpointManager
from .csv_to_chunks_text_para import (
    LargeTextProcessorPara,      # ← 存在確認が必要
    chunk_overlap_para,           # ← 存在確認が必要
    chunks_all,                   # ← 存在確認が必要
    chunks_all_async              # ← これは改修版に存在
)
from .utils import (
    show_paragraphs,
    setup_logging,
    format_time,
    format_size,
    estimate_api_calls
)
```

---

## ⚠️ 配置による影響

### **1. ファイル配置の直接的影響**

| 項目 | 影響度 | 内容 |
|------|--------|------|
| **ファイル置き換え** | 🟡 中 | `csv_to_chunks_text_para.py` → `csv_to_chunks_text_para_modified.py` |
| **インポートパス** | 🟢 低 | 内部で`from chunking.xxx`を使用（正しい） |
| **既存機能の互換性** | 🔴 高 | `__init__.py`で公開されている関数が存在するか確認が必要 |

---

### **2. 公開APIの変更影響**

#### **🔴 問題: __init__.py で公開されているが改修版に存在しない関数**

```python
# __init__.py で公開されているが、改修版に存在しない可能性がある
from .csv_to_chunks_text_para import (
    LargeTextProcessorPara,      # ❌ 改修版に存在しない
    chunk_overlap_para,           # ❌ 改修版に存在しない
    chunks_all,                   # ❌ 改修版に存在しない
    chunks_all_async              # ✅ 改修版に存在
)
```

**改修版に存在する関数:**
```python
# csv_to_chunks_text_para_modified.py
- load_text_from_csv()           # ✅ 新規追加
- save_chunks_as_csv()           # ✅ 新規追加
- save_chunks_as_text()          # ✅ 新規追加
- chunks_all_async()             # ✅ 既存（シグネチャ変更あり）
- _step1_hierarchical_split()    # 内部関数
- _step2_semantic_chunking()     # 内部関数
- _step3_continuity_check()      # 内部関数
- main()                         # エントリーポイント
```

---

### **3. 依存関係の影響**

#### **新規依存パッケージ**

```python
# csv_to_chunks_text_para_modified.py で新たに使用
import pandas as pd  # ← 新規依存
from pathlib import Path
```

**requirements.txt への追加確認:**
```bash
# chunking/requirements.txt
google-genai>=0.1.0
pydantic>=2.0.0
tqdm>=4.65.0
pandas>=2.0.0  # ← 追加が必要か確認
```

---

### **4. 呼び出し元への影響**

#### **外部からの呼び出し確認**

```python
# 想定される呼び出し元
1. make_qa.py / make_qa_register_qdrant.py
   → pipeline.py を経由して間接的に呼び出し
   → 直接的な影響は少ない

2. 直接呼び出し（コマンドライン）
   python -m chunking.csv_to_chunks_text_para -i input.txt -o output.txt
   → ✅ 改修版でも動作（mainエントリーポイント存在）

3. __init__.py 経由のインポート
   from chunking import chunks_all_async
   → ⚠️ シグネチャ変更に注意
```

---

## 📝 必要な変更一覧

### **必須変更（Priority: High）**

#### **1. ファイル配置**

```bash
# バックアップ
cp chunking/csv_to_chunks_text_para.py chunking/csv_to_chunks_text_para.py.backup

# 配置
cp csv_to_chunks_text_para_modified.py chunking/csv_to_chunks_text_para.py
```

---

#### **2. __init__.py の更新**

**変更前:**
```python
# chunking/__init__.py
from .csv_to_chunks_text_para import (
    LargeTextProcessorPara,
    chunk_overlap_para,
    chunks_all,
    chunks_all_async
)
```

**変更後:**
```python
# chunking/__init__.py
from .csv_to_chunks_text_para import (
    chunks_all_async,           # ✅ 既存（シグネチャ変更）
    load_text_from_csv,         # ✅ 新規追加
    save_chunks_as_csv,         # ✅ 新規追加
    save_chunks_as_text,        # ✅ 新規追加
)
```

**__all__ の更新:**
```python
__all__ = [
    # ... 既存 ...
    # csv_to_chunks_text_para
    "chunks_all_async",
    "load_text_from_csv",      # ✅ 新規
    "save_chunks_as_csv",      # ✅ 新規
    "save_chunks_as_text",     # ✅ 新規
]
```

---

#### **3. requirements.txt の更新**

```bash
# chunking/requirements.txt に追加
pandas>=2.0.0
```

**確認方法:**
```bash
# 既に含まれているか確認
grep pandas chunking/requirements.txt

# 含まれていない場合は追加
echo "pandas>=2.0.0" >> chunking/requirements.txt
```

---

### **推奨変更（Priority: Medium）**

#### **4. 後方互換性のための関数追加（オプション）**

既存コードが`LargeTextProcessorPara`等を使っている場合、互換性のために残す:

```python
# chunking/csv_to_chunks_text_para.py の末尾に追加

# ================================================================
# 後方互換性のための関数（非推奨）
# ================================================================

def chunks_all(*args, **kwargs):
    """
    非推奨: chunks_all_async() を使用してください

    後方互換性のために残されています。
    """
    import warnings
    warnings.warn(
        "chunks_all() is deprecated. Use chunks_all_async() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    import asyncio
    return asyncio.run(chunks_all_async(*args, **kwargs))


def chunk_overlap_para(*args, **kwargs):
    """
    非推奨: この機能は削除されました

    チャンクのオーバーラップは chunks_all_async() の
    overlap_tokens パラメータで制御してください。
    """
    raise NotImplementedError(
        "chunk_overlap_para() is removed. "
        "Use overlap_tokens parameter in chunks_all_async() instead."
    )


class LargeTextProcessorPara:
    """
    非推奨: このクラスは削除されました

    chunks_all_async() 関数を直接使用してください。
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "LargeTextProcessorPara is removed. "
            "Use chunks_all_async() function instead."
        )
```

---

### **オプション変更（Priority: Low）**

#### **5. ドキュメントの更新**

```python
# chunking/__init__.py のdocstring更新
"""
chunking パッケージ

テキストを意味的なチャンクに分割するためのツール。
非同期・並列処理により高速化。

主要機能:
- chunks_all_async(): テキストからチャンクを作成（asyncio版）
- load_text_from_csv(): CSVファイルからテキストを読み込み
- save_chunks_as_csv(): チャンクをCSV形式で保存

バージョン: 1.2.0 (CSV入力対応版)
"""
```

---

## 🔄 変更手順

### **Step 1: バックアップ**

```bash
# 現在のファイルをバックアップ
mkdir -p backups/$(date +%Y%m%d)
cp chunking/csv_to_chunks_text_para.py backups/$(date +%Y%m%d)/
cp chunking/__init__.py backups/$(date +%Y%m%d)/
cp chunking/requirements.txt backups/$(date +%Y%m%d)/
```

---

### **Step 2: ファイル配置**

```bash
# 改修版を配置
cp csv_to_chunks_text_para_modified.py chunking/csv_to_chunks_text_para.py
```

---

### **Step 3: __init__.py の更新**

```bash
# エディタで編集
nano chunking/__init__.py
```

**編集内容:**
```python
# ===== 変更箇所 =====
from .csv_to_chunks_text_para import (
    chunks_all_async,
    load_text_from_csv,         # 追加
    save_chunks_as_csv,         # 追加
    save_chunks_as_text,        # 追加
)

__all__ = [
    # ... 既存 ...
    "chunks_all_async",
    "load_text_from_csv",       # 追加
    "save_chunks_as_csv",       # 追加
    "save_chunks_as_text",      # 追加
]
```

---

### **Step 4: requirements.txt の確認・更新**

```bash
# pandasが含まれているか確認
grep -i pandas chunking/requirements.txt

# 含まれていない場合は追加
if ! grep -q "pandas" chunking/requirements.txt; then
    echo "pandas>=2.0.0" >> chunking/requirements.txt
    echo "pandas を requirements.txt に追加しました"
fi
```

---

### **Step 5: 依存パッケージのインストール**

```bash
# 仮想環境で実行（推奨）
pip install -r chunking/requirements.txt
```

---

### **Step 6: 動作確認（簡易テスト）**

```bash
# テスト実行
echo "テストテキスト" > test_input.txt

python -m chunking.csv_to_chunks_text_para \
  -i test_input.txt \
  -o test_output.csv \
  -w 4

# 出力確認
cat test_output.csv
```

---

## ✅ テスト項目

### **1. 基本動作テスト**

```bash
# テスト1: TXT入力 → CSV出力
python -m chunking.csv_to_chunks_text_para \
  -i test.txt \
  -o output.csv \
  -w 4

# テスト2: CSV入力 → CSV出力
python -m chunking.csv_to_chunks_text_para \
  -i test.csv \
  -o output.csv \
  --text-column "text" \
  -w 4

# テスト3: チェックポイント機能
# (途中でCtrl+Cで中断)
python -m chunking.csv_to_chunks_text_para \
  -i large_test.txt \
  -o output.csv \
  -w 4

# 再開
python -m chunking.csv_to_chunks_text_para \
  --resume <JOB_ID> \
  -i large_test.txt \
  -o output.csv \
  -w 4
```

---

### **2. インポートテスト**

```python
# test_import.py
import sys
sys.path.insert(0, '.')

# テスト1: パッケージインポート
from chunking import chunks_all_async
print("✅ chunks_all_async インポート成功")

# テスト2: 新規関数インポート
from chunking import load_text_from_csv, save_chunks_as_csv
print("✅ 新規関数インポート成功")

# テスト3: モデルインポート
from chunking import StructuralResult, ContinuityResult
print("✅ モデルインポート成功")

print("\n全てのインポートテスト成功！")
```

```bash
python test_import.py
```

---

### **3. 統合テスト（make_qa.py経由）**

```bash
# テスト: チャンクCSV → Q/A生成
python -m chunking.csv_to_chunks_text_para \
  -i test.txt \
  -o chunks.csv \
  -w 4

python make_qa_modified.py \
  --input-chunks chunks.csv \
  --max-docs 10
```

---

### **4. 後方互換性テスト（必要な場合）**

```python
# test_compatibility.py
from chunking import chunks_all

# 非推奨警告が表示されるか確認
try:
    result = chunks_all("テストテキスト")
    print("✅ 後方互換性あり（非推奨警告が表示されるはず）")
except NotImplementedError as e:
    print(f"❌ 後方互換性なし: {e}")
```

---

## 🔙 ロールバック手順

万が一問題が発生した場合:

```bash
# Step 1: バックアップから復元
cp backups/$(date +%Y%m%d)/csv_to_chunks_text_para.py chunking/
cp backups/$(date +%Y%m%d)/__init__.py chunking/
cp backups/$(date +%Y%m%d)/requirements.txt chunking/

# Step 2: 依存パッケージを再インストール
pip install -r chunking/requirements.txt

# Step 3: 動作確認
python -m chunking.csv_to_chunks_text_para -i test.txt -o test.csv
```

---

## 📊 影響度マトリクス

| コンポーネント | 影響度 | 対応 | 優先度 |
|--------------|--------|------|--------|
| **csv_to_chunks_text_para.py** | 🔴 高 | ファイル置き換え | High |
| **__init__.py** | 🔴 高 | インポート文の更新 | High |
| **requirements.txt** | 🟡 中 | pandas追加確認 | High |
| **make_qa.py** | 🟢 低 | 変更不要（pipeline経由） | Low |
| **pipeline.py** | 🟢 低 | 変更不要（チャンクCSV読み込み対応済み） | Low |
| **既存ユーザーコード** | 🟡 中 | 非推奨警告または移行が必要 | Medium |

---

## 🎯 チェックリスト

### **配置前の確認**

- [ ] 現在のファイルをバックアップした
- [ ] 改修版の動作をローカルで確認した
- [ ] 依存パッケージ（pandas）がインストール済み
- [ ] Git等でバージョン管理している（推奨）

### **配置作業**

- [ ] csv_to_chunks_text_para.py を置き換えた
- [ ] __init__.py を更新した
- [ ] requirements.txt を更新した（pandasを追加）
- [ ] 依存パッケージを再インストールした

### **配置後の確認**

- [ ] インポートテストが成功した
- [ ] 基本動作テストが成功した
- [ ] make_qa.py との統合テストが成功した
- [ ] ドキュメントを更新した（README等）

---

## 📋 変更サマリー

| 項目 | 変更内容 |
|------|---------|
| **ファイル配置** | `csv_to_chunks_text_para_modified.py` → `csv_to_chunks_text_para.py` |
| **__init__.py** | インポート文とエクスポートリストを更新 |
| **requirements.txt** | `pandas>=2.0.0` を追加（未含の場合） |
| **新機能** | CSV入力対応、CSV出力、テキストカラム自動検出 |
| **削除された機能** | `LargeTextProcessorPara`, `chunk_overlap_para`, `chunks_all` |
| **後方互換性** | オプションで非推奨警告付き関数を追加可能 |

---

## 🎓 推奨アプローチ

### **段階的な移行（推奨）**

```
Phase 1: テスト環境で検証
  → 別ブランチで変更
  → テスト実行
  → 問題なければ次へ

Phase 2: 本番環境への適用
  → バックアップ取得
  → 配置
  → 動作確認

Phase 3: ドキュメント更新
  → README更新
  → チームへの通知
```

---

**作成日**: 2025-01-11
**最終更新**: 2025-01-11
