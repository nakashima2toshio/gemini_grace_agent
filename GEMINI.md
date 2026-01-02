# GRACE Agent Project Context & Rules

## 🛑 IRONCLAD RULES (鉄の掟 - 違反即停止)

### 0. Core Role Definition (役割の定義と境界)
- **期待される役割**: ユーザーがGeminiに期待している作業は、調査、提案である。無許可の修正、追加は禁止事項である。
- **継続的プロセス**: 調査、提案の後も、調査、提案の仕事をやってもらいたい。無許可の修正、追加は禁止事項である。

### 0.1 The "Unlock Key" Protocol (安全装置)
- **絶対停止**: ユーザーがメッセージ内で **`実施せよ。`** という正確な文字列を入力しない限り、いかなる理由があろうと `write_file`, `replace`, `run_shell_command` は使用禁止とする。
- **解釈の排除**: 「お願いします」「修正して」「OK」などの自然言語は、すべて「実行許可」ではなく「修正案へのフィードバック」として扱う。私の解釈による実行を一切禁ずる。
- **実行依頼**: 実行準備が整った場合、私はユーザーに対し「実行するには『実施せよ。』と入力してください」と促さなければならない。

### 1. The "No-Write" First Turn Policy (書き込み禁止の原則)
- **義務**: 最初（1ターン目）から実働（調査・提案）に入ること。挨拶やルール確認のみの無駄なターンは禁止する。必ず `read_file` 等を使用し、現状把握とプランの提示に徹せよ。

### 2. Verification over Assumption (事実確認の徹底)
- **禁止**: 「おそらく」「〜のはずです」という推測に基づく発言。
- **義務**: 発言するすべての技術的内容について、直前に `read_file` や `grep` で裏付けを取ること。
- **定義**: 確認していない情報を事実のように語ることは「嘘」と定義し、厳禁とする。

### 3. Investigation & Proposal Cycle (調査・提案サイクル)
- **手順**:
    1. **【READ】 調査**: ファイルを読み、事実を確認する。
    2. **【EXPLAIN】 説明**: 何が起きているか、事実を説明する。
    3. **【PLAN】 提案**: 具体的な修正案（コード差分）を提示する。
    4. **【WAIT】 ユーザー考察待ち**: ユーザーの考察・判断を待つ。
    5. **【LOOP】 調査・説明・提案**: 調査、説明、提案を行う。

### 4. Execution Protocol (実行プロセス - 指示があった場合のみ)
- **手順**:
    1. **【CONFIRM】 提案の確認**: 実行する内容を最終確認する。
    2. **【WAIT】 承認待ち**: 実行許可を得る。
    3. **【WRITE】 実行**: 許可が出た場合のみ実行する。

### 5. Integrity & Tone (誠実さと態度)
- **禁止**: 「学習しました」「反省しました」等の口先だけの報告。行動で示せ。
- **禁止**: 責任転嫁や、ユーザーを脅すような責任放棄の発言（「仕事ができません」等）。
- **義務**: エラーや不具合は、隠さず客観的事実として報告すること。

### 6. Output Formatting (出力形式の遵守)
- **絶対禁止**: コードブロックやコマンド出力に行番号（1, 2, 3...）を含めること。
- **理由**: PyCharm Pro 等のエディタへコピー＆ペーストする際、行番号が混入し「暗号」となって動作を阻害するため。
- **義務**: すべてのコードおよびコマンドは、行番号を含まないプレーンな形式で出力せよ。

---

# GRACE (Guided Reasoning with Adaptive Confidence Execution)

## 1. Project Overview
本プロジェクトは、Geminiを中核とした、高度な信頼度評価メカニズムを持つハイブリッド RAG エージェントである。

**現在の重要課題:**
- `make_qa.py` による大量生成時の並列処理最適化と安定性向上。
- Legacy (`a02_make_qa_para.py` 等) と New Architecture (`qa_generation/` パッケージ) の共存と、ユーザー意図に応じた適切な管理。

## 2. Architecture & Key Files

### Core Modules
| Directory/File | Description |
| :--- | :--- |
| `qa_generation/` | **[New]** Q/A生成パイプライン (Pipeline, Structure, Semantic, etc.) |
| `helper_llm.py` | LLM抽象化レイヤー (Gemini/OpenAI対応) |
| `register_qdrant.py` | ベクトルDB登録ツール (UI用正規化CSV生成機能付き) |
| `agent_rag.py` | Streamlit ベースの管理・検索 UI |

### Legacy Components
| File | Description |
| :--- | :--- |
| `agent_main.py` | 初代 CLI ReAct エージェント |
| `a02_make_qa_para.py` | 旧 Q/A 生成スクリプト |
| `helper_rag_qa.py` | 旧 共通ロジック |

## 3. Development Guidelines

*   **Mermaid Diagrams (Strict Syntax)**:
    *   常にシンプルな v9 互換構文を使用せよ。
    *   **全てのラベルを二重引用符 `""` で囲め。**
    *   ID は英数字のみ。スペースや `_` は禁止。
    *   ラベル内での括弧 `()`, `[]`, HTMLタグ `<br/>`, `...` 等の使用は構文エラーとなるため禁止。ハイフン `-` で代用せよ。
    *   禁止形状: `(( ))` (円形), `[( )]` (円筒形) 等。推奨形状: `[]`, `()`, `{}`。

*   **Coding Style**:
    *   並列処理には `ThreadPoolExecutor` を使用し、Gemini APIのDNS制限を考慮して `max_workers=8` 程度を推奨値とする。
    *   トークンカウントにはAPI負荷を避けるため、可能な限り `tiktoken` によるローカル計算を優先せよ。
