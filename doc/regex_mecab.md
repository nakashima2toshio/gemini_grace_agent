# regex_mecab.py 詳細設計書

## 概要
`regex_mecab.py` は、日本語テキストから重要語句（キーワード）を抽出するためのモジュールです。
形態素解析エンジン「MeCab」を使用した高精度な複合名詞抽出を主軸としつつ、環境依存を減らすために正規表現（Regex）によるロバストなフォールバック機構を備えています。

---

## 1. MeCabのChapter

本章では、形態素解析エンジン MeCab を利用した処理ロジックについて記述します。MeCabが利用可能な環境では、品詞情報に基づく正確な複合名詞の抽出が行われます。

### クラス、関数一覧

| 名称 | 種類 | 概要 |
| :--- | :--- | :--- |
| **KeywordExtractor** | Class | キーワード抽出のメインクラス。MeCabの利用可否を管理し、抽出メソッドを提供する。 |
| **_check_mecab_availability** | Method | MeCabライブラリがインポート可能かつ動作するかを確認する。 |
| **_extract_with_mecab** | Method | MeCabを使用してテキストから名詞（特に複合名詞）を抽出し、フィルタリング・ランク付けを行う。 |
| **_extract_with_mecab_scored** | Method | MeCab抽出結果に対し、詳細なスコア計算を行い、キーワードとスコアのタプルリストを返す。 |

### 関数のIPO (Input, Output, Process)

#### `_check_mecab_availability(self)`
*   **INPUT**: なし
*   **OUTPUT**: `bool` (利用可能な場合 True, 不可の場合 False)
*   **PROCESS**:
    1. `import MeCab` を試行する。
    2. `MeCab.Tagger()` をインスタンス化し、ダミーテキスト（"テスト"）の解析を試行する。
    3. 成功すれば `True`、`ImportError` や `RuntimeError` が発生すれば `False` を返す。

#### `_extract_with_mecab(self, text: str, top_n: int, use_scoring: bool)`
*   **INPUT**:
    *   `text` (str): 分析対象のテキスト
    *   `top_n` (int): 抽出するキーワードの上限数
    *   `use_scoring` (bool): 高度なスコアリングを使用するか否か
*   **OUTPUT**: `List[str]` (抽出されたキーワードのリスト)
*   **PROCESS**:
    1. MeCab Tagger を初期化し、入力テキストを `parseToNode` で解析する。
    2. ノードを走査し、品詞が「名詞」であるトークンをバッファに蓄積する（複合名詞の連結処理）。
    3. 名詞以外の品詞が出現した時点でバッファ内の文字列を結合し、候補リストに追加する。
    4. 走査完了後、`use_scoring` が True ならば `_score_and_rank` を、False ならば `_filter_and_count` を呼び出す。
    5. 上位 `top_n` 件の結果を返す。

#### `_extract_with_mecab_scored(self, text: str, top_n: int)`
*   **INPUT**:
    *   `text` (str): 分析対象のテキスト
    *   `top_n` (int): 抽出するキーワードの上限数
*   **OUTPUT**: `List[Tuple[str, float]]` (キーワードとスコアのペアリスト)
*   **PROCESS**:
    1. `_extract_with_mecab(..., use_scoring=True)` を呼び出し、キーワードリストを取得する。
    2. 取得した各キーワードに対して `_calculate_keyword_score` を呼び出し、詳細なスコアを再計算する。
    3. キーワードとスコアのタプルリストを作成して返す。

---

## 2. RegexのChapter

本章では、正規表現（Regular Expression）を利用した処理ロジックについて記述します。MeCabがインストールされていない環境や、処理の高速化、あるいはMeCabの結果を補完するために使用されます。

### クラス、関数一覧

| 名称 | 種類 | 概要 |
| :--- | :--- | :--- |
| **_extract_with_regex** | Method | 正規表現パターンを使用してテキストから特定の文字種（カタカナ、漢字、英数字）の連続を抽出する。 |
| **_extract_with_regex_scored** | Method | 正規表現抽出結果に対し、詳細なスコア計算を行い、キーワードとスコアのタプルリストを返す。 |
| **_filter_and_count** | Method | (共通処理) ストップワード除去と単純な出現頻度によるフィルタリングを行う。 |
| **_score_and_rank** | Method | (共通処理) 頻度、長さ、文字種、重要語句辞書に基づいた高度なスコアリングを行う。 |
| **_calculate_keyword_score** | Method | (共通処理) 単一のキーワードに対するスコア計算を行う。 |

### 関数のIPO (Input, Output, Process)

#### `_extract_with_regex(self, text: str, top_n: int, use_scoring: bool)`
*   **INPUT**:
    *   `text` (str): 分析対象のテキスト
    *   `top_n` (int): 抽出するキーワードの上限数
    *   `use_scoring` (bool): 高度なスコアリングを使用するか否か
*   **OUTPUT**: `List[str]` (抽出されたキーワードのリスト)
*   **PROCESS**:
    1. 以下の正規表現パターンを定義する。
        *   `[ァ-ヴー]{2,}`: カタカナ2文字以上
        *   `[一-龥]{2,}`: 漢字2文字以上
        *   `[A-Za-z]{2,}[A-Za-z0-9]*`: 英字2文字以上で始まる英数字
    2. `re.findall` を使用して、テキストからパターンにマッチする文字列を全て抽出する。
    3. `use_scoring` が True ならば `_score_and_rank` を、False ならば `_filter_and_count` を呼び出す。
    4. 上位 `top_n` 件の結果を返す。

#### `_score_and_rank(self, words: List[str], top_n: int)`
*   **INPUT**:
    *   `words` (List[str]): 抽出された単語の候補リスト
    *   `top_n` (int): 返却する件数
*   **OUTPUT**: `List[str]` (ランク付けされたキーワードリスト)
*   **PROCESS**:
    1. 単語の出現頻度（`Counter`）を計算する。
    2. ストップワードおよび1文字以下の単語を除外する。
    3. 各単語に対して以下の要素でスコアを加算する：
        *   **頻度スコア**: 出現回数に基づく（上限あり）。
        *   **長さスコア**: 文字列長に基づく（長い単語、複合語を優遇）。
        *   **重要キーワードブースト**: 定義済みの重要語句（AI, BERT等）に含まれる場合加点。
        *   **文字種スコア**: カタカナ、英大文字、漢字など特定のパターンに対し加点。
    4. スコアの降順でソートし、上位 `top_n` 件を抽出して返す。

#### `_calculate_keyword_score(self, keyword: str, text: str)`
*   **INPUT**:
    *   `keyword` (str): 評価対象のキーワード
    *   `text` (str): 元のテキスト
*   **OUTPUT**: `float` (計算されたスコア 0.0 ~ 1.0)
*   **PROCESS**:
    1. `_score_and_rank` と同様のロジックを用いて、単一のキーワードに対するスコアを計算する。
    2. 頻度、長さ、重要度、文字種を総合的に評価し、最大 1.0 に正規化して返す。
