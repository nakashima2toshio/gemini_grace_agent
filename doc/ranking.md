# RAG Re-ranking Implementation Design (Cohere API)

## 1. 概要
RAG検索の精度向上と信頼度スコアの適正化（0.9以上での確信）を実現するため、Hybrid Search (RRF) の結果に対して **Cohere Rerank API (`rerank-multilingual-v3.0`)** を用いた再ランク付け（Re-ranking）を導入する。

現在のRRFスコア（0.666等）は相対ランク指標であり、絶対的な信頼度として扱いにくいため、Re-rankerによる確率スコア（0.0〜1.0）への変換を行う。

## 2. 変更対象ファイル
*   `agent_tools.py`

## 3. IPO詳細設計

### 3.1 新規追加関数: `rerank_results`

Cohere API を使用して検索結果のスコアを更新し、再ソートする。

*   **Input (入力)**:
    *   `query` (str): ユーザーの検索クエリ。
    *   `results` (List[Dict]): Qdrantから取得した検索結果のリスト（Payloadを含む）。
    *   `top_k` (int): 最終的に残す件数（デフォルト: 3）。
    *   `threshold` (float): スコアの足切りライン（デフォルト: 0.5）。

*   **Process (処理概要)**:
    1.  `results` が空であれば、そのまま空リストを返す。
    2.  Cohere API クライアントを初期化（APIキーは `os.getenv` または `Config` から取得）。
    3.  `results` からドキュメントのテキストリスト（Question + Answer）を作成。
    4.  `co.rerank` API をコール。
        *   `model`: `rerank-multilingual-v3.0`
        *   `query`: 入力クエリ
        *   `documents`: テキストリスト
    5.  APIレスポンスの `relevance_score` を、元の `results` の `score` に上書きする。
    6.  `threshold` 未満のスコアを持つ結果を除外する。
    7.  スコアの降順でソートし、上位 `top_k` 件を抽出する。

*   **Output (出力)**:
    *   `List[Dict]`: スコアが更新され、ソート・フィルタリング済みの検索結果リスト。

### 3.2 変更関数: `search_rag_knowledge_base_structured`

検索フロー全体を制御し、Re-rankingを組み込む。

*   **Input (入力)**:
    *   `query` (str)
    *   `collection_name` (Optional[str])

*   **Process (処理概要)**:
    1.  **Retrieval (候補取得)**:
        *   `search_collection` を呼び出す。
        *   **変更点**: `limit` を `AgentConfig.RAG_SEARCH_LIMIT`（3件）から **10〜20件** に増やす。
        *   理由: RRFで順位が低くても、Re-rankで浮上する正解候補を拾うため（Recall重視）。
    2.  **Re-ranking (再ランク付け)**:
        *   取得した候補 (`results`) を `rerank_results(query, results)` に渡す。
        *   スコアが「順位ベース」から「確率ベース」に変換される。
    3.  **Metrics Logging (メトリクス記録)**:
        *   更新されたスコアに基づいて、最高スコアなどを記録する。
    4.  **Error Handling (エラー処理)**:
        *   Cohere API キーがない場合やエラー時は、フォールバックとして既存のスコア（RRF）を使用する（あるいはDenseスコアを採用するロジックへの分岐）。

*   **Output (出力)**:
    *   `List[Dict]` or `str`: 最終的な検索結果。

## 4. 不要になる（または変更される）クラス・関数

`agent_tools.py` 内で以下の処理は、Re-ranking導入により役割が変わるか、不要になります。

| クラス/関数 | 変更/削除 | 理由 |
| :--- | :--- | :--- |
| `filter_results_by_keywords` | **維持（役割縮小）** | Re-ranking前後のどちらかで「最低限のキーワードマッチ」を確認するために残すが、Re-rankerの精度が高ければ、厳密なキーワードフィルタリングの重要度は下がる。前処理（候補取得後）に行うのが効率的。 |
| (インライン処理) `AgentConfig.RAG_SCORE_THRESHOLD` によるフィルタ | **変更** | RRFスコアに対する閾値判定ではなく、Re-rank後の確率スコアに対する判定（0.5以上など）に変更する。 |

## 5. 実装手順プラン

1.  `config.py` に `COHERE_API_KEY` の設定を追加（または環境変数参照）。
2.  `agent_tools.py` に `import cohere` を追加（エラーハンドリング付き）。
3.  `rerank_results` 関数を実装。
4.  `search_rag_knowledge_base_structured` 内のフローを「Retrieval (Broad) -> Re-ranking -> Filtering」に変更。

## 6. 期待される効果

*   **スコアの直感性**: 完全一致に近い回答は `0.9` 以上のスコアとなり、ユーザーの「0.666はおかしい」という不満を解消。
*   **精度の向上**: 文脈を考慮したRe-rankにより、キーワード検索だけでは拾えない、またはRRFで埋もれていた正解を引き上げる。
*   **信頼度計算との整合**: 0.0〜1.0の確率スコアになるため、GRACEのConfidence計算（0.8以上で高信頼度）とそのまま整合する。
