# `ui/pages/agent_chat_page.py` 処理フロー

このドキュメントは、Streamlitアプリケーションの「エージェントチャット (Agent Chat)」ページの制御フローを概説します。

## 概要

このページでは、Gemini ReActエージェントを初期化し、チャットセッションの状態を管理し、ユーザー入力、エージェントの推論（思考/行動）、ツール実行（RAG検索）、およびReflection（推敲）ステップを含む最終回答生成の対話ループを処理します。

## フローチャート (Mermaid)

```mermaid
flowchart TD
    Start(["開始: show_agent_chat_page"]) --> UI_Render["UIコンポーネント描画"]
    
    subgraph UI_Components ["UIレイアウト"]
        UI_Render --> UI_Title["タイトル & キャプション"]
        UI_Render --> UI_Exp_Docs["Expander: 元ドキュメント表示"]
        UI_Render --> UI_Exp_QA["Expander: 登録Q&Aプレビュー"]
        UI_Render --> UI_Sidebar["サイドバー設定"]
    end

    subgraph Sidebar_Logic ["サイドバー設定ロジック"]
        UI_Sidebar --> SB_Model["モデル選択"]
        UI_Sidebar --> SB_Coll["コレクション選択"]
        UI_Sidebar --> SB_Clear["ボタン: 履歴クリア"]
        
        SB_Clear -- "クリック" --> Reset_State["セッション状態リセット"]
        Reset_State --> Rerun["st.rerun (再実行)"]
    end

    UI_Components --> Init_Check{"セッション状態確認"}
    
    Init_Check -- "未初期化 OR 設定変更" --> Setup_Agent["呼出: setup_agent"]
    Setup_Agent --> Config_Gemini["genai設定 (API Key)"]
    Config_Gemini --> Build_SysPrompt["システムプロンプト構築\n(コレクション情報埋め込み)"]
    Build_SysPrompt --> Start_Chat["model.start_chat\n(ツール: search_rag...)"]
    Start_Chat --> Update_State["セッション状態更新\n(chat_session, current_model等)"]
    
    Init_Check -- "初期化済 & 変更なし" --> Display_History["チャット履歴表示"]
    Update_State --> Display_History
    
    Display_History --> User_Input{"ユーザー入力あり?"}
    
    User_Input -- "No" --> End(["待機"])
    User_Input -- "Yes" --> Process_Input["履歴にユーザーメッセージ追加"]
    Process_Input --> Call_Run_Agent["呼出: run_agent_turn"]

    subgraph ReAct_Process ["run_agent_turn ロジック"]
        Call_Run_Agent --> Send_Msg["chat_session.send_message"]
        Send_Msg --> Loop_Start{"ループ: 最大10ターン"}
        
        Loop_Start -- "制限到達" --> Reflection_Phase
        Loop_Start -- "次ターン" --> Check_Response{"モデル応答確認"}
        
        Check_Response -- "関数呼び出しあり" --> Extract_Call["ツール名 & 引数抽出"]
        Extract_Call --> Exec_Tool["ツール実行\n(search_rag_knowledge_base 等)"]
        Exec_Tool --> Tool_Log["ツール結果ログ記録"]
        Tool_Log -- "結果が [[NO_RAG_RESULT]]" --> Log_Fail["未回答質問ログ記録 (log_unanswered_question)"]
        Tool_Log --> Send_Tool_Out["chat_session.send_message\n(function_response)"]
        Log_Fail --> Send_Tool_Out
        Send_Tool_Out --> Loop_Start
        
        Check_Response -- "テキストのみ (Thought)" --> Log_Thought["思考(Thought)をログ記録"]
        Log_Thought --> Loop_Start
        
        Check_Response -- "テキストのみ (Answer)" --> Set_Draft["回答案(Draft)として保持"]
        Set_Draft --> Break_Loop["ループ脱出"]
        Break_Loop --> Reflection_Phase
        
        subgraph Reflection_Logic ["Reflection (推敲) フェーズ"]
            Reflection_Phase --> Send_Reflect["Reflectionプロンプト送信"]
            Send_Reflect --> Parse_Reflect["'Final Answer:' を解析"]
            Parse_Reflect --> Update_Final["最終回答を更新"]
        end
        
        Update_Final --> Render_Thoughts["Expanderに思考プロセス表示"]
        Render_Thoughts --> Return_Resp["最終回答を返却"]
    end

    Call_Run_Agent --> Display_Resp["エージェント回答表示"]
    Display_Resp --> Append_Hist["履歴にアシスタントメッセージ追加"]
    Append_Hist --> End
```

## 関数の詳細ロジック

### 1. `setup_agent(selected_collections, model_name)`
*   **目的:** Gemini `ChatSession` を初期化します。
*   **入力:** コレクション名のリスト、モデル名。
*   **処理:** 
    *   APIキーを取得します。
    *   利用可能なコレクション名を埋め込んだ `SYSTEM_INSTRUCTION_TEMPLATE` を構築します。
    *   ツール（`search_rag_knowledge_base`, `list_rag_collections`）を指定して `GenerativeModel` を初期化します。
    *   チャットセッションを開始します（`start_chat`）。

### 2. `run_agent_turn(chat_session, user_input)`
*   **目的:** マルチターンの ReAct ループと Reflection（推敲）を管理します。
*   **ReAct ループ:**
    *   ユーザー入力をモデルに送信します。
    *   ツール呼び出しを処理するために最大10回反復します。
    *   **ツール呼び出しの場合:** ツール（Qdrant検索など）を実行し、結果（失敗時は `log_unanswered_question` によるログ記録を含む）を記録し、結果をモデルに返します。
    *   **テキストの場合:** ログ記録用に「Thought（思考）」をキャプチャします。ツール呼び出しがない場合、そのテキストを **回答案 (Draft Answer)** として扱います。
*   **Reflection (推敲) フェーズ:**
    *   回答案を取得します。
    *   正確性とスタイルを評価するために `REFLECTION_INSTRUCTION` をモデルに送信します。
    *   出力から `Final Answer:` を解析します。
*   **出力:** 推敲された回答文字列を返します。

### 3. `show_agent_chat_page()`
*   **目的:** メインの Streamlit 描画関数です。
*   **コンポーネント:**
    *   **サイドバー:** QdrantコレクションとLLMモデルの選択、セッションリセットを処理します。
    *   **ドキュメントビューア:** `OUTPUT/` ディレクトリからファイルを読み込み、元のテキストデータを表示します。
    *   **Q&A プレビュー:** `Qdrant` コレクションをスクロールして、RAGで使用されるQ&Aペアのサンプルを表示します。
    *   **チャットインターフェース:** 標準的な Streamlit チャットインターフェース（`st.chat_message`, `st.chat_input`）。

```