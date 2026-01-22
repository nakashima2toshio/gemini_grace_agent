# Python 非同期処理（async/await）解説

## 対象読者
ソフトウェア開発の初級〜中級者で、Pythonの基本文法は理解している方

---

## 1. なぜ非同期処理が必要なのか

### 同期処理の問題点

通常のPythonプログラムは「同期処理」で動きます。

```python
# 同期処理の例
def make_coffee():
    print("コーヒーを淹れる（3分待つ）")
    time.sleep(180)  # 3分間、何もできない
    return "コーヒー完成"

def make_toast():
    print("トーストを焼く（2分待つ）")
    time.sleep(120)  # 2分間、何もできない
    return "トースト完成"

# 順番に実行 → 合計5分かかる
coffee = make_coffee()  # 3分待つ
toast = make_toast()    # さらに2分待つ
```

**問題**: コーヒーを淹れている間、トーストを焼き始められない。
実際の生活では、コーヒーメーカーが動いている間にトースターも動かせるのに。

### 非同期処理の解決策

```python
# 非同期処理の例
async def make_coffee():
    print("コーヒーを淹れる（待ち時間中は他の作業可能）")
    await asyncio.sleep(180)  # 待っている間、他の処理ができる
    return "コーヒー完成"

async def make_toast():
    print("トーストを焼く（待ち時間中は他の作業可能）")
    await asyncio.sleep(120)
    return "トースト完成"

# 並行実行 → 合計3分で済む（長い方の時間）
coffee, toast = await asyncio.gather(make_coffee(), make_toast())
```

---

## 2. 基本用語の理解

### async def - 非同期関数の定義

```python
# 通常の関数
def normal_function():
    return "結果"

# 非同期関数（コルーチン）
async def async_function():
    return "結果"
```

`async def`で定義した関数は「コルーチン」と呼ばれます。
普通に呼び出しても実行されず、`await`が必要です。

```python
# 間違い：これだけでは実行されない
result = async_function()  # → コルーチンオブジェクトが返るだけ

# 正解：awaitで実行を待つ
result = await async_function()  # → "結果"が返る
```

### await - 完了を待つ

`await`は「この処理が終わるまで待つ。待っている間は他の処理をしてもいいよ」という意味です。

```python
async def fetch_data():
    # APIからデータを取得（時間がかかる）
    response = await api_call()  # ← 待っている間、他のタスクが動ける
    return response
```

### asyncio.run() - 非同期処理の開始点

非同期処理を始めるには、イベントループが必要です。
`asyncio.run()`がそれを作成して実行します。

```python
async def main():
    result = await some_async_function()
    print(result)

# プログラムのエントリーポイント
if __name__ == "__main__":
    asyncio.run(main())  # ここでイベントループが起動
```

---

## 3. check_async.py のコード解説

### 全体構造

```
main()
  └── process_text()
        ├── step1_hierarchical_split()  # テキスト → 段落
        ├── step2_semantic_chunking()   # 段落 → チャンク
        └── step3_continuity_check()    # チャンク → 最終チャンク
```

### Step1: 基本的な非同期関数

```python
async def step1_hierarchical_split(text: str, client: genai.Client) -> list[str]:
    """
    async def で非同期関数として定義
    """
    # プロンプト作成（通常の同期処理）
    prompt = f"{PARAGRAPH_SEPARATION_PROMPT}\n\n【入力テキスト】\n{text}"

    # API呼び出し（ここでは同期的に実行される）
    response = client.models.generate_content(...)

    # 結果を返す
    return paragraphs
```

**ポイント**: この関数自体は同期的なAPI呼び出しをしていますが、
`async def`で定義することで、呼び出し元が`await`で待てるようになります。

### Step2: ループ内での非同期

```python
async def step2_semantic_chunking(paragraphs: list[str], client: genai.Client) -> list[str]:
    chunks = []

    for i, para in enumerate(paragraphs):
        # 各段落を処理
        response = client.models.generate_content(...)

        # パースしてチャンクを追加
        for chunk_para in result.paragraphs:
            chunks.append(chunk_para.full_text)

        # ★重要★ イベントループに制御を戻す
        await asyncio.sleep(0)

    return chunks
```

**`await asyncio.sleep(0)` の意味**:
- 0秒待つ = 実質的に待たない
- しかし、イベントループに「他にやることある？」と確認する
- 長いループ処理中に、他のタスクに実行機会を与える

### Step3: 条件分岐と非同期

```python
async def step3_continuity_check(chunks: list[str], client: genai.Client) -> list[str]:
    # 早期リターン（チャンクが少ない場合）
    if len(chunks) <= 1:
        return chunks

    # 連続性判定のループ
    for i in range(len(chunks) - 1):
        response = client.models.generate_content(...)
        continuity_flags.append(result.is_connected)

        await asyncio.sleep(0)  # 他のタスクに制御を渡す

    # マージ処理（通常の同期処理）
    final_chunks = [chunks[0]]
    for i, is_connected in enumerate(continuity_flags):
        if is_connected:
            final_chunks[-1] += "\n\n" + chunks[i + 1]
        else:
            final_chunks.append(chunks[i + 1])

    return final_chunks
```

### process_text: 順次実行の例

```python
async def process_text(text: str, api_key: str) -> list[str]:
    client = genai.Client(api_key=api_key)

    # await で順番に実行（直列処理）
    paragraphs = await step1_hierarchical_split(text, client)  # まずStep1
    chunks = await step2_semantic_chunking(paragraphs, client) # 次にStep2
    final_chunks = await step3_continuity_check(chunks, client) # 最後にStep3

    return final_chunks
```

**なぜ順番に実行？**:
- Step2はStep1の結果が必要
- Step3はStep2の結果が必要
- 依存関係があるため、並列実行できない

### main: エントリーポイント

```python
async def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("エラー: GOOGLE_API_KEY 環境変数を設定してください")
        return

    test_text = """..."""

    # process_textの完了を待つ
    final_chunks = await process_text(test_text, api_key)

    # 結果表示
    for i, chunk in enumerate(final_chunks, 1):
        print(f"--- 最終チャンク{i} ---")
        print(chunk)

if __name__ == "__main__":
    asyncio.run(main())  # イベントループを起動してmain()を実行
```

---

## 4. 同期版との比較

### 同期版（check_step1.py）

```python
def step1_hierarchical_split(text: str, api_key: str) -> list[str]:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(...)
    return paragraphs

def main():
    paragraphs = step1_hierarchical_split(text, api_key)

if __name__ == "__main__":
    main()
```

### 非同期版（check_async.py）

```python
async def step1_hierarchical_split(text: str, client: genai.Client) -> list[str]:
    response = client.models.generate_content(...)
    return paragraphs

async def main():
    paragraphs = await step1_hierarchical_split(text, client)

if __name__ == "__main__":
    asyncio.run(main())
```

**主な違い**:

| 項目 | 同期版 | 非同期版 |
|------|--------|----------|
| 関数定義 | `def` | `async def` |
| 関数呼び出し | `func()` | `await func()` |
| 実行開始 | `main()` | `asyncio.run(main())` |
| 待ち時間中 | 何もできない | 他のタスクを実行可能 |

---

## 5. このコードでの非同期の効果

### 現状のコード

このcheck_async.pyでは、Step1→Step2→Step3と順番に実行しているため、
実は同期版とほぼ同じ動作になっています。

```python
paragraphs = await step1(...)   # Step1完了まで待つ
chunks = await step2(...)       # Step2完了まで待つ
final = await step3(...)        # Step3完了まで待つ
```

### 非同期の真価が発揮される場面

**例1: 複数テキストを並列処理**

```python
async def process_multiple_texts(texts: list[str], api_key: str):
    client = genai.Client(api_key=api_key)

    # 複数のテキストを並列で処理
    tasks = [process_text(text, client) for text in texts]
    results = await asyncio.gather(*tasks)

    return results
```

**例2: Step2で各段落を並列処理**

```python
async def step2_parallel(paragraphs: list[str], client: genai.Client):
    # 各段落の処理を並列で実行
    tasks = [process_paragraph(para, client) for para in paragraphs]
    results = await asyncio.gather(*tasks)

    # 結果を結合
    chunks = []
    for result in results:
        chunks.extend(result)
    return chunks
```

---

## 6. よくある質問

### Q1: awaitを付け忘れるとどうなる？

```python
# 間違い
result = async_function()
print(result)  # → <coroutine object async_function at 0x...>

# 正解
result = await async_function()
print(result)  # → 期待した結果
```

警告メッセージも出ます:
`RuntimeWarning: coroutine 'async_function' was never awaited`

### Q2: 通常の関数からasync関数を呼べる？

直接は呼べません。`asyncio.run()`を使います。

```python
def normal_function():
    # これはエラー
    # result = await async_function()

    # これならOK
    result = asyncio.run(async_function())
    return result
```

### Q3: async関数から通常の関数を呼べる？

はい、普通に呼べます。

```python
def normal_function():
    return "結果"

async def async_function():
    result = normal_function()  # awaitは不要
    return result
```

### Q4: await asyncio.sleep(0) は本当に必要？

このコードでは、API呼び出しが同期的なので、`await asyncio.sleep(0)`がないと
ループ中ずっとCPUを占有してしまいます。

もしAPIクライアントが非同期対応（`await client.generate_content()`）なら、
`await asyncio.sleep(0)`は不要です。

---

## 7. まとめ

### 覚えるべき3つのキーワード

1. **async def**: 非同期関数を定義
2. **await**: 非同期関数の完了を待つ（待ち時間中は他の処理が動ける）
3. **asyncio.run()**: 非同期処理を開始する

### このコードでの学び

- 基本的な非同期関数の書き方
- awaitを使った順次実行
- `await asyncio.sleep(0)`でイベントループに制御を戻す方法

### 次のステップ

- `asyncio.gather()`で並列処理を学ぶ
- `asyncio.create_task()`でバックグラウンドタスクを学ぶ
- 非同期対応のHTTPクライアント（aiohttp, httpx）を使う

---
## check_async.py 処理フロー図

## 1. 全体処理フロー

```mermaid
flowchart TD
    subgraph Main["main()"]
        A[開始] --> B{APIキー確認}
        B -->|なし| C[エラー終了]
        B -->|あり| D[テスト用テキスト準備]
        D --> E[process_text 呼び出し]
        E --> F[最終結果表示]
        F --> G[終了]
    end

    subgraph Process["process_text()"]
        E --> P1[Gemini Client初期化]
        P1 --> P2[await step1_hierarchical_split]
        P2 --> P3[await step2_semantic_chunking]
        P3 --> P4[await step3_continuity_check]
        P4 --> P5[final_chunks 返却]
    end

    P5 --> F
```

## 2. データ処理フロー

```mermaid
flowchart LR
    subgraph Input["入力"]
        T["テキスト<br/>(文字列)"]
    end

    subgraph Step1["Step1: 階層構造化"]
        T --> S1["段落分割"]
        S1 --> P["段落リスト<br/>[段落1, 段落2, ...]"]
    end

    subgraph Step2["Step2: 意味的分割"]
        P --> S2["各段落を<br/>意味単位で分割"]
        S2 --> C["チャンクリスト<br/>[chunk1, chunk2, ...]"]
    end

    subgraph Step3["Step3: 連続性チェック"]
        C --> S3["隣接ペア判定<br/>& マージ"]
        S3 --> F["最終チャンクリスト<br/>[final1, final2, ...]"]
    end

    subgraph Output["出力"]
        F --> O["意味的に<br/>まとまった<br/>チャンク群"]
    end
```

## 3. 各Stepの詳細処理

### Step1: 階層構造化（段落分割）

```mermaid
flowchart TD
    subgraph Step1["step1_hierarchical_split()"]
        A["入力テキスト"] --> B["プロンプト作成<br/>PARAGRAPH_SEPARATION_PROMPT"]
        B --> C["Gemini API呼び出し<br/>model: gemini-2.0-flash"]
        C --> D["JSON レスポンス"]
        D --> E["StructuralResult<br/>でパース"]
        E --> F["paragraphs抽出<br/>para.full_text"]
        F --> G["段落リスト返却"]
    end
```

### Step2: 意味的分割

```mermaid
flowchart TD
    subgraph Step2["step2_semantic_chunking()"]
        A["段落リスト"] --> B{"各段落を<br/>ループ処理"}
        B --> C["プロンプト作成<br/>SEMANTIC_CHUNKING_PROMPT"]
        C --> D["Gemini API呼び出し<br/>model: gemini-2.0-flash-exp"]
        D --> E["StructuralResult<br/>でパース"]
        E --> F["チャンク抽出"]
        F --> G["await asyncio.sleep(0)<br/>制御を戻す"]
        G --> B
        B -->|完了| H["チャンクリスト返却"]
    end
```

### Step3: 文脈連続性チェック

```mermaid
flowchart TD
    subgraph Step3["step3_continuity_check()"]
        A["チャンクリスト"] --> B{"チャンク数<br/>≤ 1?"}
        B -->|Yes| C["そのまま返却"]
        B -->|No| D{"隣接ペアを<br/>ループ処理"}
        D --> E["プロンプト作成<br/>CONTINUITY_CHECK_PROMPT"]
        E --> F["Gemini API呼び出し"]
        F --> G["ContinuityResult<br/>でパース"]
        G --> H["is_connected<br/>フラグ保存"]
        H --> I["await asyncio.sleep(0)"]
        I --> D
        D -->|完了| J["マージ処理"]
        J --> K{"is_connected?"}
        K -->|True| L["結合:<br/>前チャンク += 次チャンク"]
        K -->|False| M["分離:<br/>新チャンクとして追加"]
        L --> N["最終チャンクリスト返却"]
        M --> N
    end
```

## 4. 非同期処理の流れ

```mermaid
sequenceDiagram
    participant M as main()
    participant P as process_text()
    participant S1 as step1
    participant S2 as step2
    participant S3 as step3
    participant API as Gemini API

    M->>P: await process_text()

    P->>S1: await step1_hierarchical_split()
    S1->>API: generate_content()
    API-->>S1: response
    S1-->>P: paragraphs

    P->>S2: await step2_semantic_chunking()
    loop 各段落
        S2->>API: generate_content()
        API-->>S2: response
        S2->>S2: await asyncio.sleep(0)
    end
    S2-->>P: chunks

    P->>S3: await step3_continuity_check()
    loop 各ペア
        S3->>API: generate_content()
        API-->>S3: response
        S3->>S3: await asyncio.sleep(0)
    end
    S3->>S3: マージ処理
    S3-->>P: final_chunks

    P-->>M: final_chunks
    M->>M: 結果表示
```

## 5. データ変換の具体例

```mermaid
flowchart TD
    subgraph Input
        I1[第1章AI]
        I2[第2章ML+ラーメン]
        I3[第3章DL]
    end

    subgraph Step1Out
        P1[段落1]
        P2[段落2]
        P3[段落3]
    end

    subgraph Step2Out
        C1[chunk1:AI]
        C2[chunk2:ML]
        C3[chunk3:ラーメン]
        C4[chunk4:DL]
    end

    subgraph Step3Out
        F1[final1:AI+ML]
        F2[final2:ラーメン]
        F3[final3:DL]
    end

    I1 --> P1
    I2 --> P2
    I3 --> P3
    P1 --> C1
    P2 --> C2
    P2 --> C3
    P3 --> C4
    C1 --> F1
    C2 --> F1
    C3 --> F2
    C4 --> F3
```

## 6. 使用するPydanticモデル

```mermaid
classDiagram
    class StructuralResult {
        paragraphs: List
    }

    class ParagraphUnit {
        full_text: str
        sentences: List
    }

    class ContinuityResult {
        is_connected: bool
    }

    StructuralResult --> ParagraphUnit
```

## 7. async/await のポイント

| 要素 | 役割 | 使用箇所 |
|------|------|----------|
| `async def` | 非同期関数を定義 | 全関数 |
| `await` | 非同期関数の完了を待つ | 関数呼び出し時 |
| `asyncio.run()` | イベントループ起動 | `__main__` |
| `await asyncio.sleep(0)` | 制御をイベントループに戻す | Step2, Step3のループ内 |
