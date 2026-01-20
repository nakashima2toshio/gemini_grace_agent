# Q/A生成モジュール比較レポート

## 結論: `make_qa_register_qdrant.py`が使用しているモジュール

**答え: `qa_generation/generation.py`の従来方式（Legacy Mode）**

`make_qa_register_qdrant.py`は以下の呼び出しチェーンでQ/Aペアを生成します：

```
make_qa_register_qdrant.py
    ↓
QAPipeline.run()
    ↓
QAPipeline.generate_qa()
    ↓
generate_qa_dataset(..., use_smart_generation=False)  # ← デフォルトはFalse
    ↓
QAGenerator(..., use_smart_generation=False)
    ↓
QAGenerator._generate_legacy()  # ← 従来方式を使用
```

**重要**: `use_smart_generation`パラメータのデフォルト値は`False`であり、現在はスマート生成モードは**使用されていません**。

---

## 2つのQ/A生成方式の詳細比較

### 1. `generation.py` - 統合Q/A生成モジュール

**ファイル**: `qa_generation/generation.py`
**役割**: 2つの生成方式を統合し、フラグで切り替え可能にする

#### 主要クラス: `QAGenerator`

```python
class QAGenerator:
    def __init__(
        self,
        client: Optional[LLMClient] = None,
        model: str = "gemini-2.0-flash",
        use_smart_generation: bool = False  # デフォルト: 従来方式
    ):
```

---

### 2. 従来方式 (Legacy Mode) - `generation.py`内に実装

#### 2.1 アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│          従来方式 (Legacy Mode)                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  入力: チャンク                                          │
│    ↓                                                    │
│  【ステップ1】Q/A数の決定                                │
│    ├─ トークン数カウント                                │
│    ├─ 固定的なルールベース計算                          │
│    └─ 位置バイアス補正                                  │
│         Token < 50    → 2個                            │
│         Token < 100   → 3個                            │
│         Token < 200   → base_count + 1                 │
│         Token < 300   → base_count + 2                 │
│         Token >= 300  → base_count + 3                 │
│         後半チャンク   → +1個                           │
│    ↓                                                    │
│  【ステップ2】プロンプト生成                             │
│    ├─ システムプロンプト（教育コンテンツ専門家）         │
│    ├─ ユーザープロンプト（Q/A数固定指示）               │
│    ├─ 質問タイプ指定（fact/reason/comparison/app）      │
│    └─ JSON形式での出力指示                             │
│    ↓                                                    │
│  【ステップ3】LLM API呼び出し                            │
│    └─ generate_structured() - 構造化出力               │
│    ↓                                                    │
│  【ステップ4】後処理                                     │
│    ├─ Pydanticモデルでパース（QAPairsResponse）         │
│    ├─ メタデータ追加                                    │
│    │   - source_chunk_id                               │
│    │   - doc_id                                        │
│    │   - generation_method: "legacy"                   │
│    └─ リスト化                                          │
│    ↓                                                    │
│  出力: Q/Aペアリスト（固定数）                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### 2.2 Q/A数決定ロジック（従来方式）

```python
def _legacy_determine_qa_count(self, chunk: Dict, config: Dict) -> int:
    """従来方式のQ/A数決定（後方互換性）"""
    base_count = config["qa_per_chunk"]  # 通常は3
    token_count = self.client.count_tokens(chunk['text'], model=self.model)
    chunk_position = chunk.get('chunk_idx', 0)

    # トークン数に基づく基本Q&A数決定
    if token_count < 50:
        qa_count = 2
    elif token_count < 100:
        qa_count = 3
    elif token_count < 200:
        qa_count = base_count + 1  # 4
    elif token_count < 300:
        qa_count = base_count + 2  # 5
    else:
        qa_count = base_count + 3  # 6

    # 文書後半の位置バイアス補正
    if isinstance(chunk_position, int) and chunk_position >= 5:
        qa_count += 1

    return min(qa_count, 8)  # 最大8個
```

**特徴**:
- ✅ **シンプル**: トークン数のみで判断
- ✅ **高速**: 計算コストがほぼゼロ
- ❌ **機械的**: 内容の重要度を考慮しない
- ❌ **非効率**: 重要でないチャンクからも固定数生成

#### 2.3 プロンプト（従来方式）

```python
system_prompt = """あなたは教育コンテンツ作成の専門家です。
与えられた日本語テキストから、学習効果の高いQ&Aペアを生成してください。

生成ルール:
1. 質問は明確で具体的に
2. 回答は簡潔で正確に（1-2文程度）
3. テキストの内容に忠実に
4. 多様な観点から質問を作成"""

user_prompt = f"""以下のテキストから{num_pairs}個のQ&Aペアを生成してください。

質問タイプ:
- fact: 事実確認型（〜は何ですか？）
- reason: 理由説明型（なぜ〜ですか？）
- comparison: 比較型（〜と〜の違いは？）
- application: 応用型（〜はどのように活用されますか？）

テキスト:
{chunk_text}

JSON形式で出力:
{
  "qa_pairs": [
    {
      "question": "質問文",
      "answer": "回答文",
      "question_type": "fact/reason/comparison/application"
    }
  ]
}"""
```

**特徴**:
- ✅ **明確**: 出力形式が明確に指定されている
- ✅ **構造化**: JSON形式で安定した出力
- ❌ **固定的**: Q/A数が事前に決定されている
- ❌ **柔軟性なし**: チャンクの特性を考慮しない

#### 2.4 出力データ構造（従来方式）

```python
qa_pair = {
    "question": "質問文",
    "answer": "回答文",
    "question_type": "fact",  # fact/reason/comparison/application
    "source_chunk_id": "chunk_0",
    "doc_id": "doc_0",
    "dataset_type": "wikipedia_ja",
    "chunk_idx": 0,
    "generation_method": "legacy"  # 従来方式の印
}
```

---

### 3. スマート方式 (Smart Mode) - `smart_qa_generator.py`

#### 3.1 アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│          スマート方式 (Smart Mode)                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  入力: チャンク                                          │
│    ↓                                                    │
│  【ステップ1】チャンク分析（LLMによる）                  │
│    ├─ 情報密度の評価                                    │
│    ├─ 重要度スコア計算（0.0-1.0）                       │
│    ├─ 複雑さ判定（low/medium/high）                     │
│    ├─ 主要トピックの抽出                                │
│    └─ 適切なQ/A数の決定（0-5個）                        │
│         0個: メタ情報のみ、補足情報                     │
│         1個: 単一事実                                   │
│         2個: 関連する2つの事実                          │
│         3個: 標準的な説明パラグラフ                     │
│         4-5個: 高密度技術情報、警告事項                 │
│    ↓                                                    │
│  【ステップ2】Q/A生成（LLMによる）                       │
│    ├─ 分析結果に基づく動的プロンプト                    │
│    ├─ 主要トピックの明示的指示                          │
│    ├─ 重要度に応じた品質指示                            │
│    └─ トピック付きQ/A生成                               │
│    ↓                                                    │
│  【ステップ3】品質チェック                               │
│    ├─ 期待Q/A数との整合性確認                           │
│    ├─ トピックフィールドの補完                          │
│    └─ 重複チェック                                      │
│    ↓                                                    │
│  出力: Q/Aペアリスト（動的数）+ 分析メタデータ           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### 3.2 チャンク分析（スマート方式）

```python
def analyze_chunk(self, chunk_text: str) -> Dict:
    """
    チャンクを分析してQ/A生成計画を立てる

    Returns:
        dict: {
            'qa_count': int,           # 生成すべきQ/A数（0-5）
            'key_topics': List[str],   # 主要トピック
            'importance_score': float, # 重要度（0.0-1.0）
            'complexity': str,         # 複雑さ（low/medium/high）
            'reasoning': str           # 判断理由
        }
    """
```

**分析プロンプト（一部抜粋）**:

```
以下のテキストチャンクを分析し、Q/Aペアの生成計画を立ててください。

# 分析観点
1. **情報密度**: このチャンクに含まれる独立した情報・事実の数
2. **重要度**: 情報の重要性（critical/high/medium/low）
3. **複雑さ**: 説明に必要な詳細度（high/medium/low）
4. **独立性**: 各情報が他の文脈なしで理解可能か

# 判断基準
## 0個（Q/A生成不要）:
- 補足情報のみ（「詳細は付録参照」など）
- 意味のない繰り返し
- メタ情報のみ（ページ番号、参照リンクなど）

## 1個:
- 単純な事実の記述（1つの情報のみ）

## 2個:
- 関連する2つの事実

## 3個（標準）:
- 複数の関連情報
- 標準的な説明パラグラフ

## 4-5個:
- 高密度な技術情報
- 複数の独立したポイント
- 重要な警告や注意事項を含む
```

**特徴**:
- ✅ **インテリジェント**: 内容を理解して判断
- ✅ **適応的**: チャンクの特性に応じて変化
- ✅ **効率的**: 不要なQ/A生成を回避
- ❌ **コスト**: 追加のLLM呼び出しが必要
- ❌ **遅い**: 分析ステップが追加される

#### 3.3 Q/A生成（スマート方式）

```python
def generate_qa_pairs(
    self,
    chunk_text: str,
    analysis: Optional[Dict] = None
) -> List[Dict]:
    """
    分析結果に基づいてQ/Aペアを生成

    Returns:
        List[Dict]: [{'question': str, 'answer': str, 'topic': str}, ...]
    """
```

**動的プロンプト生成**:

```python
# トピックヒントの作成（分析結果から）
topics_hint = ""
if analysis['key_topics']:
    topics_hint = "\n## 重点トピック\n以下のトピックを優先的にカバーしてください:\n" + \
                  "\n".join([f"- {topic}" for topic in analysis['key_topics']])

# 重要度に基づく指示
importance_hint = ""
if analysis['importance_score'] >= 0.8:
    importance_hint = "\n## 重要度\nこのチャンクは非常に重要です。詳細で正確なQ/Aを生成してください。"

prompt = f"""
以下のテキストから、**正確に{qa_count}個**のQ/Aペアを生成してください。

# 生成計画
- 生成数: {qa_count}個
- 重要度スコア: {analysis['importance_score']:.2f}
- 複雑さ: {analysis['complexity']}
{topics_hint}
{importance_hint}

# テキスト
{chunk_text}
"""
```

**特徴**:
- ✅ **コンテキスト対応**: チャンクの特性を反映
- ✅ **品質重視**: 重要度に応じた指示
- ✅ **トピック明示**: 主要概念を確実にカバー
- ❌ **複雑**: プロンプト構築が動的で複雑

#### 3.4 出力データ構造（スマート方式）

```python
qa_pair = {
    "question": "質問文",
    "answer": "回答文",
    "question_type": "fact",  # デフォルト
    "topic": "暗号化方式",    # ✨ 新規フィールド
    "source_chunk_id": "chunk_0",
    "doc_id": "doc_0",
    "dataset_type": "wikipedia_ja",
    "chunk_idx": 0,
    # ✨ スマート生成メタデータ
    "generation_method": "smart",
    "importance_score": 0.85,
    "complexity": "high"
}
```

---

## 処理方式の詳細比較表

### 比較1: アーキテクチャレベル

| 観点 | 従来方式 (Legacy) | スマート方式 (Smart) |
|------|------------------|---------------------|
| **モジュール** | `generation.py`内に実装 | `smart_qa_generator.py` + `generation.py`統合 |
| **クラス** | `QAGenerator._generate_legacy()` | `SmartQAGenerator` + `QAGenerator._generate_smart()` |
| **LLM呼び出し回数** | 1回/チャンク（または/バッチ） | 2回/チャンク（分析+生成） |
| **処理ステップ** | 2ステップ（Q/A数決定→生成） | 3ステップ（分析→Q/A数決定→生成） |

### 比較2: Q/A数決定ロジック

| 観点 | 従来方式 (Legacy) | スマート方式 (Smart) |
|------|------------------|---------------------|
| **決定方法** | トークン数ベースの固定ルール | LLMによる内容分析 |
| **判断基準** | トークン数のみ | 情報密度、重要度、複雑さ、独立性 |
| **Q/A数範囲** | 2-8個（最大値制限あり） | 0-5個（柔軟） |
| **0個生成** | 不可能 | 可能（不要な場合スキップ） |
| **位置バイアス** | あり（後半+1） | なし |
| **計算コスト** | ほぼゼロ（トークンカウントのみ） | 高（LLM呼び出し） |

### 比較3: プロンプト設計

| 観点 | 従来方式 (Legacy) | スマート方式 (Smart) |
|------|------------------|---------------------|
| **システムプロンプト** | 固定（教育コンテンツ専門家） | 動的（分析結果に応じて変化） |
| **Q/A数指定** | 事前に固定数を指定 | 分析結果に基づく動的数 |
| **トピック指示** | なし | 主要トピックを明示的に指示 |
| **重要度考慮** | なし | 重要度スコアに応じた品質指示 |
| **質問タイプ** | 4種類固定（fact/reason/comparison/app） | 動的（トピックベース） |

### 比較4: 出力品質

| 観点 | 従来方式 (Legacy) | スマート方式 (Smart) |
|------|------------------|---------------------|
| **トピックカバレッジ** | 保証なし | 主要トピックを確実にカバー |
| **情報の重要度** | 考慮しない | 重要な情報を優先 |
| **冗長性** | 発生しやすい（固定数生成） | 少ない（必要な分のみ） |
| **メタデータ** | 基本情報のみ | 分析情報付き（importance, complexity） |
| **topic フィールド** | なし | あり（各Q/Aの主題） |

### 比較5: パフォーマンス

| 観点 | 従来方式 (Legacy) | スマート方式 (Smart) |
|------|------------------|---------------------|
| **処理速度** | 高速 | 低速（2倍の時間） |
| **API呼び出し** | 少ない | 多い（2倍） |
| **コスト** | 低い | 高い（約2倍） |
| **メモリ使用量** | 標準 | やや高い（分析結果保持） |

### 比較6: 使用例

#### 従来方式の典型例

**入力チャンク** (150トークン):
```
この製品は赤色で、サイズはMサイズです。
価格は3,000円で、送料無料です。
```

**処理**:
1. トークン数カウント: 150 → `qa_count = 4`（base_count + 1）
2. プロンプト生成（4個固定）
3. LLM呼び出し

**出力** (4個):
```json
[
  {"question": "この製品の色は？", "answer": "赤色です"},
  {"question": "サイズは？", "answer": "Mサイズです"},
  {"question": "価格は？", "answer": "3,000円です"},
  {"question": "送料は？", "answer": "無料です"}
]
```

**問題点**: 情報量に対してQ/A数が多すぎる（4個は過剰）

#### スマート方式の典型例

**入力チャンク** (同じ内容):
```
この製品は赤色で、サイズはMサイズです。
価格は3,000円で、送料無料です。
```

**処理**:
1. **分析フェーズ**:
   ```json
   {
     "qa_count": 2,
     "key_topics": ["製品仕様", "価格"],
     "importance_score": 0.4,
     "complexity": "low",
     "reasoning": "単純な製品情報のみ"
   }
   ```
2. **生成フェーズ**: 2個のQ/A生成

**出力** (2個):
```json
[
  {
    "question": "この製品の仕様は？",
    "answer": "赤色でMサイズです",
    "topic": "製品仕様"
  },
  {
    "question": "価格と送料は？",
    "answer": "3,000円で送料無料です",
    "topic": "価格"
  }
]
```

**利点**: 情報を適切に統合し、必要最小限のQ/A数

---

## 詳細な比較: 技術的な重要チャンクでの挙動

### テストケース: 技術文書チャンク

**入力**:
```
AES-256暗号化アルゴリズムは、対称鍵暗号方式の一種で、
256ビットの鍵長を持ちます。NIST（米国国立標準技術研究所）
により承認されており、機密情報の保護に広く使用されています。
ブロック暗号として動作し、128ビットのブロックサイズで
データを処理します。CBC、GCM、CTRなど複数のモードが利用可能で、
用途に応じて選択できます。
```

### 従来方式の処理

**ステップ1: Q/A数決定**
```python
token_count = 180  # 概算
# token_count < 200 なので:
qa_count = base_count + 1 = 3 + 1 = 4
```

**ステップ2: 生成**
```
プロンプト: "以下のテキストから4個のQ&Aペアを生成してください"
```

**出力** (4個):
```json
[
  {
    "question": "AES-256とは何ですか？",
    "answer": "対称鍵暗号方式の一種で、256ビットの鍵長を持つ暗号化アルゴリズムです",
    "question_type": "fact"
  },
  {
    "question": "AES-256はどこで承認されていますか？",
    "answer": "NIST（米国国立標準技術研究所）により承認されています",
    "question_type": "fact"
  },
  {
    "question": "ブロックサイズは何ビットですか？",
    "answer": "128ビットのブロックサイズでデータを処理します",
    "question_type": "fact"
  },
  {
    "question": "利用可能なモードは？",
    "answer": "CBC、GCM、CTRなど複数のモードが利用可能です",
    "question_type": "fact"
  }
]
```

**評価**:
- ✅ 基本的な情報はカバー
- ❌ トピックが不明確
- ❌ 重要度の区別なし
- ❌ 質問タイプが全て"fact"

### スマート方式の処理

**ステップ1: 分析**
```json
{
  "qa_count": 5,
  "key_topics": [
    "暗号化方式",
    "鍵長",
    "ブロックサイズ",
    "利用モード",
    "承認機関"
  ],
  "importance_score": 0.9,
  "complexity": "high",
  "reasoning": "高密度な技術情報を含み、複数の独立したポイントがある"
}
```

**ステップ2: 生成**
```
プロンプト:
"以下のテキストから、**正確に5個**のQ/Aペアを生成してください。

# 生成計画
- 生成数: 5個
- 重要度スコア: 0.90
- 複雑さ: high

## 重点トピック
以下のトピックを優先的にカバーしてください:
- 暗号化方式
- 鍵長
- ブロックサイズ
- 利用モード
- 承認機関

## 重要度
このチャンクは非常に重要です。詳細で正確なQ/Aを生成してください。
"
```

**出力** (5個):
```json
[
  {
    "question": "AES-256の暗号化方式の特徴は何ですか？",
    "answer": "対称鍵暗号方式の一種で、256ビットの鍵長を持ち、高いセキュリティを提供します",
    "topic": "暗号化方式"
  },
  {
    "question": "AES-256の鍵長は何ビットですか？",
    "answer": "256ビットの鍵長を持ちます",
    "topic": "鍵長"
  },
  {
    "question": "AES-256のブロックサイズはどのくらいですか？",
    "answer": "128ビットのブロックサイズでデータを処理します",
    "topic": "ブロックサイズ"
  },
  {
    "question": "AES-256で利用可能な動作モードにはどのようなものがありますか？",
    "answer": "CBC、GCM、CTRなど複数のモードが利用可能で、用途に応じて選択できます",
    "topic": "利用モード"
  },
  {
    "question": "AES-256はどの機関により承認されていますか？",
    "answer": "NIST（米国国立標準技術研究所）により承認されており、機密情報の保護に広く使用されています",
    "topic": "承認機関"
  }
]
```

**評価**:
- ✅ 主要トピックを全てカバー
- ✅ 各Q/Aにトピックラベル付き
- ✅ 重要度に応じた詳細な回答
- ✅ 情報の独立性を保持

---

## 統合の仕組み（generation.py）

### フラグによる切り替え

```python
class QAGenerator:
    def __init__(
        self,
        client: Optional[LLMClient] = None,
        model: str = "gemini-2.0-flash",
        use_smart_generation: bool = False  # ← 重要なフラグ
    ):
        self.use_smart_generation = use_smart_generation

        if self.use_smart_generation:
            logger.info("🆕 スマート生成モードを有効化")
            self.smart_generator = SmartQAGenerator(model=model)
        else:
            logger.info("🔧 従来の固定Q/A数生成モードを使用")
            self.smart_generator = None
```

### 実行時の分岐

```python
def generate_for_chunk(self, chunk: Dict, config: Dict) -> List[Dict]:
    """単一チャンクからQ/Aペアを生成"""
    if self.use_smart_generation:
        # ✨ スマート生成
        return self._generate_smart(chunk, config)
    else:
        # 🔧 従来方式
        return self._generate_legacy(chunk, config)
```

---

## 長所・短所の総合比較

### 従来方式 (Legacy Mode)

#### 長所 ✅

1. **高速処理**
   - LLM呼び出し1回のみ
   - トークンカウントのみで計算
   - バッチ処理にも対応

2. **低コスト**
   - APIコストが最小
   - 大量データ処理に適している

3. **安定性**
   - シンプルなロジック
   - デバッグが容易
   - エラーハンドリングが簡単

4. **後方互換性**
   - 既存システムと完全互換
   - 移行リスクゼロ

5. **バッチ処理対応**
   - 複数チャンクを一度に処理可能
   - API効率が良い

#### 短所 ❌

1. **非効率な生成**
   - 不要なチャンクからもQ/A生成
   - 情報量に対して過剰/不足の可能性

2. **品質のばらつき**
   - 重要度を考慮しない
   - トピックカバレッジの保証なし

3. **冗長性**
   - 低品質なQ/Aも生成される
   - 後処理フィルタリングが必要

4. **メタデータ不足**
   - トピック情報なし
   - 重要度スコアなし

### スマート方式 (Smart Mode)

#### 長所 ✅

1. **インテリジェント**
   - 内容を理解してQ/A数決定
   - 重要度に応じた品質調整

2. **効率的**
   - 不要なQ/A生成を回避
   - 0個生成も可能

3. **高品質**
   - 主要トピックを確実にカバー
   - 重要情報を優先

4. **豊富なメタデータ**
   - トピックラベル付き
   - 重要度スコア、複雑さ情報

5. **適応的**
   - チャンクの特性に応じて変化
   - 柔軟なQ/A数（0-5個）

#### 短所 ❌

1. **低速**
   - LLM呼び出し2回必要
   - 処理時間が約2倍

2. **高コスト**
   - APIコストが約2倍
   - 大量データ処理では高額

3. **複雑性**
   - デバッグが困難
   - エラーハンドリングが複雑

4. **バッチ処理非対応**
   - チャンクごとに分析が必要
   - 並列化が難しい

5. **不確実性**
   - LLM判断に依存
   - 分析結果のばらつき

---

## 推奨される使用シナリオ

### 従来方式を使うべき場合

1. **大規模データセット処理**
   - 10,000チャンク以上
   - コスト重視

2. **高速処理が必要**
   - リアルタイム処理
   - バッチジョブ

3. **安定性重視**
   - 本番環境
   - ミッションクリティカル

4. **後方互換性が必要**
   - 既存システムとの統合

### スマート方式を使うべき場合

1. **品質重視**
   - 高品質なQ/Aが必要
   - トピックカバレッジが重要

2. **少量データ処理**
   - 100-1,000チャンク程度
   - コストが許容範囲

3. **多様なコンテンツ**
   - 技術文書と一般文書が混在
   - 重要度にばらつきがある

4. **分析機能が必要**
   - メタデータ活用
   - フィルタリング・ソート

---

## 現状と今後の方向性

### 現状: 従来方式がデフォルト

```python
# pipeline.pyで呼び出される
generate_qa_dataset(
    chunks,
    dataset_type,
    model,
    ...,
    use_smart_generation=False  # ← デフォルトはFalse
)
```

**理由**:
- 安定性とパフォーマンス優先
- 大量データ処理が必要
- コスト最適化

### スマート方式の有効化方法

現在、スマート方式を有効にするには、コードを直接修正する必要があります：

```python
# pipeline.pyの_generate_sync()内で
return generate_qa_dataset(
    chunks,
    dataset_type,
    self.model,
    chunk_batch_size=batch_size,
    merge_chunks=merge,
    min_tokens=min_tokens,
    max_tokens=max_tokens,
    config=self.config,
    client=self.client,
    use_smart_generation=True  # ← ここを変更
)
```

### 今後の改善案

1. **CLIオプション追加**
   ```bash
   python make_qa_register_qdrant.py \
     --input-file doc.txt \
     --collection my_docs \
     --use-smart-generation  # ← 新規オプション
   ```

2. **ハイブリッドモード**
   - 重要チャンクのみスマート生成
   - その他は従来方式
   - コストと品質のバランス

3. **キャッシング機構**
   - 分析結果のキャッシュ
   - 再処理時の高速化

4. **並列処理対応**
   - スマート方式の並列化
   - バッチ分析機能

---

## まとめ

### Q1: `make_qa_register_qdrant.py`が使用しているのは？

**答え**: `qa_generation/generation.py`の**従来方式（Legacy Mode）**

- `use_smart_generation=False`がデフォルト
- トークン数ベースの固定Q/A数決定
- 1回のLLM呼び出しで生成

### Q2: 2つの方式の最大の違いは？

| 項目 | 従来方式 | スマート方式 |
|------|---------|------------|
| **Q/A数決定** | トークン数のみ | LLMによる内容分析 |
| **処理速度** | 高速 | 低速（2倍） |
| **品質** | 標準 | 高品質 |
| **コスト** | 低い | 高い（2倍） |
| **0個生成** | 不可 | 可能 |

### Q3: どちらを使うべきか？

- **従来方式**: 大規模処理、速度重視、コスト重視
- **スマート方式**: 品質重視、少量データ、メタデータ活用

---

**作成日**: 2025-01-20
**作成者**: AI Assistant
