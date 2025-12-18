#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qa_generation/semantic.py - セマンティック分析・カバレッジ測定モジュール
"""

import re
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import tiktoken
from helper_llm import create_llm_client
from helper_embedding import create_embedding_client, get_embedding_dimensions

logger = logging.getLogger(__name__)

class SemanticCoverage:
    """意味的な網羅性を測定するクラス（Gemini API使用）"""

    def __init__(self, embedding_model="gemini-embedding-001"):
        self.embedding_model = embedding_model
        # Gemini埋め込みクライアントを使用
        self.embedding_client = create_embedding_client(provider="gemini")
        self.embedding_dims = get_embedding_dimensions("gemini")  # 3072
        # トークンカウント用のLLMクライアント (decode機能がないためtiktokenを併用)
        self.unified_client = create_llm_client(provider="gemini") 
        self.tokenizer = tiktoken.get_encoding("cl100k_base") # 強制分割・デコード用にtiktokenを使用
        
        # APIキーの有無フラグ（クライアント作成成功ならTrue）
        self.has_api_key = True

        # MeCab利用可否チェック
        self.mecab_available = self._check_mecab_availability()


    def _check_mecab_availability(self) -> bool:
        """MeCabの利用可能性をチェック"""
        try:
            import MeCab
            # 実際にインスタンス化して動作確認
            tagger = MeCab.Tagger()
            tagger.parse("テスト")
            return True
        except (ImportError, RuntimeError):
            return False

    def create_semantic_chunks(self, document: str, max_tokens: int = 200, min_tokens: int = 50,
                               prefer_paragraphs: bool = True, verbose: bool = True) -> List[Dict]:
        """
        文書を意味的に区切られたチャンクに分割（段落優先のセマンティック分割）

        重要ポイント：
        1. 段落の境界で分割（最優先 - 筆者の意図したセマンティック境界）
        2. 文の境界で分割（意味の断絶を防ぐ）
        3. トピックの変化を検出
        4. 適切なサイズを維持（埋め込みモデルの制限内）

        Args:
            document: 分割対象の文書
            max_tokens: チャンクの最大トークン数（デフォルト: 200）
            min_tokens: チャンクの最小トークン数（デフォルト: 50）
            prefer_paragraphs: 段落ベースの分割を優先するか（デフォルト: True）
            verbose: 詳細な出力を行うか

        Returns:
            チャンク辞書のリスト（id, text, type, sentences等を含む）
        """

        # Step 1: 段落ベースの分割を試行（prefer_paragraphs=Trueの場合）
        if prefer_paragraphs:
            para_chunks = self._chunk_by_paragraphs(document, max_tokens, min_tokens)

            if verbose:
                logger.info(f"段落ベースのチャンク数: {len(para_chunks)}")
                type_counts = {}
                for chunk in para_chunks:
                    chunk_type = chunk.get('type', 'unknown')
                    type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
                logger.info(f"チャンクタイプ内訳: {type_counts}")

            # 段落ベースのチャンクを標準フォーマットに変換
            chunks = []
            for i, chunk in enumerate(para_chunks):
                chunk_text = chunk['text']
                sentences = self._split_into_sentences(chunk_text)
                chunks.append({
                    "id"                : f"chunk_{i}",
                    "text"              : chunk_text,
                    "type"              : chunk['type'],
                    "sentences"         : sentences,
                    "start_sentence_idx": 0,  # 段落ベースでは文インデックスは相対的
                    "end_sentence_idx"  : len(sentences) - 1
                })
        else:
            # Step 1 (旧方式): 文単位で分割
            sentences = self._split_into_sentences(document)
            if verbose:
                logger.info(f"文の数: {len(sentences)}")

            # Step 2: 意味的に関連する文をグループ化
            chunks = []
            current_chunk = []
            current_tokens = 0

            for i, sentence in enumerate(sentences):
                sentence_tokens = self.unified_client.count_tokens(sentence, model=self.embedding_model)

                # 現在のチャンクにこの文を追加すべきか判断
                if current_tokens + sentence_tokens > max_tokens and current_chunk:
                    # Use unified client for token counting when forming chunks
                    chunk_text_tokens = self.unified_client.count_tokens(" ".join(current_chunk), model=self.embedding_model)
                    # チャンクを保存
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "id"                : f"chunk_{len(chunks)}",
                        "text"              : chunk_text,
                        "type"              : "sentence_group",
                        "sentences"         : current_chunk.copy(),
                        "start_sentence_idx": i - len(current_chunk),
                        "end_sentence_idx"  : i - 1
                    })
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens

            # 最後のチャンクを追加
            if current_chunk:
                chunks.append({
                    "id"                : f"chunk_{len(chunks)}",
                    "text"              : " ".join(current_chunk),
                    "type"              : "sentence_group",
                    "sentences"         : current_chunk,
                    "start_sentence_idx": len(sentences) - len(current_chunk),
                    "end_sentence_idx"  : len(sentences) - 1
                })

        # Step 3: トピックの連続性を考慮した再調整
        chunks = self._adjust_chunks_for_topic_continuity(chunks, min_tokens)

        return chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        段落単位で分割（セマンティック分割の最優先レベル）

        段落は筆者が意図的に作った意味的なまとまりであり、
        最も重要なセマンティック境界となる
        """
        # 空行（\n\n）で段落を分割
        paragraphs = re.split(r'\n\s*\n', text)

        # 空白のみの段落を除外
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        return paragraphs

    def _chunk_by_paragraphs(self, text: str, max_tokens: int = 200, min_tokens: int = 50) -> List[Dict[str, Any]]:
        """
        段落単位でチャンク化（セマンティック最優先）

        段落をベースにチャンクを作成し、トークン数制限を考慮する。
        段落が大きすぎる場合は文単位に分割する。

        Args:
            text: 分割対象のテキスト
            max_tokens: チャンクの最大トークン数
            min_tokens: チャンクの最小トークン数（これより小さい場合は次と結合を検討）

        Returns:
            チャンクのリスト（各チャンクは {'text': str, 'type': str} の辞書）
        """
        paragraphs = self._split_into_paragraphs(text)
        chunks = []

        for para in paragraphs:
            para_tokens = self.unified_client.count_tokens(para, model=self.embedding_model)

            if para_tokens <= max_tokens:
                # 段落がそのままチャンクとして適切
                chunks.append({'text': para, 'type': 'paragraph'})
            else:
                # 段落が大きすぎる → 文単位に分割
                sentences = self._split_into_sentences(para)
                current_chunk = []
                current_tokens = 0

                for sent in sentences:
                    sent_tokens = self.unified_client.count_tokens(sent, model=self.embedding_model)

                    if sent_tokens > max_tokens:
                        # 単一文が上限超過 → 強制分割
                        if current_chunk:
                            chunks.append({'text': ''.join(current_chunk), 'type': 'sentence_group'})
                            # No need to recalculate current_tokens, just reset
                            current_chunk = []
                            current_tokens = 0

                        # 強制分割を実施
                        forced_chunks = self._force_split_sentence(sent, max_tokens)
                        chunks.extend(forced_chunks)

                    elif current_tokens + sent_tokens > max_tokens:
                        # 追加すると上限超過 → 現在のチャンクを確定
                        if current_chunk:
                            chunks.append({'text': ''.join(current_chunk), 'type': 'sentence_group'})
                        current_chunk = [sent]
                        current_tokens = sent_tokens

                    else:
                        # 追加可能
                        current_chunk.append(sent)
                        current_tokens += sent_tokens

                # 残りを確定
                if current_chunk:
                    chunks.append({'text': ''.join(current_chunk), 'type': 'sentence_group'})

        return chunks

    def _force_split_sentence(self, sentence: str, max_tokens: int = 200) -> List[Dict[str, Any]]:
        """
        単一文が上限超過の場合に強制的に分割（最終手段）

        セマンティック境界を無視して、トークン数ベースで強制的に分割する。
        これは意味的な一貫性を犠牲にするが、処理上の制約を守るために必要。

        Args:
            sentence: 分割対象の文
            max_tokens: チャンクの最大トークン数

        Returns:
            強制分割されたチャンクのリスト
        """
        # トークンレベルで分割
        tokens = self.tokenizer.encode(sentence)
        forced_chunks = []

        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            forced_chunks.append({
                'text': chunk_text,
                'type': 'forced_split'
            })

        return forced_chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """文単位で分割（言語自動判定・MeCab優先対応）"""

        # 日本語判定（最初の100文字で判定）
        is_japanese = bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text[:100]))

        if is_japanese and self.mecab_available:
            # 日本語の場合、MeCab利用を優先（セマンティック精度向上）
            try:
                sentences = self._split_sentences_mecab(text)
                if sentences:
                    return sentences
            except Exception:
                pass  # フォールバック

        # 英語 or MeCab失敗時: 正規表現
        # 句点等で終わる塊を抽出
        sentences = re.findall(r'[^。．.！？!?]+[。．.！？!?]\s*', text)
        if not sentences:
            # 句点がない場合は全体を1つの文とする
            sentences = [text.strip()] if text.strip() else []
        else:
            # 最後の文の後に句点がないテキストが残っている場合
            last_pos = text.rfind(sentences[-1]) + len(sentences[-1])
            if last_pos < len(text):
                remaining = text[last_pos:].strip()
                if remaining:
                    sentences.append(remaining)
        
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _split_sentences_mecab(self, text: str) -> List[str]:
        """MeCabを使った文分割（日本語用）"""
        import MeCab

        tagger = MeCab.Tagger()
        node = tagger.parseToNode(text)

        sentences = []
        current_sentence = []

        while node:
            surface = node.surface
            node.feature.split(',')

            if surface:
                current_sentence.append(surface)

                # 文末判定：句点（。）、疑問符（？）、感嘆符（！）
                if surface in ['。', '．', '？', '！', '?', '!']:
                    sentence = ''.join(current_sentence).strip()
                    if sentence:
                        sentences.append(sentence)
                    current_sentence = []

            node = node.next

        # 最後の文を追加
        if current_sentence:
            sentence = ''.join(current_sentence).strip()
            if sentence:
                sentences.append(sentence)

        return sentences if sentences else []

    def _adjust_chunks_for_topic_continuity(self, chunks: List[Dict], min_tokens: int = 50) -> List[Dict]:
        """
        トピックの連続性を考慮してチャンクを調整（最小トークン数対応）

        短すぎるチャンクを隣接チャンクとマージして意味的まとまりを維持する。

        Args:
            chunks: チャンクのリスト
            min_tokens: チャンクの最小トークン数（これより小さい場合はマージを検討）

        Returns:
            調整後のチャンクリスト
        """
        adjusted_chunks = []

        for i, chunk in enumerate(chunks):
            chunk_tokens = self.unified_client.count_tokens(chunk["text"], model=self.embedding_model)

            # 最小トークン数以下の短いチャンクの場合
            if i > 0 and chunk_tokens < min_tokens:
                # 前のチャンクとマージを検討
                prev_chunk = adjusted_chunks[-1]
                combined_text = prev_chunk["text"] + " " + chunk["text"]
                combined_tokens = self.unified_client.count_tokens(combined_text, model=self.embedding_model)

                # マージしても最大トークン数（300）を超えない場合はマージ
                if combined_tokens < 300:
                    # マージ実施
                    prev_chunk["text"] = combined_text
                    prev_chunk["sentences"].extend(chunk["sentences"])
                    prev_chunk["end_sentence_idx"] = chunk["end_sentence_idx"]

                    # typeの更新（異なるtypeがマージされた場合）
                    if prev_chunk.get("type") != chunk.get("type"):
                        prev_chunk["type"] = "merged"

                    continue

            adjusted_chunks.append(chunk)

        return adjusted_chunks

    def generate_embeddings(self, doc_chunks: List[Dict]) -> np.ndarray:
        """
        チャンクのリストから埋め込みベクトルを生成（Gemini API使用）

        重要ポイント：
        1. バッチ処理で効率化
        2. エラーハンドリング
        3. 正規化（コサイン類似度計算の準備）
        """

        if not self.has_api_key:
            print("⚠️  Gemini APIキーが設定されていません。埋め込み生成をスキップします。")
            # ダミーのゼロベクトルを返す
            return np.zeros((len(doc_chunks), self.embedding_dims))

        texts = [chunk["text"] for chunk in doc_chunks]

        try:
            # Gemini Embedding APIを呼び出し
            embedding_vectors = self.embedding_client.embed_texts(texts, batch_size=100)

            # 埋め込みベクトルを正規化
            embeddings = []
            for embedding in embedding_vectors:
                embedding = np.array(embedding)
                # L2正規化（コサイン類似度の計算を高速化）
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                embeddings.append(embedding)

            return np.array(embeddings)

        except Exception as e:
            print(f"埋め込み生成エラー: {e}")
            # エラー時はゼロベクトルを返す
            return np.zeros((len(doc_chunks), self.embedding_dims))

    def generate_embedding(self, text: str) -> np.ndarray:
        """単一テキストの埋め込み生成（Gemini API使用）"""
        if not self.has_api_key:
            return np.zeros(self.embedding_dims)

        try:
            # Gemini Embedding APIを使用
            embedding = self.embedding_client.embed_text(text)
            embedding = np.array(embedding)
            # 正規化
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        except Exception as e:
            print(f"埋め込み生成エラー: {e}")
            return np.zeros(self.embedding_dims)

    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        複数テキストの埋め込みを一括生成（Gemini API使用）

        Args:
            texts: テキストのリスト
            batch_size: 1リクエストあたりのテキスト数（デフォルト: 100）

        Returns:
            埋め込みベクトルの配列 (len(texts), 3072)
        """
        if not self.has_api_key:
            print("⚠️  Gemini APIキーが設定されていません。埋め込み生成をスキップします。")
            return np.zeros((len(texts), self.embedding_dims))

        try:
            # Gemini Embedding APIを使用
            embedding_vectors = self.embedding_client.embed_texts(texts, batch_size=batch_size)

            # 埋め込みベクトルを正規化
            embeddings = []
            for embedding in embedding_vectors:
                embedding = np.array(embedding)
                # L2正規化
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                embeddings.append(embedding)

            return np.array(embeddings)

        except Exception as e:
            print(f"バッチ埋め込み生成エラー: {e}")
            # エラー時はゼロベクトルを返す
            return np.zeros((len(texts), self.embedding_dims))

    def cosine_similarity(self, doc_emb: np.ndarray, qa_emb: np.ndarray) -> float:
        """
        コサイン類似度を計算

        重要ポイント：
        1. 事前に正規化済みなら内積で計算可能
        2. 範囲は[-1, 1]、1に近いほど類似
        """

        # ベクトルが正規化済みの場合は内積で計算
        if np.allclose(np.linalg.norm(doc_emb), 1.0) and \
                np.allclose(np.linalg.norm(qa_emb), 1.0):
            return float(np.dot(doc_emb, qa_emb))

        # 正規化されていない場合は完全な計算
        dot_product = np.dot(doc_emb, qa_emb)
        norm_doc = np.linalg.norm(doc_emb)
        norm_qa = np.linalg.norm(qa_emb)

        if norm_doc == 0 or norm_qa == 0:
            return 0.0

        return float(dot_product / (float(norm_doc) * float(norm_qa)))