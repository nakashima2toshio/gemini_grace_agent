#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qa_generation/generation.py - Q/Aペア生成モジュール
"""

import logging
import time
import json
from typing import List, Dict, Optional
from helper_llm import LLMClient, create_llm_client
from models import QAPairsResponse
from config import DATASET_CONFIGS
from qa_generation.structure import merge_small_chunks

logger = logging.getLogger(__name__)

class QAGenerator:
    """Q/Aペア生成クラス"""

    def __init__(self, client: Optional[LLMClient] = None, model: str = "gemini-2.0-flash"):
        """
        Args:
            client: LLMクライアント
            model: 使用するモデル名
        """
        self.client = client if client else create_llm_client(provider="gemini")
        self.model = model

    def determine_qa_count(self, chunk: Dict, config: Dict) -> int:
        """チャンクに最適なQ/A数を決定"""
        base_count = config["qa_per_chunk"]
        # トークンカウントにはインスタンスのクライアントを使用
        token_count = self.client.count_tokens(chunk['text'], model=self.model)

        # チャンク位置を考慮（文書後半の補正）
        chunk_position = chunk.get('chunk_idx', 0)

        # トークン数に基づく基本Q&A数決定
        if token_count < 50:
            qa_count = 2
        elif token_count < 100:
            qa_count = 3
        elif token_count < 200:
            qa_count = base_count + 1
        elif token_count < 300:
            qa_count = base_count + 2
        else:
            qa_count = base_count + 3

        # 文書後半の位置バイアス補正
        if isinstance(chunk_position, int) and chunk_position >= 5:
            qa_count += 1

        return min(qa_count, 8)

    def generate_for_chunk(self, chunk: Dict, config: Dict) -> List[Dict]:
        """単一チャンクからQ/Aペアを生成"""
        num_pairs = self.determine_qa_count(chunk, config)
        lang = config["lang"]

        # 言語別のプロンプト設定
        if lang == "ja":
            system_prompt = """あなたは教育コンテンツ作成の専門家です。
与えられた日本語テキストから、学習効果の高いQ&Aペアを生成してください。

生成ルール:
1. 質問は明確で具体的に
2. 回答は簡潔で正確に（1-2文程度）
3. テキストの内容に忠実に
4. 多様な観点から質問を作成"""

            question_types_desc = """
- fact: 事実確認型（〜は何ですか？）
- reason: 理由説明型（なぜ〜ですか？）
- comparison: 比較型（〜と〜の違いは？）
- application: 応用型（〜はどのように活用されますか？）"""
        else:
            system_prompt = """You are an expert in educational content creation.
Generate high-quality Q&A pairs from the given English text.

Generation rules:
1. Questions should be clear and specific
2. Answers should be concise and accurate (1-2 sentences)
3. Stay faithful to the text content
4. Create questions from diverse perspectives"""

            question_types_desc = """
- fact: Factual questions (What is...?) 
- reason: Explanatory questions (Why...?) 
- comparison: Comparative questions (What's the difference...?) 
- application: Application questions (How is... used?)"""

        # チャンクが長すぎる場合は短縮
        max_chunk_length = 2000
        chunk_text = chunk['text']
        if len(chunk_text) > max_chunk_length:
            chunk_text = chunk_text[:max_chunk_length] + "..."
            logger.debug(f"チャンクを{max_chunk_length}文字に短縮")

        # 言語に応じたユーザープロンプト
        if lang == "ja":
            user_prompt = f"""以下のテキストから{num_pairs}個のQ&Aペアを生成してください。

質問タイプ:
{question_types_desc}

テキスト:
{chunk_text}

JSON形式で出力:
{{
  "qa_pairs": [
    {{
      "question": "質問文",
      "answer": "回答文",
      "question_type": "fact/reason/comparison/application"
    }}
  ]
}}"""
        else:
            user_prompt = f"""Generate {num_pairs} Q&A pairs from the following text.

Question types:
{question_types_desc}

Text:
{chunk_text}

Output in JSON format:
{{
  "qa_pairs": [
    {{
      "question": "question text",
      "answer": "answer text",
      "question_type": "fact/reason/comparison/application"
    }}
  ]
}}"""

        try:
            combined_input = f"{system_prompt}\n\n{user_prompt}"
            parsed_data = self.client.generate_structured(
                prompt=combined_input,
                response_schema=QAPairsResponse,
                model=self.model,
                max_output_tokens=1000
            )

            qa_pairs = []
            for qa_data in parsed_data.qa_pairs:
                qa = {
                    "question": qa_data.question,
                    "answer": qa_data.answer,
                    "question_type": qa_data.question_type,
                    "source_chunk_id": chunk.get('id', ''),
                    "doc_id": chunk.get('doc_id', ''),
                    "dataset_type": chunk.get('dataset_type', ''),
                    "chunk_idx": chunk.get('chunk_idx', 0)
                }
                qa_pairs.append(qa)

            if len(qa_pairs) == 0:
                logger.error(f"Gemini APIから解析可能なレスポンスが返されませんでした (chunk {chunk.get('id', 'unknown')})")
                raise ValueError("No parseable response from Gemini API")

            return qa_pairs

        except Exception as e:
            logger.warning(f"構造化出力失敗、テキスト生成にフォールバック: {str(e)[:100]}")
            
            try:
                # フォールバック: テキスト生成してJSON解析
                combined_input = f"{system_prompt}\n\n{user_prompt}"
                response_text = self.client.generate_content(
                    prompt=combined_input,
                    model=self.model
                )

                # JSONを抽出して解析
                import re
                json_match = re.search(r'\{.*}', response_text, re.DOTALL)
                if json_match:
                    parsed_data = json.loads(json_match.group())
                    qa_pairs = []
                    for qa_data in parsed_data.get('qa_pairs', []):
                        qa = {
                            "question": qa_data.get('question', ''),
                            "answer": qa_data.get('answer', ''),
                            "question_type": qa_data.get('question_type', 'fact'),
                            "source_chunk_id": chunk.get('id', ''),
                            "doc_id": chunk.get('doc_id', ''),
                            "dataset_type": chunk.get('dataset_type', ''),
                            "chunk_idx": chunk.get('chunk_idx', 0)
                        }
                        qa_pairs.append(qa)
                    return qa_pairs
                else:
                    raise ValueError("JSON not found in response")
            except Exception as fallback_error:
                logger.error(f"フォールバックも失敗 (chunk {chunk.get('id', 'unknown')}): {fallback_error}")
                raise fallback_error

    def generate_for_batch(self, chunks: List[Dict], config: Dict) -> List[Dict]:
        """複数チャンクから一度にQ/Aペアを生成（バッチ処理）"""
        if len(chunks) == 0:
            return []
        if len(chunks) == 1:
            return self.generate_for_chunk(chunks[0], config)

        lang = config["lang"]
        all_qa_pairs = []

        if lang == "ja":
            system_prompt = """あなたは教育コンテンツ作成の専門家です。
複数の日本語テキストから、学習効果の高いQ&Aペアを生成してください。

生成ルール:
1. 質問は明確で具体的に
2. 回答は簡潔で正確に（1-2文程度）
3. テキストの内容に忠実に
4. 多様な観点から質問を作成"""

            combined_text = ""
            chunks_data = {}
            total_pairs = 0

            for i, chunk in enumerate(chunks, 1):
                num_pairs = self.determine_qa_count(chunk, config)
                total_pairs += num_pairs
                chunk_text = chunk['text']
                if len(chunk_text) > 1000:
                    chunk_text = chunk_text[:1000] + "..."
                combined_text += f"\n\n【テキスト{i}】\n{chunk_text}"
                chunks_data[f"chunk_{i}"] = {"num_pairs": num_pairs, "chunk": chunk}

            user_prompt = f"""以下の{len(chunks)}個のテキストから、合計{total_pairs}個のQ&Aペアを生成してください。
{combined_text}

質問タイプ:
- fact: 事実確認型（〜は何ですか？）
- reason: 理由説明型（なぜ〜ですか？）
- comparison: 比較型（〜と〜の違いは？）
- application: 応用型（〜はどのように活用されますか？）

JSON形式で出力:
{{
  "qa_pairs": [
    {{
      "question": "質問文",
      "answer": "回答文",
      "question_type": "fact/reason/comparison/application"
    }}
  ]
}}"""
        else:
            system_prompt = """You are an expert in educational content creation.
Generate high-quality Q&A pairs from multiple English texts.

Generation rules:
1. Questions should be clear and specific
2. Answers should be concise and accurate (1-2 sentences)
3. Stay faithful to the text content
4. Create questions from diverse perspectives"""

            combined_text = ""
            chunks_data = {}
            total_pairs = 0

            for i, chunk in enumerate(chunks, 1):
                num_pairs = self.determine_qa_count(chunk, config)
                total_pairs += num_pairs
                chunk_text = chunk['text']
                if len(chunk_text) > 1000:
                    chunk_text = chunk_text[:1000] + "..."
                combined_text += f"\n\n【Text {i}】\n{chunk_text}"
                chunks_data[f"chunk_{i}"] = {"num_pairs": num_pairs, "chunk": chunk}

            user_prompt = f"""Generate {total_pairs} Q&A pairs from the following {len(chunks)} texts.
{combined_text}

Question types:
- fact: Factual questions (What is...?) 
- reason: Explanatory questions (Why...?) 
- comparison: Comparative questions (What's the difference...?) 
- application: Application questions (How is... used?)

Output in JSON format:
{{
  "qa_pairs": [
    {{
      "question": "question text",
      "answer": "answer text",
      "question_type": "fact/reason/comparison/application"
    }}
  ]
}}"""

        try:
            combined_input = f"{system_prompt}\n\n{user_prompt}"
            parsed_data = self.client.generate_structured(
                prompt=combined_input,
                response_schema=QAPairsResponse,
                model=self.model,
                max_output_tokens=4000
            )

            qa_index = 0
            for i, chunk in enumerate(chunks, 1):
                chunk_key = f"chunk_{i}"
                expected_pairs = chunks_data[chunk_key]["num_pairs"]

                for _ in range(expected_pairs):
                    if qa_index < len(parsed_data.qa_pairs):
                        qa_data = parsed_data.qa_pairs[qa_index]
                        qa = {
                            "question": qa_data.question,
                            "answer": qa_data.answer,
                            "question_type": qa_data.question_type,
                            "source_chunk_id": chunk.get('id', ''),
                            "doc_id": chunk.get('doc_id', ''),
                            "dataset_type": chunk.get('dataset_type', ''),
                            "chunk_idx": chunk.get('chunk_idx', 0)
                        }
                        all_qa_pairs.append(qa)
                        qa_index += 1

            if len(all_qa_pairs) == 0:
                logger.error("Gemini APIから解析可能なレスポンスが返されませんでした")
                raise ValueError("No parseable response from Gemini API")

            return all_qa_pairs

        except Exception as e:
            logger.error(f"バッチQ/A生成エラー: {e}")
            import traceback
            logger.debug(f"スタックトレース: {traceback.format_exc()}")
            logger.info("フォールバック: チャンクを個別処理します")
            for chunk in chunks:
                try:
                    qa_pairs = self.generate_for_chunk(chunk, config)
                    all_qa_pairs.extend(qa_pairs)
                except Exception as chunk_error:
                    logger.error(f"チャンク個別処理エラー: {chunk_error}")
            return all_qa_pairs


def generate_qa_dataset(
    chunks: List[Dict],
    dataset_type: str,
    model: str = "gemini-2.0-flash",
    chunk_batch_size: int = 3,
    merge_chunks: bool = True,
    min_tokens: int = 150,
    max_tokens: int = 400,
    config: Optional[Dict] = None,
    client: Optional[LLMClient] = None
) -> List[Dict]:
    """データセット全体のQ/Aペア生成"""
    if config is None:
        config = DATASET_CONFIGS.get(dataset_type)
        if not config:
            raise ValueError(f"未対応のデータセット: {dataset_type}")

    # クライアント生成（指定がなければ作成）
    if client is None:
        client = create_llm_client(provider="gemini")

    # QAGenerator初期化
    generator = QAGenerator(client=client, model=model)
    all_qa_pairs = []

    # チャンクの前処理（小さいチャンクの統合）
    if merge_chunks:
        processed_chunks = merge_small_chunks(chunks, min_tokens, max_tokens)
    else:
        processed_chunks = chunks

    total_chunks = len(processed_chunks)
    api_calls = (total_chunks + chunk_batch_size - 1) // chunk_batch_size

    logger.info(f"""
    Q/Aペア生成開始:
    - 元チャンク数: {len(chunks)}
    - 処理チャンク数: {total_chunks}
    - バッチサイズ: {chunk_batch_size}
    - API呼び出し予定: {api_calls}回
    - モデル: {model}
    """ )

    # バッチ処理
    for i in range(0, total_chunks, chunk_batch_size):
        batch = processed_chunks[i:i+chunk_batch_size]
        batch_num = i // chunk_batch_size + 1
        total_batches = api_calls

        logger.info(f"バッチ {batch_num}/{total_batches} 処理中 ({len(batch)}チャンク)...")

        # リトライ機能付きQ/A生成
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if chunk_batch_size == 1:
                    qa_pairs = generator.generate_for_chunk(batch[0], config)
                else:
                    qa_pairs = generator.generate_for_batch(batch, config)

                if qa_pairs:
                    all_qa_pairs.extend(qa_pairs)
                    logger.debug(f"バッチ {batch_num}: {len(qa_pairs)}個のQ/Aペア生成")
                break

            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"バッチ {batch_num} 生成失敗: {e}")
                    logger.info("個別処理にフォールバック...")
                    for chunk in batch:
                        try:
                            qa_pairs = generator.generate_for_chunk(chunk, config)
                            if qa_pairs:
                                all_qa_pairs.extend(qa_pairs)
                        except Exception as chunk_error:
                            logger.error(f"チャンク処理エラー: {chunk_error}")
                else:
                    wait_time = 2 ** attempt
                    logger.warning(f"リトライ {attempt + 1}/{max_retries} (待機: {wait_time}秒)")
                    time.sleep(wait_time)

        # API制限対策
        if i + chunk_batch_size < total_chunks:
            time.sleep(0.2)

    logger.info(f"""
    Q/Aペア生成完了:
    - 生成されたQ/Aペア: {len(all_qa_pairs)}個
    - 実行されたAPI呼び出し: 約{api_calls}回
    """ )

    return all_qa_pairs
