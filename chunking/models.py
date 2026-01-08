# models.py
"""
Pydanticモデル定義
Gemini APIのレスポンススキーマとして使用
"""

from typing import List
from pydantic import BaseModel, Field


class SentenceUnit(BaseModel):
    """1つの文、または意味の最小単位"""
    text: str = Field(description="1つの文、または意味の最小単位")


class ParagraphUnit(BaseModel):
    """段落単位"""
    id: int = Field(description="Paragraph ID")
    sentences: List[SentenceUnit] = Field(description="この段落に含まれる文のリスト")

    @property
    def full_text(self) -> str:
        """段落内の全文を結合して返す"""
        return "".join([s.text for s in self.sentences])


class StructuralResult(BaseModel):
    """テキスト構造化の結果"""
    paragraphs: List[ParagraphUnit]


class ContinuityResult(BaseModel):
    """文脈連続性判定の結果"""
    is_connected: bool = Field(
        description="前のテキストと次のテキストが、意味的に連続している（同じトピックである）場合はTrue、話題が転換している場合はFalse"
    )
