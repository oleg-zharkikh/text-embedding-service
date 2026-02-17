from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Входной текст для векторизации."""

    text: str = Field(..., min_length=1, description='Input text')


class EmbeddingResponse(BaseModel):
    """Эмбеддинг."""

    embedding: Dict[str, Any]


class CompareRequest(BaseModel):
    """Входные тексты для сравнения."""

    etalon: str = Field(..., min_length=1, description='Reference text')
    text: str = Field(..., min_length=1, description='Text to compare')


class CompareResponse(BaseModel):
    """Результат сравнения двух текстов - число от 0 до 1."""

    distance: float


class CompareBatchRequest(BaseModel):
    """Входные данные для сравнения - эталон и список текстов."""

    etalon: str = Field(..., min_length=1, description="Reference text")
    texts: List[str] = Field(
        ..., min_items=1, description="List of texts to compare")


class CompareBatchResponse(BaseModel):
    """Результат сравнения текстов с эталоном."""

    distances: List[float]


class KeywordsRequest(BaseModel):
    """Запрос на извлечение ключевых слов."""

    text: str = Field(..., min_length=1)
    ngram: Optional[int] = 2
    top_n: Optional[int] = 5


class KeywordsResponse(BaseModel):
    """Список ключевых слов."""

    keywords: List[str]
