from typing import Any

from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Входной текст для векторизации."""

    text: str = Field(..., min_length=1, description='Input text')


class EmbeddingResponse(BaseModel):
    """Эмбеддинг."""

    embedding: dict[str, Any]


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
    texts: list[str] = Field(
        ..., min_items=1, description="List of texts to compare")


class CompareBatchResponse(BaseModel):
    """Результат сравнения текстов с эталоном."""

    distances: list[float]


class KeywordsRequest(BaseModel):
    """Запрос на извлечение ключевых слов."""

    text: str = Field(..., min_length=1)
    ngram: int | None = 2
    top_n: int | None = 5


class KeywordsResponse(BaseModel):
    """Список ключевых слов."""

    keywords: list[str]


class A2ARequest(BaseModel):
    """Запрос на асинхронное вычисление матрицы расстояний."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description='List of texts to compute distance matrix'
    )


class A2AResponse(BaseModel):
    """Ответ с ID фоновой задачи."""

    task_id: str = Field(..., description='Unique task identifier')


class A2AStatusResponse(BaseModel):
    """Статус фоновой задачи."""

    status: str = Field(
        ...,
        description='Task status: in_process, completed, failed, not_found'
    )
    error: str | None = Field(
        None,
        description='Error message if task failed'
    )


class A2AResultResponse(BaseModel):
    """Результат вычисления матрицы расстояний."""

    task_id: str = Field(..., description='Task identifier')
    status: str = Field(..., description='Task status')
    distance_matrix: dict[str, Any] | None = Field(
        None,
        description='Distance matrix as base64 encoded numpy array'
    )
