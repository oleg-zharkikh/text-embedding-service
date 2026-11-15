import logging
import time

from fastapi import APIRouter, HTTPException, Request
from sklearn.metrics.pairwise import cosine_similarity

from app.metrics import (
    embedding_distance_matrix_size,
    embedding_text_length,
    model_embeddings_generated,
    model_inference_duration,
)
from app.schemas import (
    A2ARequest,
    A2AResponse,
    A2AResultResponse,
    A2AStatusResponse,
    CompareBatchRequest,
    CompareBatchResponse,
    CompareRequest,
    CompareResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    KeywordsRequest,
    KeywordsResponse,
)
from app.task_manager import TaskStatus
from app.utils import serialize_array

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post('/api/embedding/', response_model=EmbeddingResponse)
async def embedding_endpoint(request: EmbeddingRequest, req: Request):
    """Возвращает эмбеддинг текста в формате base64."""
    try:
        model = req.app.state.model
        embedding_text_length.record(len(request.text))

        start_time = time.time()
        embedding = model.encode(request.text, convert_to_numpy=True)
        duration = time.time() - start_time

        model_inference_duration.record(duration, {'endpoint': 'embedding'})
        model_embeddings_generated.add(1)

        return EmbeddingResponse(embedding=serialize_array(embedding))
    except Exception as e:
        logger.exception(f'Embedding error: {e}')
        raise HTTPException(
            status_code=500, detail=f'Embedding failed: {e!s}')


@router.post('/api/compare/', response_model=CompareResponse)
async def compare_endpoint(request: CompareRequest, req: Request):
    """Сравнивает два текста, возвращает косинусное расстояние.

    Чем значение ближе к 0 тем ближе семантически.
    """
    try:
        model = req.app.state.model
        embedding_text_length.record(len(request.etalon))
        embedding_text_length.record(len(request.text))

        start_time = time.time()
        emb_etalon = model.encode(
            request.etalon, convert_to_numpy=True).reshape(1, -1)
        emb_text = model.encode(
            request.text, convert_to_numpy=True).reshape(1, -1)
        cosine_sim = cosine_similarity(emb_etalon, emb_text)[0][0]
        distance = 1.0 - cosine_sim
        duration = time.time() - start_time

        model_inference_duration.record(duration, {'endpoint': 'compare'})
        model_embeddings_generated.add(2)

        return CompareResponse(distance=float(distance))
    except Exception as e:
        logger.exception(f'Compare error: {e}')
        raise HTTPException(
            status_code=500, detail=f'Comparison failed: {e!s}')


@router.post('/api/compare_batch/', response_model=CompareBatchResponse)
async def compare_batch_endpoint(request: CompareBatchRequest, req: Request):
    """Пакетное сравнение эталонного текста со списком текстов.

    Возвращает список расстояний в том же порядке, что и входные тексты.
    """
    try:
        model = req.app.state.model
        embedding_text_length.record(len(request.etalon))
        for text in request.texts:
            embedding_text_length.record(len(text))

        start_time = time.time()
        emb_etalon = model.encode(
            request.etalon, convert_to_numpy=True).reshape(1, -1)

        embs_texts = model.encode(request.texts, convert_to_numpy=True)

        cosine_sims = cosine_similarity(emb_etalon, embs_texts)[0]
        distances = (1.0 - cosine_sims).tolist()
        duration = time.time() - start_time

        model_inference_duration.record(
            duration, {'endpoint': 'compare_batch'})
        model_embeddings_generated.add(1 + len(request.texts))

        return CompareBatchResponse(distances=distances)
    except Exception as e:
        logger.exception(f'Batch compare error: {e}')
        raise HTTPException(
            status_code=500, detail=f'Batch comparison failed: {e!s}')


@router.get('/health')
async def health():
    return {'status': 'ok'}


@router.post('/a2a', response_model=A2AResponse)
async def create_a2a_task(request: A2ARequest, req: Request):
    """Создает задачу вычисления матрицы расстояний между текстами."""
    try:
        settings = req.app.state.settings
        task_manager = req.app.state.task_manager
        model = req.app.state.model

        for text in request.texts:
            embedding_text_length.record(len(text))
        embedding_distance_matrix_size.record(len(request.texts) ** 2)

        task_id = task_manager.submit_task(
            texts=request.texts,
            model=model,
            batch_size=settings.batch_size,
            normalize_embeddings=settings.normalize_embeddings
        )

        return A2AResponse(task_id=task_id)
    except Exception as e:
        logger.exception(f'A2A task creation error: {e}')
        raise HTTPException(
            status_code=500, detail=f'Failed to create task: {e!s}')


@router.get('/a2a/{task_id}/status', response_model=A2AStatusResponse)
async def get_a2a_task_status(task_id: str, req: Request):
    """Возвращает статус фоновой задачи."""
    task_manager = req.app.state.task_manager
    status = task_manager.get_task_status(task_id)
    error = (task_manager.get_task_error(task_id)
             if status == TaskStatus.FAILED
             else None)

    return A2AStatusResponse(status=status.value, error=error)


@router.get('/a2a/{task_id}/result', response_model=A2AResultResponse)
async def get_a2a_task_result(task_id: str, req: Request):
    """Возвращает результат вычисления матрицы расстояний."""
    task_manager = req.app.state.task_manager
    status = task_manager.get_task_status(task_id)

    if status == TaskStatus.NOT_FOUND:
        raise HTTPException(status_code=404, detail='Task not found')

    if status == TaskStatus.IN_PROCESS:
        return A2AResultResponse(
            task_id=task_id,
            status=status.value,
            distance_matrix=None
        )

    if status == TaskStatus.FAILED:
        error = task_manager.get_task_error(task_id)
        raise HTTPException(
            status_code=500,
            detail=f'Task failed: {error}'
        )

    result = task_manager.get_task_result(task_id)
    if result is None:
        raise HTTPException(status_code=500, detail='Result not available')

    return A2AResultResponse(
        task_id=task_id,
        status=status.value,
        distance_matrix=serialize_array(result)
    )
