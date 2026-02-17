from fastapi import APIRouter
from app.schemas import (
    EmbeddingRequest, EmbeddingResponse,
    CompareRequest, CompareResponse,
    CompareBatchRequest, CompareBatchResponse,
    KeywordsRequest, KeywordsResponse
)
from app.utils import serialize_array

import logging

from fastapi import HTTPException
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.constants import LOG_FILE, MODEL_PATH


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

model = SentenceTransformer(MODEL_PATH, local_files_only=True)
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

logger.info('Sentence transformer model loaded successfully.')

router = APIRouter()


@router.post('/api/embedding/', response_model=EmbeddingResponse)
async def embedding_endpoint(request: EmbeddingRequest):
    """Возвращает эмбеддинг текста в формате base64."""
    try:
        embedding = model.encode(request.text, convert_to_numpy=True)
        return EmbeddingResponse(embedding=serialize_array(embedding))
    except Exception as e:
        logger.error(f'Embedding error: {e}', exc_info=True)
        raise HTTPException(
            status_code=500, detail=f'Embedding failed: {str(e)}')


@router.post('/api/compare/', response_model=CompareResponse)
async def compare_endpoint(request: CompareRequest):
    """Сравнивает два текста, возвращает косинусное расстояние.

    Чем значение ближе к 0 тем ближе семантически.
    """
    try:
        emb_etalon = model.encode(
            request.etalon, convert_to_numpy=True).reshape(1, -1)
        emb_text = model.encode(
            request.text, convert_to_numpy=True).reshape(1, -1)
        cosine_sim = cosine_similarity(emb_etalon, emb_text)[0][0]
        distance = 1.0 - cosine_sim
        return CompareResponse(distance=float(distance))
    except Exception as e:
        logger.error(f'Compare error: {e}', exc_info=True)
        raise HTTPException(
            status_code=500, detail=f'Comparison failed: {str(e)}')


@router.post('/api/compare_batch/', response_model=CompareBatchResponse)
async def compare_batch_endpoint(request: CompareBatchRequest):
    """Пакетное сравнение эталонного текста со списком текстов.

    Возвращает список расстояний в том же порядке, что и входные тексты.
    """
    try:
        emb_etalon = model.encode(
            request.etalon, convert_to_numpy=True).reshape(1, -1)

        embs_texts = model.encode(request.texts, convert_to_numpy=True)

        cosine_sims = cosine_similarity(emb_etalon, embs_texts)[0]
        distances = (1.0 - cosine_sims).tolist()
        return CompareBatchResponse(distances=distances)
    except Exception as e:
        logger.error(f'Batch compare error: {e}', exc_info=True)
        raise HTTPException(
            status_code=500, detail=f'Batch comparison failed: {str(e)}')


@router.post(
    '/api/keywords/',
    response_model=KeywordsResponse,
    status_code=501
)
async def keywords_endpoint(request: KeywordsRequest):
    """Эндпоинт извлечения ключевых слов (не реализован для текущей модели)."""
    raise HTTPException(
        status_code=501,
        detail="Keywords extraction is not supported by the current model."
    )


@router.get('/health')
async def health():
    return {'status': 'ok'}
