import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from app.router import router
from app.settings import get_settings
from app.task_manager import TaskManager

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    device = settings.device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    logger.info(f'Loading model {settings.model_name} on {device}...')
    model = SentenceTransformer(settings.model_name, device=device)
    logger.info('Model loaded successfully')

    task_manager = TaskManager(
        max_workers=settings.worker_threads,
        task_ttl_minutes=settings.task_ttl_minutes
    )
    logger.info(f'Task manager initialized with {settings.worker_threads} workers')

    app.state.model = model
    app.state.task_manager = task_manager
    app.state.settings = settings

    yield

    task_manager.shutdown()
    logger.info('Application shutdown complete')


app = FastAPI(
    title='Semantic Similarity Service',
    version='3.0',
    lifespan=lifespan
)
app.include_router(router)

if __name__ == '__main__':
    import uvicorn
    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)
