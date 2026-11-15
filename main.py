import logging
from contextlib import asynccontextmanager

import psutil
import torch
from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from sentence_transformers import SentenceTransformer

from app.metrics import (
    gpu_utilization,
    process_cpu_seconds,
    process_memory,
    setup_metrics,
)
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
    setup_metrics()

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
    app.state.device = device

    FastAPIInstrumentor.instrument_app(app)

    import asyncio
    async def collect_system_metrics():
        process = psutil.Process()
        last_cpu_time = process.cpu_times()
        while True:
            await asyncio.sleep(5)
            try:
                current_cpu_time = process.cpu_times()
                cpu_delta = ((current_cpu_time.user - last_cpu_time.user) +
                             (current_cpu_time.system - last_cpu_time.system))
                process_cpu_seconds.add(cpu_delta)
                last_cpu_time = current_cpu_time

                mem_info = process.memory_info()
                process_memory.set(mem_info.rss)

                if device == 'cuda' and torch.cuda.is_available():
                    gpu_util = torch.cuda.utilization()
                    gpu_utilization.set(gpu_util)
            except Exception as e:
                logger.warning(f'Failed to collect system metrics: {e}')

    metrics_task = asyncio.create_task(collect_system_metrics())
    logger.info('System metrics collection started')

    yield

    metrics_task.cancel()
    task_manager.shutdown()
    logger.info('Application shutdown complete')


app = FastAPI(
    title='Semantic Similarity Service',
    version='3.0',
    lifespan=lifespan
)
app.include_router(router)


@app.get('/metrics')
async def metrics_endpoint():
    from fastapi.responses import Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


if __name__ == '__main__':
    import uvicorn
    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)
