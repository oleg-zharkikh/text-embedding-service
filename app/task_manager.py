import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    IN_PROCESS = 'in_process'
    COMPLETED = 'completed'
    FAILED = 'failed'
    NOT_FOUND = 'not_found'


class Task:
    __slots__ = [
        'completed_at',
        'created_at',
        'error',
        'result',
        'status',
        'task_id'
    ]

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.status = TaskStatus.IN_PROCESS
        self.result: np.ndarray | None = None
        self.error: str | None = None
        self.created_at = datetime.utcnow()
        self.completed_at: datetime | None = None


class TaskManager:
    def __init__(self, max_workers: int, task_ttl_minutes: int):
        self._tasks: dict[str, Task] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._ttl = timedelta(minutes=task_ttl_minutes)

    def _cleanup_old_tasks(self):
        cutoff = datetime.utcnow() - self._ttl
        with self._lock:
            old_task_ids = [
                task_id for task_id, task in self._tasks.items()
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
                and task.completed_at
                and task.completed_at < cutoff
            ]
            for task_id in old_task_ids:
                del self._tasks[task_id]
                logger.debug(f'Cleaned up old task: {task_id}')

    def submit_task(
        self,
        texts: list[str],
        model: SentenceTransformer,
        batch_size: int,
        normalize_embeddings: bool
    ) -> str:
        self._cleanup_old_tasks()

        task_id = str(uuid.uuid4())
        task = Task(task_id)

        with self._lock:
            self._tasks[task_id] = task

        self._executor.submit(
            self._process_task,
            task,
            texts,
            model,
            batch_size,
            normalize_embeddings
        )

        logger.info(f'Task submitted: {task_id} with {len(texts)} texts')
        return task_id

    def _process_task(
        self,
        task: Task,
        texts: list[str],
        model: SentenceTransformer,
        batch_size: int,
        normalize_embeddings: bool
    ):
        try:
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=False
            )

            cosine_sim_matrix = cosine_similarity(embeddings)
            distance_matrix = 1.0 - cosine_sim_matrix

            task.result = distance_matrix
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()

            logger.info(f'Task completed: {task.task_id}')

        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            logger.error(f'Task failed: {task.task_id}, error: {e}', exc_info=True)

    def get_task_status(self, task_id: str) -> TaskStatus:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return TaskStatus.NOT_FOUND
            return task.status

    def get_task_result(self, task_id: str) -> np.ndarray | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != TaskStatus.COMPLETED:
                return None
            return task.result

    def get_task_error(self, task_id: str) -> str | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            return task.error

    def shutdown(self):
        self._executor.shutdown(wait=True)
        logger.info('Task manager shut down')
