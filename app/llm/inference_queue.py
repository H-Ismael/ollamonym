"""Inference queue for bounded concurrency."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List

logger = logging.getLogger(__name__)


class InferenceQueue:
    """Bounded concurrency queue for LLM inference tasks."""

    def __init__(self, max_concurrency: int = 1):
        """
        Initialize inference queue.

        Args:
            max_concurrency: Maximum concurrent LLM calls
        """
        self.max_concurrency = max_concurrency
        self.executor = ThreadPoolExecutor(max_workers=max_concurrency)
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def submit(self, task: Callable[[], Any]) -> Any:
        """
        Submit a task to the queue.

        Args:
            task: Callable that performs the LLM inference

        Returns:
            Result of task
        """
        async with self.semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.executor, task)

    async def run_batch(self, tasks: List[Callable[[], Any]]) -> List[Any]:
        """Run a batch of tasks with bounded concurrency, preserving order."""
        if not tasks:
            return []

        coros = [self.submit(task) for task in tasks]
        return await asyncio.gather(*coros)

    def submit_sync(self, task: Callable[[], Any]) -> Any:
        """Synchronous submission (blocking)."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.submit(task))
        finally:
            loop.close()

    def run_batch_sync(self, tasks: List[Callable[[], Any]]) -> List[Any]:
        """Synchronous batch submission (blocking)."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.run_batch(tasks))
        finally:
            loop.close()

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)
