"""IngestionWorker — RabbitMQ consumer that processes document ingestion tasks.

Can be run as a standalone process:
  python -m insight_miner.core.ingestion.worker
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import aio_pika

from insight_miner.config import RABBITMQ_DLQ, RABBITMQ_QUEUE, RABBITMQ_URL
from insight_miner.core.document_processor import KnowledgeBaseIndex
from insight_miner.core.ingestion.pipeline import IngestionPipeline
from insight_miner.core.ingestion.task_tracker import TaskTracker

logger = logging.getLogger(__name__)


class IngestionWorker:
    """Consumes documents from RabbitMQ and processes them asynchronously."""

    def __init__(
        self,
        concurrency: int = 4,
        amqp_url: str = "",
        queue_name: str = "",
        dlq_name: str = "",
    ) -> None:
        self._concurrency = concurrency
        self._amqp_url = amqp_url or RABBITMQ_URL
        self._queue_name = queue_name or RABBITMQ_QUEUE
        self._dlq_name = dlq_name or RABBITMQ_DLQ
        self._connection: aio_pika.Connection | None = None
        self._tracker = TaskTracker()
        self._semaphore = asyncio.Semaphore(concurrency)

    async def start(self) -> None:
        """Connect to RabbitMQ and start consuming."""
        self._connection = await aio_pika.connect_robust(self._amqp_url)
        channel = await self._connection.channel()
        await channel.set_qos(prefetch_count=self._concurrency)

        # Declare main queue + dead-letter queue
        main_queue = await channel.declare_queue(
            self._queue_name,
            durable=True,
            arguments={
                "x-dead-letter-exchange": "",
                "x-dead-letter-routing-key": self._dlq_name,
            },
        )
        await channel.declare_queue(self._dlq_name, durable=True)

        logger.info(
            "worker started queue=%s dlq=%s concurrency=%d",
            self._queue_name, self._dlq_name, self._concurrency,
        )

        async with main_queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process(ignore_processed=True):
                    await self._semaphore.acquire()
                    asyncio.ensure_future(self._process_message(message))

    async def _process_message(self, message: aio_pika.IncomingMessage) -> None:
        try:
            body = json.loads(message.body.decode())
            task_id = body.get("task_id", "")
            kb_id = body.get("kb_id", "default")
            filename = body.get("filename", "")

            logger.info("worker processing task=%s kb=%s file=%s", task_id, kb_id, filename)

            # Update task status
            await self._tracker.update_status(task_id, "processing")

            # Load KB index (load existing, skip auto-detect since pipeline handles new file)
            kb_index = KnowledgeBaseIndex(kb_id)
            kb_index.load_models()
            kb_index.ensure_dirs()
            kb_index._load_existing_chroma()
            kb_index._load_bm25()
            kb_index._load_graph()

            pipeline = IngestionPipeline(kb_index)
            success = await pipeline.run(filename)

            if success:
                await self._tracker.update_status(task_id, "done")
                logger.info("worker done task=%s kb=%s file=%s", task_id, kb_id, filename)
            else:
                await self._tracker.update_status(task_id, "failed", error="Pipeline execution failed")
                logger.error("worker failed task=%s kb=%s file=%s", task_id, kb_id, filename)

        except Exception as e:
            logger.exception("worker exception: %s", e)
            try:
                body = json.loads(message.body.decode())
                await self._tracker.update_status(body.get("task_id", ""), "failed", error=str(e))
            except Exception:
                pass
        finally:
            self._semaphore.release()

    async def stop(self) -> None:
        if self._connection:
            await self._connection.close()
            logger.info("worker stopped")


# ── Standalone runner ──

async def _run_worker():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    worker = IngestionWorker()
    try:
        await worker.start()
    except KeyboardInterrupt:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(_run_worker())
