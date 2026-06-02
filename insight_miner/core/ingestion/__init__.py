"""Async document ingestion powered by RabbitMQ.

Flow:
  upload → publish to RabbitMQ → worker picks up → parse → chunk → embed → index → done
"""

from insight_miner.core.ingestion.pipeline import IngestionPipeline
from insight_miner.core.ingestion.task_tracker import TaskTracker
from insight_miner.core.ingestion.worker import IngestionWorker

__all__ = ["IngestionPipeline", "TaskTracker", "IngestionWorker"]
