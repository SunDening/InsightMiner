"""MessagePublisher — RabbitMQ publisher for async task dispatch."""

from __future__ import annotations

import json
import logging

import aio_pika

from insight_miner.config import RABBITMQ_QUEUE, RABBITMQ_URL

logger = logging.getLogger(__name__)


class MessagePublisher:
    """Publish messages to RabbitMQ. Reuses a single connection."""

    def __init__(self) -> None:
        self._connection: aio_pika.Connection | None = None
        self._channel: aio_pika.Channel | None = None

    async def publish(self, routing_key: str, body: dict) -> bool:
        try:
            if self._connection is None or self._connection.is_closed:
                self._connection = await aio_pika.connect_robust(RABBITMQ_URL)
                self._channel = await self._connection.channel()

            await self._channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(body, ensure_ascii=False).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key=routing_key,
            )
            return True
        except Exception as e:
            logger.error("publish failed routing_key=%s error=%s", routing_key, e)
            return False

    async def close(self) -> None:
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
