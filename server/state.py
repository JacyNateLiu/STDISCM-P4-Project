from __future__ import annotations

import asyncio
import os
import random
from collections import deque
from datetime import datetime
from typing import Deque, List, Optional

from . import schemas

MAX_HISTORY = 1024
REQUIRED_TILE_COUNT = 16


class DashboardState:
    """Thread-safe state container for the dashboard."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._loss_history: Deque[schemas.LossPoint] = deque(maxlen=MAX_HISTORY)
        self._latest_tiles: List[schemas.TilePayload] = []
        self._pending_samples: Deque[schemas.TilePayload] = deque()
        self._latest_iteration: Optional[int] = None
        self._latest_batch_size: int = 0
        self._last_tiles_ready: bool = False
        self._samples_available_hint: int = 0

    async def ingest(self, payload: schemas.BatchUpdateRequest) -> schemas.DashboardPacket:
        async with self._lock:
            generated_at = datetime.utcnow()
            # track duplicate iterations to offer idempotency for RPC retries
            if (
                self._latest_iteration is not None
                and payload.iteration <= self._latest_iteration
            ):
                # older iterations are ignored but we still echo the current packet
                return self._snapshot(generated_at)

            loss_point = schemas.LossPoint(
                iteration=payload.iteration,
                loss=payload.loss,
                timestamp=generated_at,
            )
            self._loss_history.append(loss_point)
            self._latest_iteration = payload.iteration
            self._latest_batch_size = payload.batch_size

            tile_candidates = [
                schemas.TilePayload(
                    sample_id=sample.sample_id,
                    image_b64=sample.image_b64,
                    prediction=sample.prediction,
                    ground_truth=sample.ground_truth,
                    confidence=sample.confidence,
                )
                for sample in payload.samples
            ]

            tiles_ready = False

            if len(tile_candidates) >= REQUIRED_TILE_COUNT:
                self._latest_tiles = random.sample(tile_candidates, REQUIRED_TILE_COUNT)
                self._pending_samples.clear()
                tiles_ready = True
            else:
                for tile in tile_candidates:
                    self._pending_samples.append(tile)
                if len(self._pending_samples) >= REQUIRED_TILE_COUNT:
                    # Keep the latest REQUIRED_TILE_COUNT aggregated samples
                    while len(self._pending_samples) > REQUIRED_TILE_COUNT:
                        self._pending_samples.popleft()
                    self._latest_tiles = list(self._pending_samples)
                    tiles_ready = True
                else:
                    tiles_ready = False

            samples_available = (
                len(self._pending_samples)
                if not tiles_ready
                else len(self._latest_tiles)
            )

            self._last_tiles_ready = tiles_ready
            self._samples_available_hint = samples_available

            packet = self._snapshot(generated_at)
            return packet.copy(update={
                "tiles_ready": tiles_ready,
                "samples_available": samples_available,
            })

    async def snapshot(self) -> schemas.DashboardPacket:
        async with self._lock:
            return self._snapshot(datetime.utcnow())

    def _snapshot(self, generated_at: datetime) -> schemas.DashboardPacket:
        return schemas.DashboardPacket(
            iteration=self._latest_iteration or 0,
            loss=self._loss_history[-1].loss if self._loss_history else 0.0,
            batch_size=self._latest_batch_size,
            tiles_ready=self._last_tiles_ready,
            samples_available=self._samples_available_hint,
            tiles=list(self._latest_tiles),
            loss_history=list(self._loss_history),
            generated_at=generated_at,
        )


FRAME_DELAY_MS = int(os.getenv("FRAME_DELAY_MS", "0"))


async def maybe_delay(custom_delay: Optional[int]) -> None:
    delay_ms = custom_delay if custom_delay is not None else FRAME_DELAY_MS
    if delay_ms > 0:
        await asyncio.sleep(delay_ms / 1000)
