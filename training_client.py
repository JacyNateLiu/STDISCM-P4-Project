"""Utility script that simulates a CNN training loop and streams data to the dashboard."""

from __future__ import annotations

import asyncio
import base64
import io
import math
import os
import random
from dataclasses import dataclass
from typing import List

import httpx
import numpy as np
from PIL import Image

RPC_URL = os.getenv("DASHBOARD_RPC_URL", "http://127.0.0.1:8000/rpc/batch_update")
TOTAL_ITERATIONS = int(os.getenv("TOTAL_ITERATIONS", "400"))
MINI_BATCH_MIN = int(os.getenv("MINI_BATCH_MIN", "6"))
MINI_BATCH_MAX = int(os.getenv("MINI_BATCH_MAX", "32"))
DELAY_MS = int(os.getenv("INJECT_DELAY_MS", "0"))
CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@dataclass
class Sample:
    prediction: str
    ground_truth: str
    confidence: float
    image_b64: str


def generate_image(side: int = 64) -> Image.Image:
    noise = np.random.randint(0, 255, (side, side, 3), dtype=np.uint8)
    image = Image.fromarray(noise, mode="RGB")
    if side < 64:
        image = image.resize((side * 2, side * 2), Image.NEAREST)
    return image


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return payload


def build_sample() -> Sample:
    gt = random.choice(CLASSES)
    predicted = random.choice(CLASSES)
    confidence = max(0.05, random.random())
    image_side = random.choice([32, 64, 96, 128, 224])
    image = generate_image(image_side)
    return Sample(
        prediction=predicted,
        ground_truth=gt,
        confidence=confidence,
        image_b64=image_to_base64(image),
    )


def compute_loss(iteration: int) -> float:
    baseline = math.exp(-iteration / 250) * 2.0
    noise = random.uniform(-0.05, 0.05)
    return max(0.01, baseline + noise)


async def stream_batches() -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        for iteration in range(1, TOTAL_ITERATIONS + 1):
            batch_size = random.randint(MINI_BATCH_MIN, MINI_BATCH_MAX)
            samples = [build_sample() for _ in range(batch_size)]
            payload = {
                "iteration": iteration,
                "loss": compute_loss(iteration),
                "batch_size": batch_size,
                "samples": [
                    {
                        "sample_id": f"{iteration}-{idx}",
                        "prediction": sample.prediction,
                        "ground_truth": sample.ground_truth,
                        "confidence": sample.confidence,
                        "image_b64": sample.image_b64,
                    }
                    for idx, sample in enumerate(samples)
                ],
                "delay_ms": DELAY_MS,
            }
            response = await client.post(RPC_URL, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                body = exc.response.text
                print(
                    f"Server returned {exc.response.status_code}: {body[:400]}"
                )
                raise
            status = response.json()
            print(
                f"iter={iteration:04d} batch={batch_size:02d} "
                f"loss={payload['loss']:.4f} tilesReady={status.get('tilesReady')}"
            )
            await asyncio.sleep(0.2)


if __name__ == "__main__":
    try:
        asyncio.run(stream_batches())
    except KeyboardInterrupt:
        print("Stopped.")
