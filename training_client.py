"""Utility script that trains a CNN on chess pieces and streams data to the dashboard."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import grpc
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from server.proto import dashboard_pb2, dashboard_pb2_grpc

GRPC_TARGET = os.getenv("DASHBOARD_GRPC_TARGET", "127.0.0.1:50051")
TOTAL_ITERATIONS = int(os.getenv("TOTAL_ITERATIONS", "400"))
DELAY_MS = int(os.getenv("INJECT_DELAY_MS", "0"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-3"))
STREAM_TILE_LIMIT = int(os.getenv("STREAM_TILE_LIMIT", "16"))
MAX_RPC_RETRIES = int(os.getenv("RPC_MAX_RETRIES", "5"))
BACKOFF_BASE_SECONDS = float(os.getenv("RPC_BACKOFF_BASE", "0.5"))
BACKOFF_MAX_SECONDS = float(os.getenv("RPC_BACKOFF_MAX", "5.0"))
BATCH_SIZE = 16
CLASSES = [
    "Bishop",
    "King",
    "Knight",
    "Pawn",
    "Queen",
    "Rook",
]

DATASET_ROOT = Path(__file__).parent / "assets"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".jfif", ".tif", ".tiff"}


@dataclass
class Sample:
    prediction: str
    ground_truth: str
    confidence: float
    image_b64: str


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return payload


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))  # CHW
    return torch.from_numpy(array)


class ChessPieceDataset(Dataset):
    def __init__(self, root: Path, classes: Sequence[str], image_size: int = 128) -> None:
        self.root = root
        self.classes = classes
        self.image_size = image_size
        self.samples: List[tuple[Path, int]] = []
        for idx, label in enumerate(classes):
            folder = root / label
            if not folder.exists():
                continue
            for path in folder.glob("*.*"):
                if path.suffix.lower() in ALLOWED_EXTENSIONS:
                    self.samples.append((path, idx))
        if not self.samples:
            raise RuntimeError(
                f"No dataset found under {root}. Make sure the chess assets are available."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label_idx = self.samples[index]
        base_image = Image.open(path).convert("RGB")

        # Preserve the original framing for display while still creating a square crop for training.
        display_image = ImageOps.contain(base_image.copy(), (self.image_size, self.image_size))
        training_image = ImageOps.fit(base_image, (self.image_size, self.image_size))
        if random.random() < 0.5:
            training_image = ImageOps.mirror(training_image)

        tensor = pil_to_tensor(training_image)
        return tensor, label_idx, display_image


def collate_batch(batch):
    tensors, labels, images = zip(*batch)
    stacked = torch.stack(list(tensors))
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return stacked, label_tensor, list(images)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class ChessTrainer:
    def __init__(self) -> None:
        batch_size = BATCH_SIZE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = ChessPieceDataset(DATASET_ROOT, CLASSES)
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            num_workers=0,
            drop_last=True,
        )
        self.iterator = iter(self.loader)
        self.model = SimpleCNN(len(CLASSES)).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def _next_batch(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)

    def train_iteration(self):
        inputs, labels, images = self._next_batch()
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        self.model.train()
        logits = self.model(inputs)
        loss = self.criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        probs = torch.softmax(logits.detach().cpu(), dim=1)
        preds = probs.argmax(dim=1)
        confidences = probs.max(dim=1).values
        return loss.item(), labels.cpu(), preds, confidences, images


def build_samples(
    labels: torch.Tensor,
    preds: torch.Tensor,
    confidences: torch.Tensor,
    images: Sequence[Image.Image],
) -> List[Sample]:
    total = len(images)
    indices = list(range(total))
    random.shuffle(indices)
    limit = min(STREAM_TILE_LIMIT, total)
    selected = indices[:limit]
    samples: List[Sample] = []
    for idx in selected:
        label_idx = int(labels[idx])
        pred_idx = int(preds[idx])
        confidence = float(confidences[idx])
        image = images[idx]
        samples.append(
            Sample(
                prediction=CLASSES[pred_idx],
                ground_truth=CLASSES[label_idx],
                confidence=confidence,
                image_b64=image_to_base64(image),
            )
        )
    return samples


async def generate_batches():
    trainer = ChessTrainer()
    for iteration in range(1, TOTAL_ITERATIONS + 1):
        loss_value, labels, preds, confidences, images = trainer.train_iteration()
        samples = build_samples(labels, preds, confidences, images)
        yield iteration, len(images), loss_value, samples
        await asyncio.sleep(0)


async def stream_batches() -> None:
    channel = grpc.aio.insecure_channel(GRPC_TARGET)
    stub = dashboard_pb2_grpc.TrainingDashboardStub(channel)
    last_acked_iteration = 0
    try:
        async for iteration, batch_size, loss_value, samples in generate_batches():
            if iteration <= last_acked_iteration:
                continue

            request = dashboard_pb2.BatchUpdateRequest(
                iteration=iteration,
                loss=loss_value,
                batch_size=batch_size,
                samples=[
                    dashboard_pb2.SampleEntry(
                        sample_id=f"{iteration}-{idx}",
                        prediction=sample.prediction,
                        ground_truth=sample.ground_truth,
                        confidence=sample.confidence,
                        image_b64=sample.image_b64,
                    )
                    for idx, sample in enumerate(samples)
                ],
                delay_ms=DELAY_MS,
            )

            response, stub, channel = await send_with_retry(
                request, stub, channel
            )
            last_acked_iteration = max(last_acked_iteration, response.iteration)

            print(
                f"iter={iteration:04d} batch={batch_size:02d} "
                f"loss={loss_value:.4f} tilesReady={response.tiles_ready}"
            )
    finally:
        await channel.close()


async def send_with_retry(request, stub, channel):
    attempt = 0
    delay = BACKOFF_BASE_SECONDS
    while True:
        attempt += 1
        try:
            response = await stub.BatchUpdate(request)
            return response, stub, channel
        except grpc.aio.AioRpcError as exc:
            error_details = exc.details() or "unknown"
            error_message = f"{exc.code().name}: {error_details}"
        except Exception as exc:  # pragma: no cover - defensive
            error_message = f"unexpected error: {exc}"

        if attempt >= MAX_RPC_RETRIES:
            raise RuntimeError(
                f"Exhausted retries sending iteration {request.iteration}: {error_message}"
            )

        print(
            "Retrying iteration "
            f"{request.iteration} (attempt {attempt}/{MAX_RPC_RETRIES}) after error: "
            f"{error_message}"
        )

        try:
            await channel.close()
        except Exception:
            pass

        jitter = delay * 0.2
        await asyncio.sleep(delay + random.uniform(0, jitter))
        delay = min(delay * 2, BACKOFF_MAX_SECONDS)

        channel = grpc.aio.insecure_channel(GRPC_TARGET)
        stub = dashboard_pb2_grpc.TrainingDashboardStub(channel)


if __name__ == "__main__":
    try:
        asyncio.run(stream_batches())
    except KeyboardInterrupt:
        print("Stopped.")
