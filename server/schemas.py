from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class SampleEntry(BaseModel):
    """Represents a single training example's visualization payload."""

    sample_id: Optional[str] = Field(
        default=None,
        description="Caller-provided identifier that helps deduplicate retries.",
    )
    image_b64: str = Field(
        description="Base64-encoded image data (PNG or JPEG).",
        min_length=32,
    )
    prediction: str = Field(description="Model prediction for the sample.")
    ground_truth: str = Field(description="Ground-truth label for the sample.")
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional probability/confidence metric in [0, 1].",
    )


class BatchUpdateRequest(BaseModel):
    iteration: int = Field(ge=0, description="Global iteration number.")
    loss: float = Field(description="Training loss value for the iteration.")
    batch_size: int = Field(ge=1, le=2048)
    samples: List[SampleEntry] = Field(..., min_items=1, max_items=128)
    timestamp: Optional[datetime] = None
    took_ms: Optional[float] = Field(
        default=None,
        description="Processing latency on the trainer side in milliseconds.",
    )
    delay_ms: Optional[int] = Field(
        default=None,
        ge=0,
        le=10000,
        description="Optional artificial delay to help test concurrency.",
    )

    @validator("samples")
    def _enforce_image_limits(cls, value: List[SampleEntry]):
        for sample in value:
            if len(sample.image_b64) > 512 * 512 * 4:
                raise ValueError("image payload appears to exceed 512x512 limit")
        return value


class LossPoint(BaseModel):
    iteration: int
    loss: float
    timestamp: datetime


class TilePayload(BaseModel):
    sample_id: Optional[str]
    image_b64: str
    prediction: str
    ground_truth: str
    confidence: Optional[float]


class DashboardPacket(BaseModel):
    iteration: int
    loss: float
    batch_size: int
    tiles_ready: bool
    samples_available: int
    tiles: List[TilePayload]
    loss_history: List[LossPoint]
    generated_at: datetime
