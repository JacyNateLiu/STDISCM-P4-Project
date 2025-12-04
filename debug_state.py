"""Ad-hoc harness that exercises dashboard fault-tolerance guarantees."""

from __future__ import annotations

import asyncio
import statistics
from typing import List

from server import schemas
from server.state import DashboardState, REQUIRED_TILE_COUNT

SMALL_B64 = "aGVsbG93b3JsZGFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWE="


def _make_samples(tag: str, count: int) -> List[schemas.SampleEntry]:
	return [
		schemas.SampleEntry(
			sample_id=f"{tag}-{idx}",
			image_b64=SMALL_B64,
			prediction="Knight",
			ground_truth="Knight",
			confidence=1.0,
		)
		for idx in range(count)
	]


def _payload(iteration: int, sample_count: int, loss: float) -> schemas.BatchUpdateRequest:
	return schemas.BatchUpdateRequest(
		iteration=iteration,
		loss=loss,
		batch_size=sample_count,
		samples=_make_samples(f"iter{iteration}", sample_count),
	)


def _summarize(packet: schemas.DashboardPacket) -> dict[str, object]:
	return {
		"iteration": packet.iteration,
		"loss": round(packet.loss, 5),
		"tiles_ready": packet.tiles_ready,
		"samples_available": packet.samples_available,
		"tile_count": len(packet.tiles),
		"loss_points": len(packet.loss_history),
	}


async def scenario_duplicate_retry() -> list[dict[str, object]]:
	state = DashboardState()
	first = await state.ingest(_payload(iteration=1, sample_count=REQUIRED_TILE_COUNT, loss=0.42))
	duplicate = await state.ingest(_payload(iteration=1, sample_count=REQUIRED_TILE_COUNT, loss=9.99))
	older = await state.ingest(_payload(iteration=0, sample_count=REQUIRED_TILE_COUNT, loss=123.0))
	return [_summarize(first), _summarize(duplicate), _summarize(older)]


async def scenario_concurrent_partials() -> list[dict[str, object]]:
	state = DashboardState()

	async def push(iteration: int, sample_count: int, delay: float) -> schemas.DashboardPacket:
		await asyncio.sleep(delay)
		return await state.ingest(_payload(iteration, sample_count, loss=iteration / 10))

	results = await asyncio.gather(
		push(1, REQUIRED_TILE_COUNT // 2, 0.0),
		push(2, REQUIRED_TILE_COUNT // 2, 0.01),
		push(3, REQUIRED_TILE_COUNT, 0.02),
	)
	return [_summarize(packet) for packet in results]


async def scenario_loss_history_consistency() -> dict[str, object]:
	state = DashboardState()
	losses: list[float] = []
	for iteration in range(1, 6):
		loss = iteration / 100
		losses.append(loss)
		await state.ingest(_payload(iteration, REQUIRED_TILE_COUNT, loss))
	snapshot = await state.snapshot()
	series = [point.loss for point in snapshot.loss_history]
	return {
		**_summarize(snapshot),
		"loss_mean": round(statistics.mean(series), 5),
		"loss_min": round(min(series), 5),
		"loss_max": round(max(series), 5),
	}


async def run_fault_tolerance_demo() -> None:
	print("=== Scenario A: Duplicate retries/out-of-order packets are ignored ===")
	for label, summary in zip(
		["initial", "duplicate", "older"], await scenario_duplicate_retry(), strict=True
	):
		print(f"{label:>9}: {summary}")

	print("\n=== Scenario B: Concurrent writers keep the latest consistent state ===")
	for idx, summary in enumerate(await scenario_concurrent_partials(), start=1):
		print(f"task {idx}: {summary}")

	print("\n=== Scenario C: Loss history remains intact after multiple updates ===")
	print(await scenario_loss_history_consistency())


if __name__ == "__main__":
	asyncio.run(run_fault_tolerance_demo())
