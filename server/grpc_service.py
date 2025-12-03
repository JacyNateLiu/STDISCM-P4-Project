from __future__ import annotations

import asyncio
import os
from concurrent import futures

import grpc
from pydantic import ValidationError

from . import schemas
from .broadcast import BroadcastHub
from .proto import dashboard_pb2, dashboard_pb2_grpc
from .state import DashboardState, maybe_delay

GRPC_BIND_ADDRESS = os.getenv("DASHBOARD_GRPC_ADDRESS", "0.0.0.0:50051")
MAX_MESSAGE_BYTES = int(os.getenv("DASHBOARD_GRPC_MAX_MESSAGE", str(32 * 1024 * 1024)))


class DashboardGrpcService(dashboard_pb2_grpc.TrainingDashboardServicer):
    """Bridges gRPC ingestion calls into the shared dashboard state."""

    def __init__(
        self,
        state: DashboardState,
        hub: BroadcastHub,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._state = state
        self._hub = hub
        self._loop = loop

    def BatchUpdate(self, request, context):  # type: ignore[override]
        try:
            payload = schemas.BatchUpdateRequest(
                iteration=request.iteration,
                loss=request.loss,
                batch_size=request.batch_size,
                samples=[
                    schemas.SampleEntry(
                        sample_id=sample.sample_id or None,
                        image_b64=sample.image_b64,
                        prediction=sample.prediction,
                        ground_truth=sample.ground_truth,
                        confidence=sample.confidence,
                    )
                    for sample in request.samples
                ],
                took_ms=request.took_ms or None,
                delay_ms=request.delay_ms or None,
            )
        except ValidationError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        future = asyncio.run_coroutine_threadsafe(
            self._process_payload(payload), self._loop
        )
        try:
            return future.result()
        except ValidationError as exc:  # pragma: no cover - defensive
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    async def _process_payload(
        self, payload: schemas.BatchUpdateRequest
    ) -> dashboard_pb2.BatchUpdateResponse:
        await maybe_delay(payload.delay_ms)
        packet = await self._state.ingest(payload)
        await self._hub.broadcast(packet.json())
        return dashboard_pb2.BatchUpdateResponse(
            status="ok",
            iteration=packet.iteration,
            loss=packet.loss,
            tiles_ready=packet.tiles_ready,
            samples_available=packet.samples_available,
        )


def build_grpc_server(
    state: DashboardState,
    hub: BroadcastHub,
    loop: asyncio.AbstractEventLoop,
) -> grpc.Server:
    options = [
        ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
        ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
    ]
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4),
        options=options,
    )
    dashboard_pb2_grpc.add_TrainingDashboardServicer_to_server(
        DashboardGrpcService(state, hub, loop), server
    )
    server.add_insecure_port(GRPC_BIND_ADDRESS)
    return server


async def start_grpc_server(state: DashboardState, hub: BroadcastHub) -> grpc.Server:
    loop = asyncio.get_running_loop()
    server = build_grpc_server(state, hub, loop)
    server.start()
    return server


async def stop_grpc_server(server: grpc.Server) -> None:
    stop_future = server.stop(0)
    await asyncio.to_thread(stop_future.result)


def grpc_server_address() -> str:
    return GRPC_BIND_ADDRESS
