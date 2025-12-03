from __future__ import annotations

from pathlib import Path

import logging

import grpc
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .broadcast import BroadcastHub
from .grpc_service import grpc_server_address, start_grpc_server, stop_grpc_server
from .schemas import BatchUpdateRequest
from .state import DashboardState, maybe_delay

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Training Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state = DashboardState()
hub = BroadcastHub()
grpc_server: grpc.Server | None = None

app.mount("/dashboard", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/dashboard/")


@app.post("/rpc/batch_update")
async def batch_update(payload: BatchUpdateRequest) -> dict[str, object]:
    await maybe_delay(payload.delay_ms)
    packet = await state.ingest(payload)
    await hub.broadcast(packet.json())
    return {
        "status": "ok",
        "tilesReady": packet.tiles_ready,
        "loss": packet.loss,
        "iteration": packet.iteration,
    }


@app.websocket("/ws")
async def websocket_stream(websocket: WebSocket) -> None:
    client_id = await hub.register(websocket)
    try:
        snapshot = await state.snapshot()
        await websocket.send_text(snapshot.json())
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await hub.unregister(client_id)


@app.on_event("startup")
async def _start_grpc_server() -> None:
    global grpc_server
    if grpc_server is None:
        grpc_server = await start_grpc_server(state, hub)
        logging.getLogger(__name__).info(
            "gRPC ingestion server listening on %s", grpc_server_address()
        )


@app.on_event("shutdown")
async def _stop_grpc_server() -> None:
    global grpc_server
    if grpc_server is not None:
        await stop_grpc_server(grpc_server)
        grpc_server = None
