from __future__ import annotations

import asyncio
from typing import Dict

from fastapi import WebSocket


class BroadcastHub:
    """Minimal pub-sub helper for WebSocket clients."""

    def __init__(self) -> None:
        self._connections: Dict[int, WebSocket] = {}
        self._lock = asyncio.Lock()
        self._next_client_id = 0

    async def register(self, websocket: WebSocket) -> int:
        await websocket.accept()
        async with self._lock:
            client_id = self._next_client_id
            self._next_client_id += 1
            self._connections[client_id] = websocket
        return client_id

    async def unregister(self, client_id: int) -> None:
        async with self._lock:
            websocket = self._connections.pop(client_id, None)
        if websocket:
            await self._safe_close(websocket)

    async def broadcast(self, message: str) -> None:
        async with self._lock:
            items = list(self._connections.items())
        for client_id, websocket in items:
            try:
                await websocket.send_text(message)
            except Exception:
                await self._safe_close(websocket)
                async with self._lock:
                    self._connections.pop(client_id, None)

    async def _safe_close(self, websocket: WebSocket) -> None:
        try:
            await websocket.close()
        except Exception:
            pass
