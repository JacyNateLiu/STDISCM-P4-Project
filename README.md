# How to Run
1) python -m pip install -r server/requirements.txt
2) uvicorn dashboard.server.app:app --reload --port 8000
3) Open http://127.0.0.1:8000 in your browser
4) From a second terminal in the same folder run python training_client.py

# Training Dashboard

This project provides a lightweight real-time dashboard for visualizing convolutional neural network (CNN) training runs. It implements an RPC-inspired ingestion endpoint that AI training jobs can call to stream their mini-batch results. The dashboard is delivered as a FastAPI web application with a static, interactive frontend.

## Key Features

- **Real-time ingest pipeline** – Training jobs send JSON RPC payloads to `/rpc/batch_update`, which the server validates, aggregates into 16-tile batches, and broadcasts through a WebSocket.
- **Mini-batch aggregation** – When upstream training batches are smaller than the mandated 16 tiles, the server accumulates the entries until it can render a complete tile. For larger batches, a random 16-sample subset is emitted to keep the UI responsive.
- **Loss tracking** – Every RPC call contributes a loss point. The frontend plots the sliding history using Chart.js and only keeps the latest 1,024 iterations in memory.
- **Interactive dashboard** – Users can pause/resume streaming updates, inspect the current FPS, and zoom in on the loss curve. Image tiles and prediction labels refresh whenever new state arrives.
- **Fault tolerance hooks** – Optional retry-friendly semantics (idempotent iteration keys) guard against duplicate RPC calls, and configurable artificial frame delays help stress-test concurrency.

## Architecture Overview

```
training_client.py  -->  POST /rpc/batch_update  -->  FastAPI server  -->  WebSocket broadcast  -->  frontend/main.js
```

1. **RPC ingestion (HTTP POST)**
   - Pydantic schema enforces payload correctness (iteration, loss, latency metadata, and up to 64 sample entries).
   - The server optionally waits for `FRAME_DELAY_MS` to simulate slow frames.
   - After validation, the data is merged into the rolling dashboard state.

2. **Dashboard state manager**
   - Maintains the latest 16-tile snapshot, accumulated loss history, and most recent iteration metadata in a thread-safe structure.
   - Provides idempotent updates by comparing iteration numbers, so the dashboard tolerates duplicate batch submissions.

3. **WebSocket broadcaster**
   - A background task fans out state changes to any connected dashboards.
   - Clients receive incremental patches and reconcile them locally to minimize refresh work.

4. **Frontend**
   - Vanilla HTML/CSS/JS with Chart.js for the loss curve and requestAnimationFrame measurements for FPS.
   - Image tiles auto-scale (ensuring maximum 512×512) and fall back to placeholders until a full aggregated set arrives.

See the Usage section below for setup instructions once the implementation is complete.

## Setup

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

    ```powershell
    python -m pip install -r server/requirements.txt
    ```

3. Start the dashboard (serves both the API and the static UI):

    ```powershell
    uvicorn dashboard.server.app:app --reload --port 8000
    ```

4. Open `http://127.0.0.1:8000` in a browser to view the dashboard. All static assets live under `/dashboard/*`, and the WebSocket stream is exposed at `/ws`.

### Environment Toggles

`FRAME_DELAY_MS` &mdash; Optional server-side delay (in milliseconds) injected before broadcasting each RPC update. Helpful for concurrency testing.

## RPC Contract

Training jobs stream batches through `POST /rpc/batch_update` using the following JSON structure:

```json
{
   "iteration": 42,
   "loss": 0.3821,
   "batch_size": 24,
   "delay_ms": 0,
   "samples": [
      {
         "sample_id": "42-0",
         "prediction": "cat",
         "ground_truth": "cat",
         "confidence": 0.91,
         "image_b64": "<base64 png>"
      }
   ]
}
```

- Up to 128 samples can be included per request.
- Images must be ≤512×512. Smaller resolutions are automatically pixel scaled by the UI.
- If a full set of 16 tiles is not provided in a single request, the server aggregates consecutive mini-batches until it can render a 16-tile mosaic. Batches larger than 16 are uniformly down-sampled on the server side.
- Duplicate or out-of-order iterations are ignored so RPC retries remain idempotent.

The server responds with a light acknowledgement:

```json
{
   "status": "ok",
   "iteration": 42,
   "loss": 0.3821,
   "tilesReady": true
}
```

## Sample Training Client

`training_client.py` fabricates a CNN training loop, generates synthetic image batches with Pillow/Numpy, and pushes them to the dashboard. Run it after starting the server:

```powershell
python training_client.py
```

Key environment variables:

- `DASHBOARD_RPC_URL` &mdash; Override the default `http://127.0.0.1:8000/rpc/batch_update` target.
- `TOTAL_ITERATIONS` &mdash; Total number of simulated updates (default 400).
- `MINI_BATCH_MIN` / `MINI_BATCH_MAX` &mdash; Control the random mini-batch sizes to validate the aggregation logic.
- `INJECT_DELAY_MS` &mdash; Mirrors the server-side delay flag so you can stress-test the buffering at both ends.

## Frontend Behavior

- Displays a 4×4 tile of the latest aggregated samples plus a mirrored prediction tile grid for easy inspection of `prediction vs. ground-truth`.
- Streams loss values into a Chart.js line plot (animation disabled to keep up with high-frequency updates).
- Shows live FPS (computed via `requestAnimationFrame`) and exposes Pause/Resume controls to inspect a frozen frame.
- Automatically reconnects the WebSocket client and buffers the most recent packet while paused, ensuring no data is lost.

## Testing Checklist

- Run the FastAPI server with `uvicorn`.
- Launch `training_client.py` to stream random batches; confirm:
   - Image + label tiles refresh every time 16 new samples are ready.
   - The tile status badge flips between `ready 16/16` and `aggregating N/16` for tiny mini-batches.
   - The loss line chart receives every iteration.
   - The FPS counter updates and remains visible.
- Optionally set `FRAME_DELAY_MS=500` (server) or `INJECT_DELAY_MS=200` (client) to simulate slow producers and observe that the UI remains responsive.
