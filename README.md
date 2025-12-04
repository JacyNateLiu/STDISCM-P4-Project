# How to Run
All commands below assume your terminal is opened in this folder (the project root).

1) (Recommended) Use Python 3.12 and create/activate a virtual environment:

   ```powershell
   py -3.12 -m venv .winvenv
   .\.winvenv\Scripts\Activate.ps1
   ```

2) Install dependencies:

   ```powershell
   python -m pip install -r server/requirements.txt
   ```

   PyTorch is now required for the trainer. Install the CPU build (or follow
   [pytorch.org](https://pytorch.org/get-started/locally/) for GPU-specific commands):

   ```powershell
   python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

3) Start the FastAPI + gRPC server (leave it running):

   ```powershell
   python -m uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
   ```

4) Open the dashboard UI in a browser:

   ```text
   http://127.0.0.1:8000
   ```

5) In a **second** terminal, activate the same virtual environment and run the trainer (it now trains a small CNN over the bundled chess dataset and streams real losses/predictions over gRPC only):

   ```powershell
   .\.winvenv\Scripts\Activate.ps1
   python training_client.py
   ```

   The HTTP ingestion endpoint has been removed to avoid redundancy, so the trainer exclusively targets the gRPC server (`DASHBOARD_GRPC_TARGET` controls the host/port if you need to customize it).

The trainer now learns directly from the chess-piece tiles under `assets/` (already included in this repo). You can drop additional PNG/JPG samples into `assets/<PieceName>/` to customize the dataset.

## Debug / Fault-Tolerance Smoke Test

Use `debug_state.py` to reproduce the concurrency and retry guarantees without spinning up the whole stack. From the project root, run:

```powershell
.\.winvenv\Scripts\Activate.ps1
python debug_state.py
```
