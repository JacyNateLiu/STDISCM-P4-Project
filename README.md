# How to Run
All commands below assume your terminal is opened in this folder (the project root).

1) (Recommended) Use Python 3.12 and create/activate a virtual environment:

   ```powershell
   py -3.12 -m venv .winvenv
   .\.winvenv\Scripts\Activate.ps1
   ```

2) Install dependencies:

   ```powershell
   cd dashboard
   python -m pip install -r server/requirements.txt
   ```

3) Start the FastAPI + gRPC server (leave it running):

   ```powershell
   python -m uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
   ```

4) Open the dashboard UI in a browser:

   ```text
   http://127.0.0.1:8000
   ```

5) In a **second** terminal, activate the same virtual environment and run the synthetic trainer (streams over gRPC by default):

   ```powershell
   .\.winvenv\Scripts\Activate.ps1
   cd dashboard
   python training_client.py
   ```