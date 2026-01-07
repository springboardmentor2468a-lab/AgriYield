FinalProject
============

Structure:
- Frontend/: contains `app.py` (Streamlit app)
- Backend/: trained model artifacts (.pkl)
- Test/: regression notebook
- OtherFiles/: CSV data and `requirements.txt`

Quick start:

1. Create venv and install dependencies (recommended):

```bash
cd "FinalProject"
./setup_venv.sh
```

2. Run the app (will prefer `.venv` then fallback to existing `.venv_new`):

```bash
./run_streamlit.sh
```

If your system uses zsh and you see startup errors, run under bash:

```bash
SHELL=/bin/bash ./run_streamlit.sh
```

The app opens in Firefox by default. To change browser, edit `run_streamlit.sh` and set `BROWSER` accordingly.
