#!/bin/bash
set -e

echo "Starting FastAPI on 8000..."
uvicorn src.main:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit on 8501..."
streamlit run streamlit_app/Home.py --server.port=8501 --server.address=0.0.0.0 &

wait