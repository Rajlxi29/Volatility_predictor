from dotenv import load_dotenv
from pathlib import Path
import sys
import os

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR/".env")

HF_TOKEN = os.getenv("hf_token")
SERAPI = os.getenv("Serpapi")
OPENAI_API = os.getenv("OPENAI_API_KEY")
OPENAI_BASE = os.getenv("OPENAI_API_BASE")

DB_PATH = BASE_DIR/"investments.db"
ARTIFACT_PATH = BASE_DIR/"artifact"
VOLATILITY_MODEL = ARTIFACT_PATH/"model.pkl"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


