import os
from dotenv import load_dotenv

load_dotenv()

# Google Cloud / Vertex AI
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "wfp-oev-2404-evidence-mining")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-pro")
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 8192
LLM_EXTRACTION_MAX_TOKENS = 16384  # Higher limit for single-pass event extraction

# Cloud Run
PORT = int(os.getenv("PORT", "8080"))

# API Keys
RELIEFWEB_APPNAME = os.getenv("RELIEFWEB_APPNAME", "WFP-EW-gFDVc8Qw15Cx7")
SEERIST_API_KEY = os.getenv("SEERIST_API_KEY", "")
SEERIST_BASE_URL = "https://app.seerist.com/hyperionapi/v1/wod"
SEERIST_DEFAULT_PAGE_SIZE = 50

# Graph
MAX_CORRECTION_ATTEMPTS = 3

# Defaults
DEFAULT_UPDATE_PERIOD_DAYS = 60
