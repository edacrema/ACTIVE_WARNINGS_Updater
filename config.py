import os
from dotenv import load_dotenv

load_dotenv()

# Google Cloud / Vertex AI
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "wfp-oev-2404-evidence-mining")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-pro")
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 8192

# Cloud Run
PORT = int(os.getenv("PORT", "8080"))

# API Keys
RELIEFWEB_APPNAME = os.getenv("RELIEFWEB_APPNAME", "WFP-EW-gFDVc8Qw15Cx7")

# Graph
MAX_CORRECTION_ATTEMPTS = 3

# Defaults
DEFAULT_UPDATE_PERIOD_DAYS = 60
