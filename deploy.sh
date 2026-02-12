#!/usr/bin/env bash
#
# Deploy the Active Warning Updater to Google Cloud Run.
#
# Prerequisites:
#   1. gcloud CLI installed and authenticated (gcloud auth login)
#   2. Docker or Cloud Build enabled on the project
#   3. The Cloud Run service account needs the "Vertex AI User" IAM role
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh                          # uses defaults
#   ./deploy.sh my-project us-central1   # custom project/region
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override with arguments or env vars)
# ---------------------------------------------------------------------------
PROJECT_ID="${1:-${GCP_PROJECT_ID:-wfp-oev-2404-evidence-mining}}"
REGION="${2:-${GCP_LOCATION:-us-central1}}"
SERVICE_NAME="active-warning-updater"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "============================================"
echo "  Deploying ${SERVICE_NAME}"
echo "  Project : ${PROJECT_ID}"
echo "  Region  : ${REGION}"
echo "  Image   : ${IMAGE_NAME}"
echo "============================================"

# ---------------------------------------------------------------------------
# 1. Set the active project
# ---------------------------------------------------------------------------
gcloud config set project "${PROJECT_ID}"

# ---------------------------------------------------------------------------
# 2. Enable required APIs (idempotent)
# ---------------------------------------------------------------------------
echo ">> Enabling required GCP APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    aiplatform.googleapis.com \
    artifactregistry.googleapis.com \
    --quiet

# ---------------------------------------------------------------------------
# 3. Build the container image using Cloud Build
# ---------------------------------------------------------------------------
echo ">> Building container image with Cloud Build..."
gcloud builds submit \
    --tag "${IMAGE_NAME}" \
    --timeout=1200 \
    --quiet

# ---------------------------------------------------------------------------
# 4. Deploy to Cloud Run
# ---------------------------------------------------------------------------
echo ">> Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_NAME}" \
    --region "${REGION}" \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 900 \
    --concurrency 4 \
    --min-instances 0 \
    --max-instances 3 \
    --set-env-vars "GCP_PROJECT_ID=${PROJECT_ID},GCP_LOCATION=${REGION},LLM_MODEL=gemini-2.5-pro,RELIEFWEB_APPNAME=WFP-EW-gFDVc8Qw15Cx7" \
    --quiet

# ---------------------------------------------------------------------------
# 5. Print the service URL
# ---------------------------------------------------------------------------
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region "${REGION}" \
    --format "value(status.url)")

echo ""
echo "============================================"
echo "  Deployment complete!"
echo "  URL: ${SERVICE_URL}"
echo "============================================"
