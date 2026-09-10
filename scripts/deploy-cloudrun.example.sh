#!/bin/bash
set -e

# ====================================================================
# Template deployment script for Sorix documentation to Google Cloud Run
# Copy this script or use it as reference. Configuration is read from
# the gitignored file: .env.deploy
# ====================================================================

ENV_FILE=".env.deploy"

if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Error: Configuration file '$ENV_FILE' not found!"
    echo "ℹ️ Please create it by copying the template:"
    echo "   cp .env.deploy.example .env.deploy"
    echo "   (and adjust your GCP project and service account settings)"
    exit 1
fi

# Load environment configuration
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# Load Google Cloud SDK if available in user's Downloads or PATH
if [ -f "$HOME/Downloads/google-cloud-sdk/path.bash.inc" ]; then
    # shellcheck disable=SC1091
    source "$HOME/Downloads/google-cloud-sdk/path.bash.inc"
fi

echo "========================================================"
echo "🚀 Building, pushing & deploying Sorix Documentation"
echo "   Service: $SERVICE_NAME"
echo "   Project: $PROJECT_ID"
echo "   Region:  $REGION"
echo "========================================================"

# 1. Compile documentation locally with uv & mkdocs
echo "📚 Compiling documentation with MkDocs..."
uv run mkdocs build

# 2. Configure Docker auth for Google Artifact Registry
echo "🔑 Configuring Docker authentication for $REGION-docker.pkg.dev..."
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet

# 3. Build Docker image (Nginx with pre-compiled site)
echo "📦 Building Docker image ($IMAGE_NAME:latest)..."
docker build -f ./Dockerfile -t "$IMAGE_NAME:latest" .

# 3. Tag and Push to Google Artifact Registry
echo "🏷️ Tagging image as $REGISTRY_URL..."
docker tag "$IMAGE_NAME:latest" "$REGISTRY_URL"

echo "⬆️ Pushing image to Google Artifact Registry..."
docker push "$REGISTRY_URL"

# 4. Deploy to Google Cloud Run
echo "☁️ Deploying service $SERVICE_NAME to Cloud Run ($REGION)..."
gcloud run deploy "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --image "$REGISTRY_URL" \
  --region "$REGION" \
  --platform managed \
  --service-account "$SERVICE_ACCOUNT" \
  --timeout="$TIMEOUT" \
  --memory="$MEMORY" \
  --cpu="$CPU" \
  --cpu-boost \
  --concurrency="${CONCURRENCY:-80}" \
  $AUTHENTICATION

# 5. Clean up inactive revisions
echo "🧹 Cleaning inactive revisions..."
gcloud run revisions list --service "$SERVICE_NAME" --project "$PROJECT_ID" --region "$REGION" \
  --filter="status.conditions.type:Active AND status.conditions.status:False" \
  --format='value(metadata.name)' | xargs -r -L1 gcloud run revisions delete --quiet --project "$PROJECT_ID" --region "$REGION" || true

echo "========================================================"
echo "✅ Sorix Docs deployed successfully to Cloud Run!"
echo "   You can now map your custom domain (e.g. sorix.mitchellmirano.com)"
echo "   in Google Cloud Run -> Manage Custom Domains."
echo "========================================================"
