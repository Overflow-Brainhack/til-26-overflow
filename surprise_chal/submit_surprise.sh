#!/usr/bin/env bash
#
# Build, push, and submit your SURPRISE player.
#
# Run this from your Vertex AI Workbench notebook (in this directory). The
# notebook is already authenticated as your team's service account, which is
# the identity the evaluator requires — no extra login.
#
# Usage:
#   ./submit_surprise.sh                                        # submit the algo agent
#   AGENT=llm OPENROUTER_API_KEY=sk-... ./submit_surprise.sh    # submit the LLM agent
#
# The competition coordinates are fixed (region asia-southeast1, project til-26)
# and TEAM_NAME is already set in your notebook environment, so you normally
# pass nothing. They can be overridden via env vars only if the organisers say so.

set -euo pipefail

TEAM_NAME="${TEAM_NAME:?TEAM_NAME is not set — it should already be set in your Workbench notebook}"
REGION="${REGION:-asia-southeast1}"
PROJECT="${PROJECT:-til-ai-2026}"
TAG="${TAG:-latest}"
AGENT="${AGENT:-algo}"                 # algo | llm

# The image name MUST end in "surprise" — that suffix is how the evaluator
# knows which task you're submitting.
IMAGE_NAME="${TEAM_NAME}-surprise"
GAR_HOST="${REGION}-docker.pkg.dev"
IMAGE="${GAR_HOST}/${PROJECT}/repo-til-26-${TEAM_NAME}/${IMAGE_NAME}:${TAG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The LLM agent needs its key baked into the model's container env.
CONTAINER_ENV="AGENT=${AGENT}"
if [[ "$AGENT" == "llm" ]]; then
  : "${OPENROUTER_API_KEY:?AGENT=llm requires OPENROUTER_API_KEY}"
  CONTAINER_ENV="${CONTAINER_ENV},OPENROUTER_API_KEY=${OPENROUTER_API_KEY}"
  [[ -n "${OPENROUTER_MODEL:-}" ]] && CONTAINER_ENV="${CONTAINER_ENV},OPENROUTER_MODEL=${OPENROUTER_MODEL}"
fi

echo "==> Submitting ${AGENT} agent for team ${TEAM_NAME}"
echo "    image: ${IMAGE}"

# 1. build participant/
docker build --build-arg AGENT="$AGENT" --build-arg OPENROUTER_API_KEY="$OPENROUTER_API_KEY" -t "$IMAGE" "${SCRIPT_DIR}/participant"

# 2. push to your Artifact Registry repo
gcloud auth configure-docker "$GAR_HOST" --quiet
docker push "$IMAGE"

# 3. upload as a Vertex model — this is what triggers evaluation
gcloud ai models upload \
  --project="$PROJECT" \
  --region="$REGION" \
  --display-name="$IMAGE_NAME" \
  --container-image-uri="$IMAGE" \
  --container-ports=6700 \
  --container-health-route=/health \
  --container-predict-route=/observe \
  --container-env-vars="$CONTAINER_ENV"

echo "==> Submitted. Watch the competition Discord for your queue position and result."
