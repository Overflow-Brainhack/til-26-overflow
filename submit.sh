#!/usr/bin/env bash
#
# submit.sh — local replacement for `til submit`.
#
# Runs the three steps described in til-26.wiki/Submitting-models.md from a
# local machine (instead of the Workbench instance):
#
#   1. docker tag    — local image -> Artifact Registry URI
#   2. docker push   — push to GAR
#   3. gcloud ai models upload — register the image with Vertex AI for eval
#
# Prerequisites (one-time setup, see the wiki for details):
#   • gcloud + docker installed and on PATH
#   • TEAM_ID exported (in .env or your shell)
#
# Auth (pick one):
#   Impersonation (default) — your local gcloud identity must have
#     roles/iam.serviceAccountTokenCreator on the team service account.
#     Run `gcloud auth login` once; tokens refresh automatically.
#
#   Token fallback — set GCLOUD_ACCESS_TOKEN in .env (or your shell).
#     Run `gcloud auth print-access-token` on your Workbench instance and
#     copy the output.  Tokens expire after ~1 hour; refresh when needed.
#     The token fallback takes precedence if the variable is set.
#
# Usage:
#   ./submit.sh CHALLENGE [TAG]                 # CHALLENGE in {asr,cv,nlp,ae,noise}; TAG defaults to "latest"
#   ./submit.sh ae
#   ./submit.sh nlp best-rag
#   ./submit.sh --dry-run cv                    # print commands without running them
#   ./submit.sh --build ae                      # build image (linux/amd64) then submit

set -euo pipefail

REGION="asia-southeast1"
REGISTRY="${REGION}-docker.pkg.dev"
PROJECT="til-ai-2026"
SERVICE_ACCOUNT="svc-overflow@til-ai-2026.iam.gserviceaccount.com"

# ─── per-challenge config (port + predict route) ───────────────────────────
challenge_port() {
  case "$1" in
  asr) echo 5001 ;;
  cv) echo 5002 ;;
  noise) echo 5003 ;;
  nlp) echo 5004 ;;
  ae) echo 5005 ;;
  *) return 1 ;;
  esac
}
challenge_route() {
  # All challenges' predict route is /<challenge>.
  echo "/$1"
}
VALID_CHALLENGES="asr cv noise nlp ae"

# ─── arg parsing ───────────────────────────────────────────────────────────
DRY_RUN=0
SKIP_LOGIN=0
BUILD=0
CHALLENGE=""
TAG="latest"

usage() {
  sed -n '2,/^set/p' "$0" | sed 's/^# \?//;/^set/d'
  exit "${1:-0}"
}

while (($#)); do
  case "$1" in
  -h | --help) usage 0 ;;
  --dry-run) DRY_RUN=1 ;;
  --build) BUILD=1 ;;
  --skip-login) SKIP_LOGIN=1 ;;
  --)
    shift
    break
    ;;
  -*)
    echo "unknown flag: $1" >&2
    usage 2
    ;;
  *) if [[ -z "$CHALLENGE" ]]; then
    CHALLENGE="$1"
  else
    TAG="$1"
  fi ;;
  esac
  shift
done

if [[ -z "$CHALLENGE" ]]; then
  echo "error: CHALLENGE is required" >&2
  usage 2
fi

if ! PORT=$(challenge_port "$CHALLENGE"); then
  echo "error: unknown challenge '$CHALLENGE' (expected one of: $VALID_CHALLENGES)" >&2
  exit 2
fi
ROUTE=$(challenge_route "$CHALLENGE")

# ─── env config ────────────────────────────────────────────────────────────
# Source .env if present (without polluting the user's shell).
if [[ -f .env ]]; then
  set -a
  . ./.env
  set +a
fi

TEAM_ID="${TEAM_ID:-${TEAM_NAME:-}}"
if [[ -z "$TEAM_ID" ]]; then
  cat >&2 <<EOF
error: TEAM_ID not set.
  Add 'TEAM_ID=your-team-id' to .env, or export it in your shell.
  (Falling back to TEAM_NAME would also work.)
EOF
  exit 2
fi

# ─── auth setup ────────────────────────────────────────────────────────────
# Token fallback takes precedence if GCLOUD_ACCESS_TOKEN is set.
if [[ -n "${GCLOUD_ACCESS_TOKEN:-}" ]]; then
  AUTH_MODE="token"
  # Let gcloud pick up the token from the environment for all subcommands.
  export CLOUDSDK_AUTH_ACCESS_TOKEN="$GCLOUD_ACCESS_TOKEN"
else
  AUTH_MODE="impersonate"
  export CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT="$SERVICE_ACCOUNT"
fi

REPO="${REGISTRY}/${PROJECT}/repo-til-26-${TEAM_ID}"
IMAGE_NAME="${TEAM_ID}-${CHALLENGE}"
LOCAL_REF="${IMAGE_NAME}:${TAG}"
REMOTE_REF="${REPO}/${IMAGE_NAME}:${TAG}"

# ─── tool checks ───────────────────────────────────────────────────────────
need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    if ((DRY_RUN)); then
      echo "warning: '$1' not found on PATH (dry-run continues anyway)" >&2
    else
      echo "error: '$1' not found on PATH" >&2
      exit 127
    fi
  fi
}
need docker
need gcloud

# ─── helpers ───────────────────────────────────────────────────────────────
# Print to stderr so callers like `run cmd > file` capture only cmd's stdout.
run() {
  printf '\033[1;34m>>\033[0m %s\n' "$*" >&2
  if ((DRY_RUN)); then return 0; fi
  "$@"
}

# ─── build (if requested) ──────────────────────────────────────────────────
if ((BUILD)); then
  run docker build --platform linux/amd64 -t "$LOCAL_REF" "$CHALLENGE"
fi

# ─── verify the local image exists ─────────────────────────────────────────
if ((!DRY_RUN && !BUILD)); then
  if ! docker image inspect "$LOCAL_REF" >/dev/null 2>&1; then
    cat >&2 <<EOF
error: local image '$LOCAL_REF' not found.
  Build it first:
    cd $CHALLENGE && docker build --platform linux/amd64 -t '$LOCAL_REF' .
  Or pass --build to have this script do it automatically.
EOF
    exit 1
  fi
fi

# ─── plan summary ──────────────────────────────────────────────────────────
cat <<EOF
Submitting:
  challenge:   $CHALLENGE
  team:        $TEAM_ID
  local image: $LOCAL_REF
  remote ref:  $REMOTE_REF
  port:        $PORT
  predict:     $ROUTE
  health:      /health
  auth:        $AUTH_MODE${AUTH_MODE:+ ($([[ $AUTH_MODE == impersonate ]] && echo "$SERVICE_ACCOUNT" || echo "GCLOUD_ACCESS_TOKEN"))}
  dry-run:     $((DRY_RUN ? 1 : 0))
EOF
echo

# ─── docker login (if not skipped) ─────────────────────────────────────────
if ((!SKIP_LOGIN)); then
  printf '\033[1;34m>>\033[0m %s\n' \
    "docker login -u oauth2accesstoken --password-stdin https://$REGISTRY" >&2
  if ((!DRY_RUN)); then
    if [[ "$AUTH_MODE" == "impersonate" ]]; then
      gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin "https://$REGISTRY"
    else
      echo "$GCLOUD_ACCESS_TOKEN" | docker login -u oauth2accesstoken --password-stdin "https://$REGISTRY"
    fi
  fi
fi

# ─── tag + push ────────────────────────────────────────────────────────────
run docker tag "$LOCAL_REF" "$REMOTE_REF"
run docker push "$REMOTE_REF"

# ─── upload to Vertex AI ───────────────────────────────────────────────────
run gcloud ai models upload \
  --project="$PROJECT" \
  --region="$REGION" \
  --display-name="$IMAGE_NAME" \
  --container-image-uri="$REMOTE_REF" \
  --container-health-route="/health" \
  --container-predict-route="$ROUTE" \
  --container-ports="$PORT" \
  --version-aliases="default"

echo
echo "✓ Submitted $LOCAL_REF as $IMAGE_NAME on $REGION."

