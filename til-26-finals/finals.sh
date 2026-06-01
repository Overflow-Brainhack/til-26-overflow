#!/bin/bash
set -o pipefail

FINALS_TASKS=(ae asr cv noise nlp)
script_dir="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

available_commands() {
    echo "Available til commands:"
    echo "  submit      Submit a Docker image for evaluation on the TIL leaderboard."
    echo "              Pass TASK='finals' to build + push the orchestration server image."
    echo "              Add --submit_all to also push the task images (ae, asr, cv, noise, nlp)."
    echo "              Add --build_all  to force a rebuild of every finals image first."
    echo "  build       Build a Docker image for one of the tasks from a Dockerfile in the 'til-26' directory."
    echo "              Pass TASK='finals' to build the orchestration server in ./finals/ plus any task"
    echo "              :finals images not yet present locally. Add --build_all to rebuild them all."
    echo "  test        Run a full end-to-end finals match locally (test competition server +"
    echo "              opponent stubs + your model containers + orchestration server) and"
    echo "              watch it to completion. Flags: --keep-up, --no-build."
    echo "  help        Show this help message."
    echo ""
    til_folder_warning
}

param_info() {
    echo "Parameters:"
    echo "  TASK        The competition task, one of 'asr', 'cv', 'noise', 'nlp', 'ae', or 'finals'."
    echo "  tag         The tag for the Docker image, defaults to 'latest' if not provided."
    echo "              For TASK='finals', tag is always 'finals' and this argument is ignored."
}

til_folder_warning() {
    echo "NOTE: commands assume your repository directory is named 'til-26'."
    echo "If your repository directory is named something else, set the TIL_FOLDER environment variable to the correct path."
    echo "For example, if your repository is in '/home/jupyter/my-til-repo', run 'export TIL_FOLDER=/home/jupyter/my-til-repo' before running til commands."
    echo "Consider also adding 'export TIL_FOLDER=/home/jupyter/my-til-repo' to your .bash_profile so you don't have to remember to set it every time!"
}

submit_usage() {
    echo "Usage:     til submit TASK [tag]"
    echo "           til submit finals [--submit_all] [--build_all]"
    param_info
    echo "Examples:  til submit asr custom-tag"
    echo "           til submit finals"
    echo "           til submit finals --submit_all"
    echo "           til submit finals --submit_all --build_all"
    echo ""
    echo "  --submit_all  Also push the task images (${FINALS_TASKS[*]}) to Artifact Registry"
    echo "                and upload them to Model Registry. By default 'submit finals' only"
    echo "                pushes the orchestration server image."
    echo "  --build_all   Force a rebuild of every finals image before pushing (ignores images"
    echo "                already present locally)."
}

build_usage() {
    echo "Usage:     til build TASK [tag]"
    echo "           til build finals [--build_all]"
    param_info
    echo "Examples:  til build asr custom-tag"
    echo "           til build finals"
    echo "           til build finals --build_all"
    echo ""
    echo "  --build_all   Rebuild every finals image even if it already exists locally."
    echo "                Without it, task images present locally are reused and only the"
    echo "                orchestration server is (re)built."
    til_folder_warning
}

test_usage() {
    echo "Usage:     til test [--keep-up] [--no-build]"
    echo "Example:   til test"
    echo ""
    echo "  test         Run a full end-to-end finals match locally: brings up the test"
    echo "               competition server + opponent stubs + your model containers +"
    echo "               orchestration server, starts a match, and watches it to completion."
    echo "               Requires your local '<team>-<task>:finals' images ('til build finals')."
    echo "  --keep-up    Leave the stack running after the match (Ctrl-C to stop)."
    echo "  --no-build   Reuse the existing til-finals image instead of rebuilding it."
}

assert_team_and_task() {
    if [ -z "$TEAM_NAME" ];
        then echo "No team name found in environment!"
        exit 1;
    fi
    if [ -z "$1" ];
        then echo "No task provided!"
        param_info
        exit 1;
    fi
    case "$1" in
        asr|cv|noise|nlp|ae)
            ;;
        *)
            echo "ERROR: unknown task '$1'!"
            param_info
            exit 1;
            ;;
    esac
}

assert_team() {
    if [ -z "$TEAM_NAME" ];
        then echo "No team name found in environment!"
        exit 1;
    fi
}

get_task_port() {
    case "$1" in
        asr)
            echo 5001
            ;;
        cv)
            echo 5002
            ;;
        noise)
            echo 5003
            ;;
        nlp)
            echo 5004
            ;;
        ae)
            echo 5005
            ;;
        *)
            echo "ERROR: unknown task '$1'!"
            param_info
            exit 1;
            ;;
    esac
}

til_folder=${TIL_FOLDER:-$HOME/til-26}

ar_repo_for_team() {
    echo "asia-southeast1-docker.pkg.dev/til-ai-2026/repo-til-26-$TEAM_NAME"
}

# Builds <team>-<task>:finals from $til_folder/<task>/Dockerfile. Fail-fast on build failure.
# With force != "true", an image that already exists locally is reused (the default for
# `build finals`); --build_all sets force="true" to rebuild it regardless.
ensure_task_finals_image() {
    local task="$1"
    local force="${2:-false}"
    local image="$TEAM_NAME-$task:finals"

    if [ "$force" != "true" ] && docker image inspect "$image" >/dev/null 2>&1; then
        echo "Reusing existing local image '$image' (pass --build_all to force a rebuild)."
        return 0
    fi

    if [ ! -d "$til_folder/$task" ]; then
        echo "ERROR: directory '$til_folder/$task' does not exist."
        echo "       Expected your til-26 repo to contain a '$task' directory with a Dockerfile."
        til_folder_warning
        exit 1
    fi

    echo "Building '$image' from $til_folder/$task..."
    local cwd
    cwd=$(pwd)
    cd "$til_folder/$task" || { echo "ERROR: could not enter '$til_folder/$task'."; exit 1; }
    if ! docker build -t "$image" -f Dockerfile .; then
        cd "$cwd"
        echo ""
        echo "ERROR: Could not build '$image' from '$til_folder/$task/Dockerfile'."
        echo "       Make sure your til-26 repo has a working '$task/Dockerfile' and that all model"
        echo "       artifacts it needs are in place. You can retry the build with:"
        echo "           ./finals.sh build $task finals"
        exit 1
    fi
    cd "$cwd"
    echo "Built '$image'."
}

# Builds <team>-server:finals from ./finals/Dockerfile (relative to this script). < scuffed. but works
build_server_finals_image() {
    local image="$TEAM_NAME-server:finals"
    local finals_dir="$script_dir/finals"

    if [ ! -d "$finals_dir" ]; then
        echo "ERROR: finals directory '$finals_dir' not found. Are you running finals.sh from the til-26-finals repo?"
        exit 1
    fi

    echo "Building orchestration server image '$image' from $finals_dir..."
    if ! docker build -t "$image" -f "$finals_dir/Dockerfile" "$finals_dir"; then
        echo ""
        echo "ERROR: Could not build '$image' from '$finals_dir/Dockerfile'."
        echo "       Check that ./finals/ in your til-26-finals checkout is intact (try 'git submodule update --init')."
        exit 1
    fi
    echo "Built '$image'."
}

# Tags + pushes <team>-<task>:finals to Artifact Registry and uploads it to
# Model Registry via gcloud ai models upload (real predict-route + port).
push_task_finals_image() {
    local task="$1"
    local image="$TEAM_NAME-$task:finals"
    local ar_ref
    ar_ref="$(ar_repo_for_team)/$TEAM_NAME-$task:finals"
    local port
    port=$(get_task_port "$task")

    echo ""
    echo "── Pushing $task ──────────────────────────────────────"
    echo "Tagging '$image' as '$ar_ref'..."
    docker tag "$image" "$ar_ref" || { echo "ERROR: 'docker tag' failed for $image."; exit 1; }
    echo "Pushing '$ar_ref' to Artifact Registry..."
    docker push "$ar_ref" || { echo "ERROR: 'docker push' failed for $ar_ref."; exit 1; }
    echo "Uploading '$ar_ref' to Model Registry..."
    gcloud ai models upload --region asia-southeast1 --display-name "$TEAM_NAME-$task" \
        --container-image-uri "$ar_ref" --container-health-route /health --container-predict-route "/$task" \
        --container-ports "$port" --version-aliases default \
        || { echo "ERROR: 'gcloud ai models upload' failed for $TEAM_NAME-$task."; exit 1; }
}

# Tags + pushes <team>-server:finals. participant_server.py is a WebSocket
# *client* with no HTTP predict endpoint, so the route/port flags are
# placeholders -- the server-side :finals gate is expected to short-circuit
# endpoint creation for these.
push_server_finals_image() {
    local image="$TEAM_NAME-server:finals"
    local ar_ref
    ar_ref="$(ar_repo_for_team)/$TEAM_NAME-server:finals"

    echo ""
    echo "── Pushing server ─────────────────────────────────────"
    echo "Tagging '$image' as '$ar_ref'..."
    docker tag "$image" "$ar_ref" || { echo "ERROR: 'docker tag' failed for $image."; exit 1; }
    echo "Pushing '$ar_ref' to Artifact Registry..."
    docker push "$ar_ref" || { echo "ERROR: 'docker push' failed for $ar_ref."; exit 1; }
    curl -X POST https://asia-southeast1-til-ai-2026.cloudfunctions.net/evaluator-finals-hardware \
      -H "Content-Type: application/json" \
      -d "{\"team\": \"$TEAM_NAME\"}"
}

# Builds every finals image. force="true" (from --build_all) rebuilds task images even
# if they exist locally; otherwise present task images are reused. The server is always
# (re)built since it is the fast-moving piece you are usually iterating on.
build_finals() {
    local force="${1:-false}"
    assert_team
    echo "Building finals images for team '$TEAM_NAME' (force rebuild: $force)..."
    for task in "${FINALS_TASKS[@]}"; do
        ensure_task_finals_image "$task" "$force"
    done
    build_server_finals_image
    echo ""
    echo "All finals images ready locally."
}

submit_finals() {
    local submit_all="${1:-false}"
    local build_all="${2:-false}"
    assert_team

    if [ "$submit_all" = "true" ]; then
        build_finals "$build_all"
        echo ""
        echo "Pushing all finals task images for team '$TEAM_NAME'..."
        for task in "${FINALS_TASKS[@]}"; do
            push_task_finals_image "$task"
        done
    else
        echo "Building server image only for team '$TEAM_NAME'..."
        echo "(Pass --submit_all to also build and push the task images: ${FINALS_TASKS[*]}.)"
        build_server_finals_image
        echo ""
        echo "Pushing server image for team '$TEAM_NAME'..."
    fi

    push_server_finals_image
    echo ""
    echo "Images submitted. See you at MBS <3"
}

# ───────────────────────────── finals end-to-end test ──────────────────────
#
# Compose does the orchestration. `docker compose up` (attached, so logs stream
# straight to your terminal) brings the whole stack up in dependency order via
# depends_on: HQ -> 5 model containers -> til-finals (which only reports healthy
# once it's actually connected to the HQ). A one-shot `til-starter` service then
# POSTs /start and polls /match_status until the match ends; --abort-on-container-exit
# tears the stack down when it does. This script just prepares the env + a local
# -image override and runs that single `up`.
#
# Globals below are read by the EXIT trap, so they must NOT be declared `local`.
TEST_OVERRIDE=""
TEST_CLEANED=""
declare -a TEST_DC=()

# Reads KEY=VALUE lines from .env without clobbering vars already in the environment
# (matching docker compose precedence: real env wins over the .env file).
load_env_defaults() {
    local envfile="$script_dir/.env" line key
    [ -f "$envfile" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|\#*) continue ;; esac
        key="${line%%=*}"
        [ "$key" = "$line" ] && continue                  # no '='; skip
        case "$key" in *[!A-Za-z0-9_]*) continue ;; esac   # not a valid var name; skip
        if [ -z "${!key+x}" ]; then
            export "$key=${line#*=}"
        fi
    done < "$envfile"
}

finals_test_cleanup() {
    [ -n "$TEST_CLEANED" ] && return
    TEST_CLEANED=1
    if [ "${KEEP_UP:-false}" = "true" ]; then
        echo ""
        echo "── --keep-up set: leaving the stack running. Tear down later with:"
        echo "     docker compose -f docker-compose-test.yml down --remove-orphans"
    elif [ "${#TEST_DC[@]}" -gt 0 ]; then
        echo ""
        echo "── Tearing down the stack ─────────────────────────────"
        "${TEST_DC[@]}" down --remove-orphans >/dev/null 2>&1 || true
    fi
    [ -n "$TEST_OVERRIDE" ] && rm -f "$TEST_OVERRIDE" 2>/dev/null
    return 0
}

run_finals_test() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: docker not found on PATH."
        exit 1
    fi
    if ! docker compose version >/dev/null 2>&1; then
        echo "ERROR: 'docker compose' (v2) is not available."
        exit 1
    fi

    cd "$script_dir" || { echo "ERROR: cannot cd to '$script_dir'."; exit 1; }

    # .env is needed for compose interpolation + env_file. Create it from the example if absent.
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            echo "NOTE: created .env from .env.example. Edit TEAM_NAME / HOST_DATA_DIR for a full run."
        else
            echo "ERROR: no .env and no .env.example to copy from in $script_dir."
            exit 1
        fi
    fi

    load_env_defaults
    : "${COMPETITION_SERVER_PORT:=8000}"
    : "${CONFIG:=config_test}"
    : "${REPO_NAME:=local-test}"   # only used for base-file interpolation; images come from the override
    export COMPETITION_SERVER_PORT CONFIG REPO_NAME

    assert_team   # TEAM_NAME may have come from .env just now

    # NOTE: the HQ accepts WebSocket connections only from teams in its config. We do NOT
    # rewrite any config here — the test competition server and its stubs seat TEAM_NAME at
    # slot 0 themselves (config.py / stub_participants.py inject it from the env), so your
    # team need not be listed in config_test.json.

    # The HQ bind-mounts HOST_DATA_DIR and loads its stage data (corpus, NLP questions,
    # ASR/CV pools) at startup — it is REQUIRED. Without it the competition server crashes
    # on boot and /health never comes up, so fail fast here with guidance instead.
    if [ -z "${HOST_DATA_DIR:-}" ] || [ ! -d "${HOST_DATA_DIR:-/nonexistent}" ]; then
        echo "ERROR: HOST_DATA_DIR ('${HOST_DATA_DIR:-<unset>}') is not a directory."
        case "${HOST_DATA_DIR:-}" in
            *'${'*) echo "       It still contains an unexpanded '\${...}' placeholder — put a real path in .env." ;;
        esac
        echo "       The competition server needs your stage data to start. Set HOST_DATA_DIR in"
        echo "       .env to your stage directory (must contain asr/, cv/, nlp/), e.g.:"
        echo "           HOST_DATA_DIR=/home/jupyter/advanced"
        exit 1
    fi
    if [ ! -f "$HOST_DATA_DIR/nlp/nlp.jsonl" ]; then
        echo "WARNING: '$HOST_DATA_DIR/nlp/nlp.jsonl' not found. The HQ loads asr/, cv/, nlp/ at"
        echo "         startup and crashes on boot if they're missing (then HQ /health times out)."
    fi
    export HOST_DATA_DIR

    local team="$TEAM_NAME"

    # Required local model images (the full finals stack). The server image is (re)built
    # below by `up --build`; the model images are used as-is via the override (pull_policy
    # never), so error early with a friendly message if any are missing.
    local missing=() t
    for t in asr cv noise nlp ae; do
        docker image inspect "$team-$t:finals" >/dev/null 2>&1 || missing+=("$team-$t:finals")
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "ERROR: missing local model image(s): ${missing[*]}"
        echo "       Build your finals images first:"
        echo "           bash finals.sh build finals"
        exit 1
    fi

    # Temp compose override: point the server + model images at the local :finals tags
    # with pull_policy:never so the whole test runs offline against locally built images.
    TEST_OVERRIDE="$(mktemp --suffix=.yml 2>/dev/null || mktemp)"

    TEST_DC=(docker compose -f "$script_dir/docker-compose-test.yml" -f "$TEST_OVERRIDE")
    trap finals_test_cleanup EXIT
    trap 'echo; echo "Interrupted."; exit 130' INT TERM

    mkdir -p "$script_dir/artifacts"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo " TIL-26 finals end-to-end test"
    echo "   team:    $team"
    echo "   config:  $CONFIG   (HQ on port $COMPETITION_SERVER_PORT)"
    echo "   data:    $HOST_DATA_DIR"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "Bringing up the whole stack. Compose orders it (HQ -> models -> til-finals),"
    echo "then til-starter waits for til-finals to connect, POSTs /start, and polls"
    echo "/match_status until the match ends. Logs stream below; Ctrl-C aborts."
    echo ""

    local up_flags=(--build)
    [ "${NO_BUILD:-false}" = "true" ] && up_flags=()

    if [ "${KEEP_UP:-false}" = "true" ]; then
        # No --abort: the stack stays up after the match. til-starter still POSTs /start,
        # watches the match, and prints the final scores; `up` blocks until you Ctrl-C.
        "${TEST_DC[@]}" up "${up_flags[@]}"
    else
        # til-starter exits when the match ends; --abort-on-container-exit then stops the
        # whole stack and `up` returns with the exit code of whichever container exited.
        "${TEST_DC[@]}" up "${up_flags[@]}" --abort-on-container-exit
        local rc=$?
        echo ""
        if [ "$rc" -ne 0 ]; then
            echo "Stack exited with code $rc — a container crashed or the match timed out."
            echo "Scroll up for the failing container's logs, or run:"
            echo "    docker compose -f docker-compose-test.yml logs <service>"
        fi
        finals_test_summary
    fi
}

# Best-effort leaderboard from the newest match's persisted match_results.jsonl
# (the artifacts dir is bind-mounted, so it survives teardown).
finals_test_summary() {
    local match_dir results
    match_dir="$(ls -1dt "$script_dir/artifacts"/match_* 2>/dev/null | head -n1 || true)"
    [ -n "$match_dir" ] || return 0
    results="$match_dir/match_results.jsonl"
    echo "── Match summary ──────────────────────────────────────"
    if [ -f "$results" ] && command -v jq >/dev/null 2>&1; then
        grep '"type": *"summary"' "$results" | tail -n1 \
            | jq -r '.scores[]? | "   \(.team): final=\(.final)  ae=\(.ae)  mult=\(.mission_multiplier)  batches=\(.batches_completed)"' 2>/dev/null \
            || echo "   (open ${match_dir#"$script_dir"/}/match_results.jsonl)"
    else
        echo "   scores in ${match_dir#"$script_dir"/}/match_results.jsonl (install jq for a table)"
    fi
    echo "   Artifacts: ${match_dir#"$script_dir"/}"
}

case "$1" in
    submit)
        if [ "$2" = "finals" ]; then
            submit_all=false
            build_all=false
            for arg in "${@:3}"; do
                case "$arg" in
                    --submit_all) submit_all=true ;;
                    --build_all)  build_all=true ;;
                    *) echo "WARNING: ignoring unknown flag '$arg' for 'submit finals'." ;;
                esac
            done
            submit_finals "$submit_all" "$build_all"
            exit 0
        fi
        assert_team_and_task "$2"
        task="$2"

        if [ -z "$3" ];
            then echo "No tag provided, defaulting to 'latest'."
        fi
        tag=${3:-"latest"}
        image="$TEAM_NAME-$task"
        image_ref="$image:$tag"

        echo "Image:   $image"
        echo "Tag:     ${tag:-<none>}"

        port=$(get_task_port $task)
        if [[ "$image" == asia-southeast1-docker.pkg.dev/til-ai-2026/* ]]; then
            echo "Image $image is already an Artifact Registry tag, not retagging"
            ar_ref=$image:$tag
        else
            ar_ref=asia-southeast1-docker.pkg.dev/til-ai-2026/repo-til-26-$TEAM_NAME/$image:$tag
            echo "Tagging '$image:$tag' as '$ar_ref'..."
            docker tag $image:$tag $ar_ref
        fi
        echo "Pushing '$ar_ref' to Artifact Registry..."
        docker push $ar_ref && \
        echo "Submitting '$ar_ref' for automatic evaluation..." && \
        gcloud ai models upload --region asia-southeast1 --display-name "$TEAM_NAME-$task" \
            --container-image-uri $ar_ref --container-health-route /health --container-predict-route /$task \
            --container-ports $port --version-aliases default
        ;;
    build)
        if [ "$2" = "finals" ]; then
            build_all=false
            for arg in "${@:3}"; do
                case "$arg" in
                    --build_all) build_all=true ;;
                    --submit_all) echo "NOTE: --submit_all has no effect on 'build'; use 'submit finals --submit_all'." ;;
                    *) echo "WARNING: ignoring unknown flag '$arg' for 'build finals'." ;;
                esac
            done
            build_finals "$build_all"
            exit 0
        fi
        assert_team_and_task "$2"
        task="$2"
        if [ -z "$3" ];
            then echo "No tag provided, defaulting to 'latest'."
        fi
        tag=${3:-"latest"}
        echo "Building image for task '$task' with tag '$tag'..."
        cwd=$(pwd)
        cd "$til_folder/$task" || { echo "Could not find directory for task '$task'! Are you sure you have a directory named '$task' in your 'til-26' directory?"; til_folder_warning; cd "$cwd"; exit 1; }
        docker build -t $TEAM_NAME-$task:$tag -f Dockerfile .
        cd "$cwd"
        echo "Build complete! You can now test this image with 'til test $task $tag' or submit this image with 'til submit $task $tag'"
        ;;
    test)
        # This repo only tests the finals stack end-to-end, so `test` always runs the
        # full match. A literal `finals` arg is accepted (and ignored) for muscle memory.
        KEEP_UP=false
        NO_BUILD=false
        for arg in "${@:2}"; do
            case "$arg" in
                finals)     ;;  # optional — finals is the only thing this tests
                --keep-up)  KEEP_UP=true ;;
                --no-build) NO_BUILD=true ;;
                *) echo "WARNING: ignoring unknown arg '$arg' for 'test'." ;;
            esac
        done
        run_finals_test
        ;;
    help|--help|-h)
        available_commands
        ;;
    *)
        echo "Unknown til command: $1"
        available_commands
        exit 1;
        ;;
esac
