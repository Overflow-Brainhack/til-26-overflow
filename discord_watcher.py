#!/usr/bin/env python3
"""
discord_watcher.py — selfbot that listens for evaluation results and auto-submits.

Requires DISCORD_TOKEN and DISCORD_CHANNEL_ID in .env (or environment).
Requires gcloud authentication: `gcloud auth application-default login`
Optional env vars:
  DISCORD_GUILD_ID    — restrict to a specific guild
  WATCH_CHALLENGES    — comma-separated list of challenges to act on (default: all five)
  SUBMIT_FLAGS        — space-separated flags prepended to submit.sh (e.g. "--build --dry-run")

Run:
    python discord_watcher.py
    python discord_watcher.py 2>&1 | tee logs/watcher.log
"""

import asyncio
import logging
import os
import re
import shutil
import sys
import tomllib
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import discord
from dotenv import load_dotenv

# ── constants ──────────────────────────────────────────────────────────────────

VALID_CHALLENGES = frozenset({"asr", "cv", "noise", "nlp", "ae"})

RESULT_PATTERN = re.compile(
    r"\*\*Image name\*\*:\s+`\S+-(?P<challenge>\w+)`\s+"
    r"\*\*Image tag\*\*:\s+`(?P<tag>[^`]+)`\s+"
    r"\*\*Submission time\*\*:\s+.+?\n"
    r"\*\*Errors\*\*:\s+(?P<errors>\d+)\s+of\s+\d+\s+tests\s+"
    r"\*\*Score\*\*:\s+(?P<score>[\d.]+)\s+"
    r"\*\*Speed\*\*:\s+(?P<speed>[\d.]+)",
    re.MULTILINE,
)

REPO_ROOT = Path(__file__).parent

# ── data classes ───────────────────────────────────────────────────────────────


@dataclass
class Config:
    token: str
    channel_id: int
    guild_id: int | None
    watch_challenges: frozenset[str]  # empty = watch all valid challenges
    submit_flags: list[str]           # extra flags prepended to submit.sh
    dry_run: bool                     # log the command without executing it


@dataclass
class EvalResult:
    challenge: str
    tag: str
    errors: int
    score: float
    speed: float


# ── config loading ─────────────────────────────────────────────────────────────


def load_config() -> Config:
    load_dotenv(REPO_ROOT / ".env")

    token = os.environ.get("DISCORD_TOKEN", "")
    if not token:
        raise ValueError(
            "DISCORD_TOKEN is not set.\n"
            "  Add it to .env or export it in your shell.\n"
            "  Obtain it from browser DevTools → Network → any Discord API request → Authorization header."
        )

    raw_channel = os.environ.get("DISCORD_CHANNEL_ID", "")
    if not raw_channel:
        raise ValueError(
            "DISCORD_CHANNEL_ID is not set.\n"
            "  Add it to .env.  Enable Developer Mode in Discord, then right-click the channel → Copy ID."
        )
    try:
        channel_id = int(raw_channel)
    except ValueError:
        raise ValueError(f"DISCORD_CHANNEL_ID must be a numeric snowflake, got: {raw_channel!r}")

    raw_guild = os.environ.get("DISCORD_GUILD_ID", "")
    guild_id: int | None = None
    if raw_guild:
        try:
            guild_id = int(raw_guild)
        except ValueError:
            raise ValueError(f"DISCORD_GUILD_ID must be a numeric snowflake, got: {raw_guild!r}")

    raw_challenges = os.environ.get("WATCH_CHALLENGES", "")
    if raw_challenges:
        watch_challenges = frozenset(c.strip().lower() for c in raw_challenges.split(",") if c.strip())
        unknown = watch_challenges - VALID_CHALLENGES
        if unknown:
            raise ValueError(
                f"WATCH_CHALLENGES contains unknown challenge(s): {', '.join(sorted(unknown))}\n"
                f"  Valid values: {', '.join(sorted(VALID_CHALLENGES))}"
            )
    else:
        watch_challenges = frozenset()  # empty means "watch all"

    raw_flags = os.environ.get("SUBMIT_FLAGS", "")
    submit_flags = raw_flags.split() if raw_flags else []

    dry_run = os.environ.get("WATCHER_DRY_RUN", "").lower() in ("1", "true", "yes")

    return Config(
        token=token,
        channel_id=channel_id,
        guild_id=guild_id,
        watch_challenges=watch_challenges,
        submit_flags=submit_flags,
        dry_run=dry_run,
    )


# ── queue loading ──────────────────────────────────────────────────────────────


def load_queue(path: Path) -> dict[str, deque[str]]:
    if not path.exists():
        return {}

    with open(path, "rb") as f:
        data = tomllib.load(f)

    queue: dict[str, deque[str]] = {}
    for challenge, section in data.items():
        challenge = challenge.lower()
        if challenge not in VALID_CHALLENGES:
            raise ValueError(
                f"queue.toml: unknown challenge '{challenge}'\n"
                f"  Valid values: {', '.join(sorted(VALID_CHALLENGES))}"
            )
        tags = section.get("tags", [])
        if not tags:
            raise ValueError(f"queue.toml: [{challenge}] tags list is empty")
        queue[challenge] = deque(tags)

    return queue


# ── message parsing ────────────────────────────────────────────────────────────


def parse_result(content: str) -> EvalResult | None:
    m = RESULT_PATTERN.search(content)
    if not m:
        return None

    challenge = m.group("challenge").lower()
    if challenge not in VALID_CHALLENGES:
        return None

    return EvalResult(
        challenge=challenge,
        tag=m.group("tag"),
        errors=int(m.group("errors")),
        score=float(m.group("score")),
        speed=float(m.group("speed")),
    )


# ── subprocess runner ──────────────────────────────────────────────────────────


_PROJECT = "til-ai-2026"
_SERVICE_ACCOUNT = "svc-overflow@til-ai-2026.iam.gserviceaccount.com"
_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def _gcloud_token_sync() -> str:
    import google.auth
    import google.auth.impersonated_credentials
    import google.auth.transport.requests

    request = google.auth.transport.requests.Request()

    #source_creds, _ = google.auth.default(scopes=_SCOPES, quota_project_id=_PROJECT)
    source_creds, _ = google.auth.default(scopes=_SCOPES)
    source_creds.refresh(request)

    impersonated = google.auth.impersonated_credentials.Credentials(
        source_credentials=source_creds,
        target_principal=_SERVICE_ACCOUNT,
        target_scopes=_SCOPES,
        quota_project_id=_PROJECT,
    )
    impersonated.refresh(request)
    return impersonated.token


async def _gcloud_token(logger: logging.Logger) -> str | None:
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _gcloud_token_sync)
    except Exception as exc:
        logger.warning("Could not fetch gcloud token via google-auth: %s", exc)
        return None


_RUNNING_PROCS: set[asyncio.subprocess.Process] = set()
_SUBMIT_LOCK = asyncio.Lock()


def _terminate_all(logger: logging.Logger) -> None:
    for proc in list(_RUNNING_PROCS):
        if proc.returncode is None:
            logger.info("Terminating subprocess pid=%d", proc.pid)
            try:
                proc.terminate()
            except ProcessLookupError:
                pass


# ─── native submission (no bash, no submit.sh) ─────────────────────────────────

_REGION = "asia-southeast1"
_REGISTRY = f"{_REGION}-docker.pkg.dev"
_CHALLENGE_PORTS = {"asr": 5001, "cv": 5002, "noise": 5003, "nlp": 5004, "ae": 5005}


async def _run_streaming(
    cmd: list[str], logger: logging.Logger, *,
    stdin_data: bytes | None = None, label: str = "cmd", env: dict | None = None,
) -> int:
    logger.info(">> %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=env,
    )
    _RUNNING_PROCS.add(proc)
    try:
        if stdin_data is not None:
            assert proc.stdin is not None
            proc.stdin.write(stdin_data)
            proc.stdin.close()
        assert proc.stdout is not None
        async for raw_line in proc.stdout:
            logger.info("[%s] %s", label, raw_line.decode(errors="replace").rstrip())
        await proc.wait()
    finally:
        _RUNNING_PROCS.discard(proc)
    return proc.returncode if proc.returncode is not None else -1


async def run_submit(
    challenge: str, tag: str, flags: list[str], dry_run: bool, logger: logging.Logger
) -> None:
    team_id = os.environ.get("TEAM_ID", "overflow")
    build = "--build" in flags
    image_name = f"{team_id}-{challenge}"
    local_ref = f"{image_name}:{tag}"
    repo = f"{_REGISTRY}/{_PROJECT}/repo-til-26-{team_id}"
    remote_ref = f"{repo}/{image_name}:{tag}"
    port = _CHALLENGE_PORTS.get(challenge)
    if port is None:
        logger.error("Unknown challenge: %s", challenge)
        return

    if dry_run:
        logger.info(
            "[dry-run] would build=%s, tag/push %s -> %s, then upload to Vertex AI",
            build, local_ref, remote_ref,
        )
        return

    if _SUBMIT_LOCK.locked():
        logger.info("Queued behind running submission: %s:%s", challenge, tag)
    async with _SUBMIT_LOCK:
        await _run_submit_locked(challenge, build, local_ref, remote_ref, port, image_name, logger)


async def _run_submit_locked(
    challenge: str, build: bool, local_ref: str, remote_ref: str,
    port: int, image_name: str, logger: logging.Logger,
) -> None:
    docker = shutil.which("docker") or shutil.which("docker.exe")
    gcloud = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if not docker:
        logger.error("docker executable not found on PATH")
        return
    if not gcloud:
        logger.error("gcloud executable not found on PATH")
        return

    token = await _gcloud_token(logger)
    if not token:
        logger.error("Could not fetch service-account token — aborting submission")
        return

    if build:
        logger.info("=== docker build ===")
        rc = await _run_streaming(
            [docker, "build", "--platform", "linux/amd64", "-t", local_ref, challenge],
            logger, label="build",
        )
        if rc != 0:
            logger.error("docker build failed (rc=%d) — aborting", rc)
            return

    logger.info("=== docker login ===")
    rc = await _run_streaming(
        [docker, "login", "-u", "oauth2accesstoken", "--password-stdin", f"https://{_REGISTRY}"],
        logger, stdin_data=token.encode(), label="login",
    )
    if rc != 0:
        logger.error("docker login failed (rc=%d) — aborting", rc)
        return

    logger.info("=== docker tag + push ===")
    rc = await _run_streaming([docker, "tag", local_ref, remote_ref], logger, label="tag")
    if rc != 0:
        logger.error("docker tag failed (rc=%d) — aborting", rc)
        return
    rc = await _run_streaming([docker, "push", remote_ref], logger, label="push")
    if rc != 0:
        logger.error("docker push failed (rc=%d) — aborting", rc)
        return

    logger.info("=== gcloud ai models upload ===")
    gcloud_env = dict(os.environ)
    gcloud_env["CLOUDSDK_AUTH_ACCESS_TOKEN"] = token
    gcloud_env.pop("CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT", None)
    rc = await _run_streaming(
        [
            gcloud, "ai", "models", "upload",
            f"--project={_PROJECT}",
            f"--region={_REGION}",
            f"--display-name={image_name}",
            f"--container-image-uri={remote_ref}",
            "--container-health-route=/health",
            f"--container-predict-route=/{challenge}",
            f"--container-ports={port}",
            "--version-aliases=default",
        ],
        logger, label="upload", env=gcloud_env,
    )
    if rc == 0:
        logger.info("✓ Submitted %s as %s on %s", local_ref, image_name, _REGION)
    else:
        logger.error("gcloud ai models upload failed (rc=%d)", rc)


# ── discord client ─────────────────────────────────────────────────────────────


class WatcherClient(discord.Client):
    def __init__(
        self, config: Config, queue: dict[str, deque[str]], logger: logging.Logger
    ) -> None:
        super().__init__()
        self._config = config
        self._queue = queue
        self._queue_pos: dict[str, int] = {ch: 0 for ch in queue}
        self._logger = logger

    async def on_ready(self) -> None:
        challenges = ",".join(sorted(self._config.watch_challenges)) or "all"
        flags = " ".join(self._config.submit_flags) or "(none)"
        self._logger.info(
            "Logged in as %s — channel=%d%s challenges=%s submit_flags=%s dry_run=%s",
            self.user,
            self._config.channel_id,
            f" guild={self._config.guild_id}" if self._config.guild_id else "",
            challenges,
            flags,
            self._config.dry_run,
        )
        if self._queue:
            self._logger.info("Queue loaded:")
            for ch, tags in sorted(self._queue.items()):
                self._logger.info("  %s: %s (cycles)", ch, list(tags))
        else:
            self._logger.warning("No queue entries found — no submissions will be made")
        asyncio.create_task(self._stdin_loop())

    async def on_message(self, message: discord.Message) -> None:
        if message.channel.id != self._config.channel_id:
            return
        if self._config.guild_id and getattr(message.guild, "id", None) != self._config.guild_id:
            return
        if "evaluation has finished" not in message.content:
            return

        self._logger.debug("Raw message content: %r", message.content)

        result = parse_result(message.content)
        if result is None:
            self._logger.warning(
                "Matched trigger phrase but could not parse result from message id=%d", message.id
            )
            self._logger.warning("Raw content was: %r", message.content)
            return

        watched = self._config.watch_challenges
        if watched and result.challenge not in watched:
            self._logger.info(
                "Ignoring eval for %s (not in WATCH_CHALLENGES=%s)",
                result.challenge,
                ",".join(sorted(watched)),
            )
            return

        self._logger.info(
            "Eval result detected — challenge=%s evaluated_tag=%s errors=%d score=%.3f speed=%.3f",
            result.challenge,
            result.tag,
            result.errors,
            result.score,
            result.speed,
        )

        q = self._queue.get(result.challenge)
        if q is None:
            self._logger.warning(
                "No queue entry for %s — add [%s] to queue.toml to enable auto-submit",
                result.challenge,
                result.challenge,
            )
            return

        self._fire(result.challenge)

    def _fire(self, challenge: str) -> None:
        q = self._queue.get(challenge)
        if q is None:
            self._logger.warning("No queue entry for %s", challenge)
            return

        n = len(q)
        pos = self._queue_pos[challenge]
        next_tag = q[0]
        q.rotate(-1)
        self._queue_pos[challenge] = (pos + 1) % n
        self._logger.info(
            "Submitting %s:%s (queue position %d/%d)", challenge, next_tag, pos + 1, n
        )
        asyncio.create_task(
            run_submit(challenge, next_tag, self._config.submit_flags, self._config.dry_run, self._logger)
        )

    async def _stdin_loop(self) -> None:
        loop = asyncio.get_event_loop()
        self._logger.info(
            "Manual trigger ready — press Enter to submit all queued challenges, "
            "or type a challenge name (e.g. 'ae') then Enter for a specific one."
        )
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:  # EOF
                break
            text = line.strip().lower()
            if text == "":
                targets = sorted(self._queue)
                if not targets:
                    self._logger.warning("Queue is empty — nothing to submit")
                else:
                    for ch in targets:
                        self._fire(ch)
            elif text in VALID_CHALLENGES:
                self._fire(text)
            else:
                self._logger.warning(
                    "Unknown input %r — type a challenge name (%s) or press Enter for all",
                    text,
                    ", ".join(sorted(VALID_CHALLENGES)),
                )


# ── entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logger = logging.getLogger("discord_watcher")

    # One-shot mode: python discord_watcher.py --submit CHALLENGE [TAG]
    args = sys.argv[1:]
    if args and args[0] == "--submit":
        if len(args) < 2 or args[1].lower() not in VALID_CHALLENGES:
            logger.error("Usage: python discord_watcher.py --submit CHALLENGE [TAG]")
            sys.exit(2)
        challenge = args[1].lower()
        tag = args[2] if len(args) >= 3 else "latest"
        load_dotenv(REPO_ROOT / ".env")
        raw_flags = os.environ.get("SUBMIT_FLAGS", "")
        flags = raw_flags.split() if raw_flags else []
        try:
            asyncio.run(run_submit(challenge, tag, flags, dry_run=False, logger=logger))
        except KeyboardInterrupt:
            logger.info("Interrupted — terminating subprocesses")
            _terminate_all(logger)
        return

    try:
        config = load_config()
        queue = load_queue(REPO_ROOT / "queue.toml")
    except ValueError as exc:
        logger.error("Configuration error:\n%s", exc)
        sys.exit(1)

    client = WatcherClient(config, queue, logger)
    try:
        client.run(config.token)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received — terminating subprocesses")
    finally:
        _terminate_all(logger)


if __name__ == "__main__":
    main()
