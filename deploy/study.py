"""One door to the diff participant app, wherever it is served.

    uv run python deploy/study.py local [--fake] [--direct] [--purpose smoke|participant]
    uv run python deploy/study.py stop
    uv run python deploy/study.py status
    uv run python deploy/study.py links [P1 P2 …]
    uv run python deploy/study.py vercel [--prod] [--purpose smoke|participant]

Two places can serve the same app from the same materials (researcher, 2026-11-23, when the Neon
quota kept Vercel down):

- **local**: this machine, in a tmux session, sessions in SQLite. Reached through an SSH tunnel as
  ``http://localhost:<port>/`` (the default; the app listens on loopback only), or with ``--direct``
  by this machine's address, ``http://<ip>:<port>/``, where entry is by personal link because the
  lab app has no access code.
- **vercel**: ``deploy/vercel/stage.py`` and ``vercel deploy``, sessions in Neon.

Both read the setup, cues, scene parts, items and videos named once, in ``deploy/vercel/stage.py``.
Local serves the working tree; Vercel serves the commit at HEAD. Phase 2's videos are served
locally as the web copies Vercel serves, when they have been made, so both show the same picture.
The two keep separate session stores: a participant starts and ends in one place.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "deploy" / "vercel"))
import stage  # type: ignore[import-not-found]  # noqa: E402  (the one list of what the study serves)

TMUX = "diff-local"
PORT = 8782
# Logs, the smoke store and running.json; never dated, so a restart resumes its sessions.
STATE = REPO / "artifacts/diff/local"
RUNNING = STATE / "running.json"
SITE = REPO / "artifacts/diff/vercel/site"
PRODUCTION = "https://concap-exp.vercel.app"


def machine_address() -> str:
    """The address this machine is reached at: the one its default route leaves from."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("192.0.2.1", 9))  # a documentation address; nothing is sent
        return str(probe.getsockname()[0])


def load_key() -> bool:
    """ANTHROPIC_API_KEY from the shell, else from the repository's untracked .env."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    env = REPO / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip().removeprefix("export ").strip() == "ANTHROPIC_API_KEY" and value.strip():
                os.environ["ANTHROPIC_API_KEY"] = value.strip().strip("'\"")
                return True
    return False


def answers(url: str, timeout: float = 5.0) -> int | None:
    """The HTTP status a GET of ``url`` gets, or None when nothing answers."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            return int(response.status)
    except urllib.error.HTTPError as error:
        return int(error.code)
    except (urllib.error.URLError, OSError):
        return None


def tmux_alive() -> bool:
    found = subprocess.run(["tmux", "has-session", "-t", TMUX], capture_output=True, check=False)
    return found.returncode == 0


def running() -> dict[str, Any] | None:
    return json.loads(RUNNING.read_text(encoding="utf-8")) if RUNNING.is_file() else None


def phase2_video(path: str) -> str:
    """The file to serve for a phase-2 video: Vercel's web copy when it exists, else the original."""
    copy = stage.web_copy(path, SITE.parent / "web")
    return str(copy.relative_to(REPO)) if copy.is_file() else path


def serve_command(args: argparse.Namespace, host: str) -> list[str]:
    command = [
        sys.executable, "-m", "diff", "live", "serve",
        "--phase1-setup", str(stage.SETUP),
        "--sources", str(stage.SOURCES),
        "--phase2-scene", str(stage.SCENES),
        "--items", str(stage.ITEMS),
        "--purpose", args.purpose,
        "--entry", args.entry,
        "--sessions", str(args.sessions),
        "--log-dir", str((STATE / args.purpose).relative_to(REPO)),
        "--host", host,
        "--port", str(args.port),
    ]  # fmt: skip
    for video_id, path in stage.VIDEOS:
        command += ["--video", f"{video_id}={phase2_video(path)}"]
    if args.fake:
        command.append("--fake-worker")
    return command


def how_to_open(state: dict[str, Any]) -> str:
    port, address = state["port"], state["address"]
    fake = ", FAKE captions" if state["fake"] else ""
    lines = [f"local app: {state['purpose']}, entry by {state['entry']}{fake}"]
    if state["direct"]:
        lines.append(f"  open  http://{address}:{port}/   (from a network that reaches this machine)")
    else:
        user = os.environ.get("USER", "<user>")
        lines += [
            "  on your own computer, keep this running:",
            f"    ssh -N -L {port}:127.0.0.1:{port} {user}@{address}",
            f"  then open  http://localhost:{port}/",
        ]
    if state["entry"] == "link":
        lines.append("  each participant needs a personal link:  uv run python deploy/study.py links P1 P2")
    return "\n".join(lines)


def local(args: argparse.Namespace) -> int:
    args.entry = args.entry or ("link" if args.direct or args.purpose == "participant" else "id")
    address = machine_address()
    host = address if args.direct else "127.0.0.1"
    if not args.fake and not load_key():
        raise SystemExit(
            "no ANTHROPIC_API_KEY in the shell or in .env at the repository root.\n"
            "  put  ANTHROPIC_API_KEY=…  in .env (untracked), or pass --fake for placeholder captions"
        )
    command = serve_command(args, host)
    if args.foreground:  # what the tmux session runs; the key is in this process's environment only
        os.chdir(REPO)
        os.execvpe(command[0], command, {**os.environ, "PYTHONPATH": str(REPO / "src")})
    if tmux_alive() or answers(f"http://{host}:{args.port}/") is not None:
        raise SystemExit(f"something already serves port {args.port}; run `deploy/study.py stop` first")
    STATE.mkdir(parents=True, exist_ok=True)
    again = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--foreground"]
    log = STATE / "serve.log"
    shell = f"cd {shlex.quote(str(REPO))} && {shlex.join(again)} 2>&1 | tee {shlex.quote(str(log))}"
    subprocess.run(["tmux", "new-session", "-d", "-s", TMUX, shell], check=True)
    state = {
        "address": address, "port": args.port, "direct": args.direct, "purpose": args.purpose,
        "entry": args.entry, "fake": args.fake,
        "base": f"http://{address if args.direct else 'localhost'}:{args.port}",
        "store": str(_store(args.purpose)),
    }  # fmt: skip
    for _ in range(90):
        if answers(f"http://{host}:{args.port}/", timeout=2.0) == 200:
            RUNNING.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            print(how_to_open(state))
            print(f"  server output: tmux attach -t {TMUX}   (or {log.relative_to(REPO)})")
            return 0
        if not tmux_alive():
            break
        time.sleep(1.0)
    subprocess.run(["tmux", "kill-session", "-t", TMUX], capture_output=True, check=False)
    said = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    tail = "\n".join(said.splitlines()[-15:])
    raise SystemExit(f"the local app did not start:\n{tail}")


def _store(purpose: str) -> Path:
    """Where ``diff live serve`` keeps this purpose's sessions (its own defaults)."""
    if purpose == "participant":
        return Path("artifacts/diff/participants.sqlite")
    return (STATE / purpose / "smoke-sessions.sqlite").relative_to(REPO)


def stop(_: argparse.Namespace) -> int:
    was = tmux_alive()
    subprocess.run(["tmux", "kill-session", "-t", TMUX], capture_output=True, check=False)
    RUNNING.unlink(missing_ok=True)
    print("local app stopped" if was else "the local app was not running")
    return 0


def status(_: argparse.Namespace) -> int:
    state = running()
    if state and tmux_alive():
        host = state["address"] if state["direct"] else "127.0.0.1"
        code = answers(f"http://{host}:{state['port']}/")
        print(how_to_open(state) if code == 200 else f"local app: tmux is up but the port answers {code}")
    else:
        print("local app: not running   (start: uv run python deploy/study.py local)")
    code = answers(PRODUCTION + "/", timeout=15.0)
    deploy = SITE / "deploy.json"
    staged = json.loads(deploy.read_text(encoding="utf-8")) if deploy.is_file() else {}
    known = {None: "does not answer from this machine", 500: "answers 500 (database: check Neon's quota)"}
    health = known.get(code, f"answers {code}")
    print(f"vercel: {PRODUCTION} {health}")
    if staged:
        commit = staged["commit"][:7]
        print(f"  last staged: {staged['purpose']}, entry by {staged['entry']}, commit {commit}")
    return 0


def links(args: argparse.Namespace) -> int:
    state = running()
    if state is None:
        raise SystemExit("the local app is not running; for Vercel's links use deploy/vercel/links.py")
    command = [sys.executable, "-m", "diff", "links", *args.participants]
    command += ["--store", state["store"], "--base", state["base"]]
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    return subprocess.run(command, cwd=REPO, env=env, check=False).returncode


def vercel(args: argparse.Namespace) -> int:
    site = SITE.relative_to(REPO)
    staging = [sys.executable, "deploy/vercel/stage.py", "--out", str(site), "--upload"]
    staging += ["--purpose", args.purpose]
    deploying = ["vercel", "deploy", *(["--prod"] if args.prod else [])]
    if not args.yes:
        print("this would run (add --yes to run it):")
        print(f"  {shlex.join(staging)}\n  cd {site} && {shlex.join(deploying)}")
        return 0
    subprocess.run(staging, cwd=REPO, check=True)
    return subprocess.run(deploying, cwd=SITE, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("local", help="serve on this machine (SSH tunnel, or --direct by address)")
    start.add_argument("--purpose", choices=("smoke", "participant"), default="smoke")
    start.add_argument(
        "--entry", choices=("id", "link"), help="default: id; link with --direct or participants"
    )
    start.add_argument("--direct", action="store_true", help="listen on this machine's address, not loopback")
    start.add_argument("--fake", action="store_true", help="placeholder captions, no API key or calls")
    start.add_argument("--sessions", type=int, default=2, help="sessions at once")
    start.add_argument("--port", type=int, default=PORT)
    start.add_argument("--foreground", action="store_true", help="serve in this terminal, no tmux")
    start.set_defaults(handler=local)

    commands.add_parser("stop", help="stop the local app").set_defaults(handler=stop)
    commands.add_parser("status", help="what serves the study now").set_defaults(handler=status)

    link = commands.add_parser("links", help="personal links for the running local app")
    link.add_argument("participants", nargs="*", help="P1 P2 …; none lists every link")
    link.set_defaults(handler=links)

    cloud = commands.add_parser("vercel", help="stage HEAD and deploy it to Vercel")
    cloud.add_argument("--purpose", choices=("smoke", "participant"), default="smoke")
    cloud.add_argument("--prod", action="store_true", help="the participants' address, not a preview")
    cloud.add_argument("--yes", action="store_true", help="run it; without this the commands are printed")
    cloud.set_defaults(handler=vercel)

    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
