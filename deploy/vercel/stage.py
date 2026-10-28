"""Stage the diff participant app as a Vercel site, from a committed tree.

    uv run python deploy/vercel/stage.py --out artifacts/diff/vercel/site [--upload]

What goes where (Vercel limits: 4.5 MB per function response, 500 MB per Python bundle, 100 MB
per Hobby CLI upload):
- the ``diff`` package and the ``dpo`` modules it imports, as committed at HEAD (read with
  ``git show``, so uncommitted edits in the tree never reach a deploy);
- the setup, the phase-2 cues and the items, and every still and mask the setup names, at their
  repository paths (the server reads masks and stills itself);
- the videos go to the project's Vercel Blob store with ``--upload``, named by their content's
  SHA-256 so no URL carries a clip key. ``blob-manifest.json`` beside the site remembers what is
  uploaded. Every video goes as a web copy (``web/`` beside the site, made once per source):
  phase 2's re-encoded (``WEB_COPY``), phase 1's remuxed with their streams untouched
  (``FAST_START``).

The study link's access code is kept in ``access-code`` beside the site (made on the first stage,
replaced with ``--new-code``); it goes into ``deploy.json``, never into a served file.

The setup is checked with ``setup_problems`` against the full repository first; the deployed app
does not re-check, because the videos are not in its bundle. ``.vercel/`` (the project link) in
``--out`` survives a re-stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SETUP = Path("artifacts/diff/phase1/setup-draft.json")
SOURCES = Path("data/phase2/phase2-cues.json")
ITEMS = Path("artifacts/diff/robustness-20261020/items.json")
VIDEOS = (
    ("viewing-1", "data/phase2/viewing-1.mp4"),
    ("viewing-2", "data/phase2/viewing-2.mp4"),
    ("viewing-3", "data/phase2/viewing-3.mp4"),
)
# Pinned to uv.lock; the lab's other dependencies (torch, transformers) are not needed to serve.
REQUIREMENTS = (
    "anthropic==2.3.0",
    "fastapi==0.139.2",
    "numpy==2.4.6",
    "pillow==12.3.0",
    "psycopg[binary]==3.3.6",
    "psycopg-pool==3.3.3",
)
VERCEL_JSON = {
    "$schema": "https://openapi.vercel.sh/vercel.json",
    # Beside the Neon database (sin1); Vercel offers Neon no nearer region to Seoul.
    "regions": ["sin1"],
    "functions": {"app.py": {"maxDuration": 60}},
}
# The project link and the CLI's pulled env survive a re-stage; neither is ever uploaded.
LINK_STATE = (".vercel", ".env.local", ".gitignore")
VERCELIGNORE = (".env*", "__pycache__", "*.pyc", ".agents", ".claude", "skills-lock.json")
# Phase 2's viewing videos are 1080p60 at 5.8 Mbps, 665 MB a participant, and Hobby's Blob
# allows 10 GB of downloads a month (over it, Blob is shut for 30 days). 720p capped at 2 Mbps
# is about 225 MB; frame rate and timing are kept, and the sound is copied, not re-encoded.
WEB_COPY = (
    "-vf", "scale=-2:720", "-c:v", "libx264", "-preset", "slow", "-crf", "23",
    "-maxrate", "2000k", "-bufsize", "4000k", "-pix_fmt", "yuv420p",
    "-c:a", "copy", "-movflags", "+faststart",
)  # fmt: skip
# Phase 1's clips have their index (moov) after the media, so a browser fetches the file's end
# before it can show a frame. A remux moves it to the front; the streams are copied unchanged.
FAST_START = ("-map", "0", "-c", "copy", "-movflags", "+faststart")
CODE_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"
MEDIA_SUFFIXES = (".png", ".mp4", ".webm")


def access_code() -> str:
    """Six characters a participant can type from a printed link: letters and digits, one of each
    at least, nothing that reads as another (0/o, 1/l/i). The gate compares in lower case."""
    while True:
        code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(6))
        if any(c.isdigit() for c in code) and any(c.isalpha() for c in code):
            return code


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout


def server_modules() -> list[Path]:
    """Every ``diff`` and ``dpo`` source file the server imports, relative to ``src``."""
    code = (
        "import sys, diff.participant_server, diff.phase1_flow, diff.phase1_tweak, diff.phase1_usable,"
        " diff.claude_writer, diff.live_server, diff.phase1_setup, diff.workbook_instrument;"
        "print('\\n'.join(sorted(m.__file__ for n, m in sys.modules.items()"
        " if n.split('.')[0] in ('diff', 'dpo') and getattr(m, '__file__', None))))"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, cwd=REPO)
    return [Path(line).relative_to(REPO / "src") for line in out.stdout.split()]


def setup_files(setup: Any) -> set[str]:
    """Every repository-relative media file the setup names."""
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for inner in value.values():
                walk(inner)
        elif isinstance(value, list):
            for inner in value:
                walk(inner)
        elif isinstance(value, str) and value.endswith(MEDIA_SUFFIXES) and not value.startswith("/"):
            found.add(value)

    walk(setup)
    return found


def copy(relative: Path | str, out: Path) -> None:
    target = out / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / relative, target)


def copy_committed(relative: Path, out: Path, commit: str) -> None:
    target = out / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    blob = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{commit}:{relative.as_posix()}"], check=True, capture_output=True
    )
    target.write_bytes(blob.stdout)


def blob_token(out: Path) -> str:
    """The store's read-write token, which ``vercel blob create-store`` put in ``.env.local``."""
    for line in (out / ".env.local").read_text().splitlines():
        name, _, value = line.partition("=")
        if name == "BLOB_READ_WRITE_TOKEN":
            return value.strip().strip('"')
    raise SystemExit(f"no BLOB_READ_WRITE_TOKEN in {out / '.env.local'}: run `vercel env pull` there")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def web_copy(path: str, web: Path) -> Path:
    """Where a video's web copy is, named after its source's digest."""
    return web / f"{digest(REPO / path)[:32]}.mp4"


def make_web_copy(path: str, web: Path, options: tuple[str, ...]) -> Path:
    target = web_copy(path, web)
    if not target.exists():
        print(f"making a web copy of {path}", flush=True)
        web.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial.mp4")
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(REPO / path)]
        subprocess.run([*command, *options, str(partial)], check=True)
        partial.rename(target)
    return target


def upload(source: Path, manifest: dict[str, str], token: str, cwd: Path) -> str:
    sha = digest(source)
    if sha not in manifest:
        pathname = f"m/{sha[:32]}{source.suffix}"
        print(f"uploading {source.relative_to(REPO)} → {pathname}", flush=True)
        # Run outside the site, whose .env.local also holds an OIDC token the CLI would prefer.
        reply = subprocess.run(
            [
                "vercel",
                "blob",
                "put",
                str(source),
                "--pathname",
                pathname,
                "--access",
                "public",
                "--force",
                "--non-interactive",
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            env={**os.environ, "BLOB_READ_WRITE_TOKEN": token},
        )
        urls = [word for word in (reply.stdout + reply.stderr).split() if word.startswith("https://")]
        if not urls:
            raise SystemExit(f"no URL in vercel blob put's reply:\n{reply.stdout}{reply.stderr}")
        manifest[sha] = urls[0]
    return manifest[sha]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--purpose", choices=("smoke", "participant"), default="smoke")
    parser.add_argument("--sessions", type=int, default=4, help="concurrent sessions per instance")
    parser.add_argument("--upload", action="store_true", help="upload missing videos to Blob")
    parser.add_argument("--closed", action="store_true", help="every request gets the closed page")
    parser.add_argument("--new-code", action="store_true", help="replace the access code (closes old links)")
    parser.add_argument("--dirty", action="store_true", help="stage uncommitted deploy/vercel files")
    args = parser.parse_args()

    head = git("rev-parse", "HEAD").strip()
    dirty = git("status", "--porcelain", "--", "deploy/vercel").strip()
    if dirty and not args.dirty:
        raise SystemExit(f"commit these first, a deploy is always a commit:\n{dirty}")
    commit = head + ("+deploy-dirty" if dirty else "")
    committed = set(git("ls-tree", "-r", "--name-only", head, "--", "src/diff", "src/dpo").split())
    modules = [module for module in server_modules() if f"src/{module.as_posix()}" in committed]

    sys.path.insert(0, str(REPO / "src"))
    import diff.workbook_instrument as workbook_instrument
    from diff.phase1_setup import read_setup, setup_problems

    setup = read_setup(REPO / SETUP)
    problems = setup_problems(setup, args.purpose, root=REPO, instrument=workbook_instrument)
    if problems:
        raise SystemExit(f"the setup cannot serve {args.purpose} sessions:\n  " + "\n  ".join(problems))

    out: Path = args.out.resolve()
    kept = out.with_name(out.name + ".kept")
    shutil.rmtree(kept, ignore_errors=True)
    kept.mkdir(parents=True)
    for name in LINK_STATE:
        if (out / name).exists():
            shutil.move(out / name, kept / name)
    shutil.rmtree(out, ignore_errors=True)
    kept.rename(out)

    package_data = [
        Path(name).relative_to("src")
        for name in committed
        if name.startswith("src/diff/") and not name.endswith(".py")
    ]
    for module in sorted(set(modules) | set(package_data)):
        copy_committed(Path("src") / module, out, head)
    for staged in ("diff", "dpo"):
        shutil.move(out / "src" / staged, out / staged)
    (out / "src").rmdir()

    media = sorted(setup_files(setup))
    clips = [path for path in media if path.endswith((".mp4", ".webm"))]
    for path in media:
        if path not in clips:
            copy(path, out)
    for path in (SETUP, SOURCES, ITEMS):
        copy(path, out)

    web = out.with_name("web")
    copies = {path: FAST_START for path in clips} | {path: WEB_COPY for _, path in VIDEOS}
    blobbed = {
        path: make_web_copy(path, web, options) if args.upload else web_copy(path, web)
        for path, options in copies.items()
    }
    manifest_path = out.with_name("blob-manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if args.upload:
        token = blob_token(out)
        for source in blobbed.values():
            upload(source, manifest, token, out.parent)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    urls = {}
    for path, source in blobbed.items():
        if source.exists() and (sha := digest(source)) in manifest:
            urls[path] = manifest[sha]
    missing = [path for path in blobbed if path not in urls]

    code_path = out.with_name("access-code")
    if args.new_code or not code_path.exists():
        code_path.write_text(access_code() + "\n")
        code_path.chmod(0o600)
    deploy = {
        "commit": commit,
        "access_code": code_path.read_text().strip(),
        "open": not args.closed,
        "purpose": args.purpose,
        "sessions": args.sessions,
        "phase1_setup": str(SETUP),
        "sources": str(SOURCES),
        "items": str(ITEMS),
        "videos": [list(video) for video in VIDEOS],
        "media": urls,
    }
    (out / "deploy.json").write_text(json.dumps(deploy, indent=2) + "\n")
    for name in ("app.py", "pg_store.py"):  # pull.py stays here: it runs in the lab
        shutil.copy2(HERE / name, out / name)
    (out / "requirements.txt").write_text("\n".join(REQUIREMENTS) + "\n")
    (out / ".python-version").write_text("3.12\n")
    (out / ".vercelignore").write_text("\n".join(VERCELIGNORE) + "\n")
    (out / "vercel.json").write_text(json.dumps(VERCEL_JSON, indent=2) + "\n")

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and ".vercel" not in p.parts)
    print(f"staged {commit} ({args.purpose}) in {out}: {size / 1e6:.1f} MB")
    if missing:
        print(f"{len(missing)} of {len(blobbed)} videos are not on Blob yet (pass --upload):", *missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
