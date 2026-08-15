"""Enforce that nothing derived from personal chats can be committed.

This exists because the first .gitignore silently FAILED. It used
`data/raw/` + `!data/raw/.gitkeep`, and git cannot re-include a file whose
parent directory is excluded -- the negation forced git to descend, and a
trailing-slash pattern matches directories rather than the files inside. Every
WhatsApp export sat untracked-and-addable, one `git add -A` from a public repo.

A .gitignore is a security control here, so it gets tested like one.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

MUST_BE_IGNORED = [
    # raw corpus
    "data/raw/WhatsApp Chat with Someone.txt",
    "data/raw/nested/deep/chat.txt",
    "data/raw/anything.zip",
    # derived artifacts
    "data/derived/pairs.jsonl",
    "data/derived/sft_pairs.jsonl",
    "data/derived/audit_sample.txt",
    "data/derived/name_map.json",
    "data/derived/contacts.json",
    "data/derived/census.json",
    "data/derived/draft_cache.db",
    "data/derived/draft_cache.db-wal",
    "data/derived/draft_cache.db-shm",
    # secrets
    ".env",
    ".env.local",
    "credentials.json",
    "token.json",
    ".secrets/gmail_token.json",   # OAuth cache written at runtime
    ".secrets/anything.json",
    "client_secret_123.json",
    "gcp-oauth.keys.json",
    # weights
    "voice-q4km.gguf",
    "merged_fp16/model.safetensors",
    "outputs/checkpoint-200/adapter_model.bin",
    # editors
    ".qodo/agents/x.yaml",
    ".venv/pyvenv.cfg",
]

MUST_NOT_BE_IGNORED = [
    "src/mehullm/pipeline/whatsapp_parser.py",
    "tests/test_whatsapp_parser.py",
    "tests/fixtures/android_hinglish.txt",
    "pyproject.toml",
    "README.md",
    ".gitignore",
    ".env.example",
    "docs/PLAN.md",
]


def _ignored(path: str) -> bool:
    r = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", path],
        cwd=REPO, capture_output=True,
    )
    return r.returncode == 0


pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


@pytest.mark.parametrize("path", MUST_BE_IGNORED)
def test_sensitive_paths_are_ignored(path):
    assert _ignored(path), (
        f"{path!r} is NOT ignored -- it could be committed to a public repo."
    )


@pytest.mark.parametrize("path", MUST_NOT_BE_IGNORED)
def test_source_paths_are_not_ignored(path):
    assert not _ignored(path), f"{path!r} is ignored but must be committed."


def test_gitkeep_survives_the_directory_exclusion():
    """The negation must actually work -- that is the whole reason for /* form."""
    assert not _ignored("data/raw/.gitkeep")
    assert not _ignored("data/derived/.gitkeep")


def test_no_sensitive_file_is_currently_tracked():
    """The real invariant: check what git ACTUALLY has, not just the patterns."""
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True
    ).stdout.splitlines()
    bad = [
        f
        for f in out
        if (f.startswith("data/") and not f.endswith(".gitkeep"))
        or f.endswith((".gguf", ".safetensors", ".db"))
        or f in {".env", "credentials.json", "token.json"}
    ]
    assert not bad, f"sensitive files are tracked by git: {bad}"
