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


def test_data_dirs_have_no_negation_hole():
    """`data/` is excluded WHOLESALE, with no .gitkeep exception.

    A `!data/raw/.gitkeep` exception forces git to descend into an excluded
    directory; a trailing-slash pattern then does not match the files inside,
    which is precisely how the WhatsApp exports became addable. Assert the
    exception has not crept back in.
    """
    # Strip comments first -- the explanatory comment above literally contains
    # `!data/raw/.gitkeep`, and matching that would fail the test on prose.
    patterns = [
        ln.strip()
        for ln in (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    holes = [p for p in patterns if p.startswith(("!data/", "!.secrets/"))]
    assert not holes, f"negation inside an excluded data dir re-opens the leak: {holes}"
    for p in ("data/raw/.gitkeep", "data/derived/.gitkeep", ".secrets/.gitkeep"):
        assert _ignored(p), f"{p} must be ignored -- no placeholder should be tracked"


def test_runtime_creates_the_data_dirs():
    """Nothing holds the directories' place in git, so code must make them."""
    import inspect

    from mehullm.pipeline import build_sft, neutralize

    assert "mkdir(parents=True, exist_ok=True)" in inspect.getsource(build_sft.build)
    assert "mkdir(parents=True, exist_ok=True)" in inspect.getsource(
        neutralize.DraftCache.__init__
    )


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
