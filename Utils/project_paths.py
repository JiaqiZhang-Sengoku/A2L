"""Resolve every project-relative path from this movable project's root."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path):
    """Resolve a path independently of the shell's current working directory."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def relative_path(path):
    """Keep saved configurations portable; retain explicitly external absolute paths."""
    resolved = resolve_path(path)
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)
