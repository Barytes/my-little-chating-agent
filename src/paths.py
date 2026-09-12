"""Project-root paths. Config stays at the repo root; runtime files go under runs/."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROVIDERS_CONFIG = PROJECT_ROOT / "providers.json"
ENV_FILE = PROJECT_ROOT / ".env"

RUNS_DIR = PROJECT_ROOT / "runs"
TASKS_DIR = RUNS_DIR / "tasks"
TRAJECTORIES_DIR = RUNS_DIR / "trajectories"


def ensure_runtime_dirs() -> None:
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)
