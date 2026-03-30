from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency in Phase 1
    yaml = None


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML config file."""
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")

    if config_path.suffix.lower() == ".json":
        return json.loads(text)

    if config_path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to load YAML configs. "
                "Install dependencies from requirements.txt first."
            )
        data = yaml.safe_load(text)
        return data or {}

    raise ValueError(f"Unsupported config format: {config_path.suffix}")
