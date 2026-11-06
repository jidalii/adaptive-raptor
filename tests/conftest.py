import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIGS = sorted((ROOT / "configs" / "experiments").glob("*.yaml"))
SMOKE = ROOT / "configs" / "experiments" / "smoke_mock.yaml"
REAL_QASPER = ROOT / "data" / "processed" / "qasper" / "v0.3"


@pytest.fixture
def minimal_config_dict(tmp_path):
    return {
        "experiment": {
            "name": "unit",
            "type": "raptor_baseline",
            "output_dir": str(tmp_path / "outputs"),
        }
    }


@pytest.fixture
def no_api_keys(monkeypatch):
    for env in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    # keep run.py's load_dotenv() from re-reading a developer's real .env
    monkeypatch.setattr("adaptive_raptor.run.load_dotenv", lambda *a, **k: False)


def smoke_overrides(tmp_path: Path) -> list[str]:
    return [
        f"experiment.output_dir={tmp_path / 'out'}",
        f"raptor.index_dir={tmp_path / 'idx'}",
        f"runtime.cache_dir={tmp_path / 'cache'}",
    ]


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
