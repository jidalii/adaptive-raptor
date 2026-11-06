import json

import yaml

from adaptive_raptor.run import EXIT_INVALID, EXIT_NOT_IMPLEMENTED, EXIT_OK, main
from tests.conftest import EXAMPLE_CONFIGS, ROOT

H1 = ROOT / "configs/experiments/h1.yaml"


def test_validate_only_all_examples(capsys):
    for path in EXAMPLE_CONFIGS:
        assert main(["--config", str(path), "--validate-only"]) == EXIT_OK, path


def test_validate_only_has_no_side_effects(tmp_path):
    out = tmp_path / "out"
    code = main(["--config", str(H1), "--validate-only", "--set", f"experiment.output_dir={out}"])
    assert code == EXIT_OK
    assert not out.exists()


def test_invalid_config_exit_code(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("experiment: {name: x, type: h1_optimizer}\n")
    assert main(["--config", str(bad), "--validate-only"]) == EXIT_INVALID
    assert "optimizer.enabled" in capsys.readouterr().err


def test_missing_file_exit_code(tmp_path):
    assert main(["--config", str(tmp_path / "nope.yaml"), "--validate-only"]) == EXIT_INVALID


def test_placeholder_run_reports_not_implemented(tmp_path, capsys):
    out = tmp_path / "out"
    code = main(["--config", str(H1), "--set", f"experiment.output_dir={out}"])
    assert code == EXIT_NOT_IMPLEMENTED
    (run_dir,) = list((out / "h1_router_f100").iterdir())
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "not_implemented"
    assert status["planned_steps"]
    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    assert resolved["experiment"]["type"] == "h1_optimizer"
    assert resolved["generation"]["temperature"] == 0.0  # defaults are saved
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    assert meta["overrides"] == [f"experiment.output_dir={out}"]
    assert "not_implemented" in capsys.readouterr().out


def test_runs_never_overwrite(tmp_path):
    out = tmp_path / "out"
    args = ["--config", str(H1), "--set", f"experiment.output_dir={out}"]
    main(args)
    main(args)
    assert len(list((out / "h1_router_f100").iterdir())) == 2
