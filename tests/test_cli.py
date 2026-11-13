import json

import yaml

from adaptive_raptor.run import EXIT_INVALID, EXIT_NOT_IMPLEMENTED, EXIT_OK, main
from tests.conftest import EXAMPLE_CONFIGS, ROOT

H1 = ROOT / "configs/experiments/h1.yaml"
PLACEHOLDER = ROOT / "configs/experiments/ablation.yaml"  # a runner that is still a placeholder


def test_validate_only_all_examples(capsys):
    from tests.test_config import TEMPLATE_FILL

    for path in EXAMPLE_CONFIGS:
        sets = [x for s in TEMPLATE_FILL.get(path.stem, []) for x in ("--set", s)]
        assert main(["--config", str(path), "--validate-only", *sets]) == EXIT_OK, path


def test_validate_only_has_no_side_effects(tmp_path):
    out = tmp_path / "out"
    from tests.test_config import TEMPLATE_FILL

    sets = [x for s in TEMPLATE_FILL["h1"] for x in ("--set", s)]
    code = main(
        ["--config", str(H1), "--validate-only", "--set", f"experiment.output_dir={out}", *sets]
    )
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
    code = main(["--config", str(PLACEHOLDER), "--set", f"experiment.output_dir={out}"])
    assert code == EXIT_NOT_IMPLEMENTED
    (run_dir,) = list((out / "ablation_strategy_budget").iterdir())
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "not_implemented"
    assert status["planned_steps"]
    resolved = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    assert resolved["experiment"]["type"] == "strategy_budget_ablation"
    assert resolved["generation"]["temperature"] == 0.0  # defaults are saved
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    assert meta["overrides"] == [f"experiment.output_dir={out}"]
    assert "not_implemented" in capsys.readouterr().out


def test_runs_never_overwrite(tmp_path):
    out = tmp_path / "out"
    args = ["--config", str(PLACEHOLDER), "--set", f"experiment.output_dir={out}"]
    main(args)
    main(args)
    assert len(list((out / "ablation_strategy_budget").iterdir())) == 2


def test_run_writes_timestamped_log_file(tmp_path, capsys):
    import re

    out = tmp_path / "out"
    main(["--config", str(PLACEHOLDER), "--set", f"experiment.output_dir={out}"])
    (run_dir,) = list((out / "ablation_strategy_budget").iterdir())
    log_file = tmp_path / "logs" / f"ablation_strategy_budget_{run_dir.name}.log"
    text = log_file.read_text()
    assert re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\dZ INFO ", text)
    assert "run finished: status=not_implemented" in text
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    assert meta["log_file"] == str(log_file)
