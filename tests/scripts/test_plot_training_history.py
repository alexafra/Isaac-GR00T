import json
from argparse import Namespace

from scripts.analysis_tools import plot_training_history


def test_training_and_validation_loss_use_contrasting_colors(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoint-20"
    checkpoint_dir.mkdir(parents=True)
    history = [
        {"step": 10, "loss": 0.4, "grad_norm": 1.0, "learning_rate": 1e-4},
        {"step": 10, "eval_loss": 0.5},
        {"step": 20, "loss": 0.3, "grad_norm": 0.8, "learning_rate": 5e-5},
        {"step": 20, "eval_loss": 0.35},
    ]
    with (checkpoint_dir / "trainer_state.json").open("w") as state_file:
        json.dump({"log_history": history}, state_file)

    output_dir = tmp_path / "analysis"
    monkeypatch.setattr(
        plot_training_history,
        "parse_args",
        lambda: Namespace(run_dir=run_dir, output_dir=output_dir, smooth_window=2),
    )
    saved_figures = []

    def capture_figure(figure, *_args, **_kwargs):
        saved_figures.append(figure)

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", capture_figure)

    plot_training_history.main()

    loss_lines = {line.get_label(): line for line in saved_figures[0].axes[0].lines}
    assert loss_lines["Training loss"].get_color() == "tab:gray"
    assert loss_lines["Training 2-point mean"].get_color() == "tab:blue"
    assert loss_lines["Validation loss"].get_color() == "tab:orange"
    assert loss_lines["Training 2-point mean"].get_linestyle() == "--"
    assert loss_lines["Validation loss"].get_linestyle() == "-"
