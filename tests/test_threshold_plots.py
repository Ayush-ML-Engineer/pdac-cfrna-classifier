import numpy as np

from pdac_cfrna.threshold_plots import plot_confusion_matrices


def test_confusion_figure_is_written_with_a_held_out_panel(tmp_path):
    y = np.array([1, 1, 0, 0, 0])
    score = np.array([0.9, 0.4, 0.3, 0.2, 0.1])
    held_out = {"held-out": {"tp": 1.5, "fn": 0.5, "tn": 2.8, "fp": 0.2}}
    path = tmp_path / "cm.png"
    plot_confusion_matrices({"Validation": (y, score)}, 0.35, path, held_out)
    assert path.stat().st_size > 0


def test_confusion_figure_without_held_out_still_works(tmp_path):
    y = np.array([1, 0])
    path = tmp_path / "cm.png"
    plot_confusion_matrices({"Validation": (y, np.array([0.8, 0.2]))}, 0.5, path)
    assert path.stat().st_size > 0
