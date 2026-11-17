import json

import pandas as pd
import pytest

from pdac_cfrna.predict import label_scores, read_threshold
from pdac_cfrna.thresholds import confusion_counts


def test_score_equal_to_the_threshold_is_called_pdac():
    scores = pd.Series([0.1, 0.28, 0.9], index=["a", "b", "c"], name="pdac_score")
    labels = label_scores(scores, 0.28)
    assert list(labels) == [0, 1, 1]
    assert list(labels.index) == ["a", "b", "c"]
    assert labels.name == "pdac_label"


def test_labels_agree_with_the_confusion_counts_that_chose_the_threshold():
    y = pd.Series([1, 1, 0, 0, 1, 0])
    scores = pd.Series([0.9, 0.3, 0.3, 0.1, 0.28, 0.5])
    labels = label_scores(scores, 0.3)
    counts = confusion_counts(y.to_numpy(), scores.to_numpy(), 0.3)
    assert int(((labels == 1) & (y == 1)).sum()) == counts["tp"]
    assert int(((labels == 1) & (y == 0)).sum()) == counts["fp"]


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_threshold_outside_zero_one_is_rejected(bad):
    with pytest.raises(ValueError, match="between 0 and 1"):
        label_scores(pd.Series([0.5]), bad)


def test_read_threshold_reads_the_chosen_value(tmp_path):
    path = tmp_path / "threshold.json"
    path.write_text(json.dumps({"rule": "fbeta", "threshold": 0.28}), encoding="utf-8")
    assert read_threshold(path) == 0.28


@pytest.mark.parametrize("content", [{"rule": "fbeta"}, {"threshold": "high"}, {"threshold": 2}])
def test_read_threshold_rejects_a_file_without_a_usable_value(tmp_path, content):
    path = tmp_path / "threshold.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="no threshold"):
        read_threshold(path)


def test_read_threshold_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_threshold(tmp_path / "nope.json")
