import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from qewton.optim.tuner.results.analyzer import TuningAnalyzer  # noqa: E402
from qewton.optim.tuner.results.tune_results import TuneResultCollector  # noqa: E402


@pytest.fixture
def analyzer():
    rng = np.random.default_rng(0)
    n = 12
    df = pd.DataFrame(
        {
            "lr": rng.uniform(1e-4, 1e-1, n),
            "loss": rng.uniform(0.01, 1.0, n),
            "error": rng.uniform(0.1, 2.0, n),
        }
    )
    meta = {
        TuneResultCollector.objective_names_key: ["loss", "error"],
        TuneResultCollector.objective_values_key: ["minimize", "minimize"],
        TuneResultCollector.hp_key: ["lr"],
    }
    return TuningAnalyzer.from_df(df, meta)


def test_with_combined_score_returns_a_new_analyzer(analyzer):
    combined = analyzer.with_combined_score({"loss": 0.7, "error": 0.3})
    assert "combined_score" in combined.metrics
    assert "combined_score" not in analyzer.metrics
    assert "combined_score" not in analyzer.df.columns
    # 1 = best, so the best run carries the highest combined score
    assert combined.best_run("combined_score")["combined_score"] == pytest.approx(
        combined.df["combined_score"].max()
    )


def test_the_combined_score_respects_the_weights(analyzer):
    combined = analyzer.with_combined_score({"loss": 1.0, "error": 0.0})
    assert np.allclose(combined.df["combined_score"], combined.normalized_score("loss"))


def test_the_combined_score_survives_filter(analyzer):
    combined = analyzer.with_combined_score({"loss": 0.7, "error": 0.3})
    sub = combined.filter(lr=(analyzer.df["lr"].min(), analyzer.df["lr"].median()))
    assert "combined_score" in sub.metrics
    assert sub.best_run("combined_score")["combined_score"] == pytest.approx(
        sub.df["combined_score"].max()
    )


def test_with_combined_score_is_idempotent(analyzer):
    once = analyzer.with_combined_score({"loss": 0.7, "error": 0.3})
    twice = once.with_combined_score()
    assert np.allclose(twice.df["combined_score"], once.df["combined_score"])


def test_no_combined_score_for_a_single_metric():
    df = pd.DataFrame({"lr": [1e-3, 1e-2], "loss": [0.5, 0.1]})
    meta = {
        TuneResultCollector.objective_names_key: ["loss"],
        TuneResultCollector.objective_values_key: ["minimize"],
        TuneResultCollector.hp_key: ["lr"],
    }
    analyzer = TuningAnalyzer.from_df(df, meta).with_combined_score()
    assert "combined_score" not in analyzer.metrics
