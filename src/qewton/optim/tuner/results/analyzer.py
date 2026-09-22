from __future__ import annotations
import pandas as pd
import numpy as np
import json
from pathlib import Path

from qewton.optim.tuner.results.tune_results import TuneResultCollector
from qewton.constraints.base import ConstraintObjective
from qewton.visualization.plots.table.parallel_coordinates import ParallelCoordinatesPlot
from qewton.visualization.plots.spec import ColorSpec, VariableSpec
from qewton.visualization.figure import Figure
from qewton.visualization.plots.table.scatter_table import TableScatter
from qewton.visualization.plots.table.heatmap import TableHeatMap


class TuningAnalyzer:
    def __init__(self, file_path: str | Path):

        # Check for the existence of the file path and load the CSV and JSON files
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")

        csv_path = file_path / TuneResultCollector.file_names["csv"]
        json_path = file_path / TuneResultCollector.file_names["json"]

        # Load the CSV and JSON files
        self.df_raw = pd.read_csv(csv_path)
        with open(json_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        self._init_common(self.df_raw, self.meta)

    @classmethod
    def from_df(cls, df_raw: pd.DataFrame, meta: dict):
        """Alternate constructor: build an Analyzer from an already-loaded
        DataFrame + meta dict, without touching disk. Used by filter()."""
        obj = cls.__new__(cls)
        obj.df_raw = df_raw
        obj.meta = meta
        obj._init_common(df_raw, meta)
        return obj

    def _init_common(self, df_raw: pd.DataFrame, meta: dict):
        # Extract objective/metric and hyperparameter information from the JSON metadata
        self.metrics: dict[str, str] = {}  # dict of metric_name -> objective (min/max)
        for k, v in zip(
            meta[TuneResultCollector.objective_names_key],
            meta[TuneResultCollector.objective_values_key],
        ):
            self.metrics[k] = v

        self.hp_names: list[str] = meta[TuneResultCollector.hp_key]
        self._validate_schema(df_raw)

        # Identify categorical and numeric hyperparameters
        self.categorical_params = [
            c for c in self.hp_names if not pd.api.types.is_numeric_dtype(df_raw[c])
        ]
        self.numeric_params = [
            c for c in self.hp_names if c not in self.categorical_params
        ]

        self.df, self.df_failed = self._split_failed_runs(df_raw)

    def _validate_schema(self, df_raw: pd.DataFrame):
        missing = [m for m in self.metrics if m not in df_raw.columns]
        missing += [hp for hp in self.hp_names if hp not in df_raw.columns]
        if missing:
            raise ValueError(
                f"Metrics or HyperParameter in JSON but not in CSV: {missing}"
            )

    def _split_failed_runs(
        self, df_raw: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        mask = df_raw[list(self.metrics)].isna().any(axis=1)
        return df_raw[~mask].copy(), df_raw[mask].copy()  # type: ignore

    def __repr__(self):
        return f"<TuningAnalyzer: {len(self.df)} runs ({len(self.df_failed)} failed), \
        metrics={list(self.metrics)}>"

    def __len__(self):
        return len(self.df)

    #################################################################################
    # Filtering
    def filter(self, **kwargs) -> TuningAnalyzer:
        """Return a new TuningAnalyzer restricted to the matching rows.

        Args:
            kwargs: Objective or Parameters and values to filter by. The values can be:
                - exact match or a list of allowed values
                - a (lo, hi) tuple for numeric values to specify a range.

        Example:
            sub = analyzer.filter(batch_size=(16, 64), optimizer="adam")
        """
        mask = pd.Series(True, index=self.df_raw.index)
        for k, v in kwargs.items():
            if k not in self.df_raw.columns:
                raise KeyError(f"Unknown column: {k}")
            if isinstance(v, tuple) and len(v) == 2:
                mask &= self.df_raw[k].between(*v)
            elif isinstance(v, (list, set)):
                mask &= self.df_raw[k].isin(v)
            else:
                mask &= self.df_raw[k] == v
        filtered_df = self.df_raw[mask]
        if filtered_df.empty:
            print(f"Filter {kwargs} matched no runs")
        return TuningAnalyzer.from_df(filtered_df, self.meta)

    #################################################################################
    # Score normalization and combination methods
    def _metric_direction_max(self, metric):
        """Returns True if the metric is to be maximized, False if minimized."""
        direction = self.metrics[metric]
        return direction == str(ConstraintObjective.MAXIMIZE)

    def normalized_score(self, metric):
        """Returns a Series in [0,1], where 1 = best, regardless of min/max direction."""
        col = self.df[metric]
        lo, hi = col.min(), col.max()
        if hi == lo:
            return pd.Series(1.0, index=col.index)
        norm = (col - lo) / (hi - lo)
        return norm if self._metric_direction_max(metric) else 1 - norm

    def combined_score(self, weights: dict | None = None):
        """Computes for all metrics a combined score in [0,1], where 1 = best,
        using the given weights. Each metric is normalized to [0,1] first,
        then the weighted average is computed. For the metrics the min/max direction is
        respected, such that 1 always equals the best value.

        Args:
            weights (dict, optional): A dictionary mapping metric names to their weights.
                If None, all metrics are weighted equally.
        """
        if weights is None:
            weights = {m: 1.0 for m in self.metrics}
        scores = sum(self.normalized_score(m) * w for m, w in weights.items())
        return scores / sum(weights.values())

    #################################################################################
    # Ranking
    def best_run(self, metric: str | None = None) -> pd.Series:
        """returns the row corresponding to the best run for the given metric, respecting
        the min/max direction of the metric.

        Args:
            metric (str | None): The key of the objective/metric to rank by. By default,
                the first metric in the list is used.

        Returns:
            pd.Series: The parameters that produced the best value for the given metric,
                as a pandas Series.
        """
        if metric is None:
            metric = list(self.metrics.keys())[0]
        idx = (
            self.df[metric].idxmax()
            if self._metric_direction_max(metric)
            else self.df[metric].idxmin()
        )
        return self.df.loc[idx]

    def top_k(self, metric: str | None = None, k=5) -> pd.DataFrame:
        """Returns the top k rows for the given metric,
        respecting the min/max direction of the metric.

        Args:
            metric (str | None, optional): The metric/objective to order by.
                Defaults to None and picks the first objective in self.metrics.
            k (int, optional): The number of top rows to return. Defaults to 5.

        Returns:
            pd.DataFrame: The top k rows for the given metric, sorted in ascending
                order if the metric is to be minimized, and descending order if it
                is to be maximized.
        """
        if metric is None:
            metric = list(self.metrics.keys())[0]
        direction = not self._metric_direction_max(metric)
        return self.df.sort_values(metric, ascending=direction).head(k)

    def config_diff(self, metric: str | None = None, top_n=5, bottom_n=5) -> pd.DataFrame:
        """Compares the top and bottom k runs for a given metric, and returns a DataFrame
        showing the mean of numeric hyperparameters for the top and bottom runs.

        Args:
            metric (str | None, optional): The metric/objective to compare with.
                Defaults to None and takes the first value in self.metrics.
            top_n (int, optional): The number of top runs to compare. Defaults to 5.
            bottom_n (int, optional): The number of bottom runs to compare. Defaults to 5.

        Returns:
            pd.DataFrame: The mean/mode of hyperparameters for the top and bottom runs,
                with hyperparameter names as the index and columns for top_mean,
                bottom_mean, top_mode, and bottom_mode.
        """
        if metric is None:
            metric = list(self.metrics.keys())[0]
        ascending = not self._metric_direction_max(metric)
        sorted_df = self.df.sort_values(metric, ascending=ascending)
        top = sorted_df.head(top_n)
        bottom = sorted_df.tail(bottom_n)
        rows = {}
        for p in self.numeric_params:
            top_mean, bottom_mean = top[p].mean(), bottom[p].mean()
            top_std, bottom_std = top[p].std(), bottom[p].std()

            # pooled std for Cohen's d; guard against zero-variance groups
            n1, n2 = len(top), len(bottom)
            pooled_std = (
                np.sqrt(
                    ((n1 - 1) * top_std**2 + (n2 - 1) * bottom_std**2) / (n1 + n2 - 2)
                )
                if (n1 + n2) > 2
                else np.nan
            )
            cohens_d = (
                (top_mean - bottom_mean) / pooled_std
                if pooled_std and pooled_std > 0
                else np.nan
            )

            row = {
                "top_mean": top_mean,
                "top_std": top_std,
                "bottom_mean": bottom_mean,
                "bottom_std": bottom_std,
                "cohens_d": cohens_d,
            }
            rows[p] = row

        return pd.DataFrame(rows).T

    #################################################################################
    # Statistics and analysis
    def metric_summary(self):
        """Summarizes the metrics/objectives in the DataFrame."""
        return self.df[list(self.metrics)].describe()

    def metric_correlations(self):
        """Computes the correlation matrix for the metrics/objectives in the DataFrame."""
        return self.df[list(self.metrics)].corr()

    def param_importance(self, metric: str | None = None) -> pd.Series:
        """Computes the importance of each hyperparameter with respect to the given metric.
        For numeric hyperparameters, the absolute value of the Pearson correlation coefficient
        is used. For categorical hyperparameters, the correlation ratio (eta) is used.
        Values of order 1 indicate strong correlation, while values near 0 indicate weak
        correlation.

        Args:
            metric (str | None): The metric/objective to compute importance against.
                If None, the first metric in self.metrics is used.
        """
        if metric is None:
            metric = list(self.metrics.keys())[0]
        scores = {}
        for p in self.numeric_params:
            scores[p] = abs(self.df[p].corr(self.df[metric]))
        for p in self.categorical_params:
            scores[p] = self._correlation_ratio(self.df[p], self.df[metric])
        return pd.Series(scores).sort_values(ascending=False)

    @staticmethod
    def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
        """Eta (correlation ratio): sqrt(between-group variance / total variance).
        0 = category has no bearing on the value, 1 = category fully determines it.
        """
        df = pd.DataFrame({"cat": categories, "val": values}).dropna()
        if df["cat"].nunique() < 2:
            return np.nan
        grand_mean = df["val"].mean()
        ss_between = (
            df.groupby("cat")["val"]
            .apply(lambda g: len(g) * (g.mean() - grand_mean) ** 2)
            .sum()
        )
        ss_total = ((df["val"] - grand_mean) ** 2).sum()
        return np.sqrt(ss_between / ss_total) if ss_total > 0 else np.nan

    def bootstrap_ci(
        self, metric: str | None = None, group_cols=None, n_boot=1000, ci=0.95
    ) -> pd.Series:
        """Computes a bootstrap confidence interval for the mean of the given metric,
        optionally across groups defined by group_cols.
        Returns a DataFrame with the mean and confidence interval bounds for each group.

        Args:
            metric (str | None, optional): The metric/objective that should be analyzed.
                Defaults to None and picks the first value in self.metrics.
            group_cols (_type_, optional): The columns to group by before computing
                the bootstrap CI. Defaults to None.
            n_boot (int, optional): The number of bootstrap samples to draw.
                Defaults to 1000.
            ci (float, optional): The confidence level for the interval. Defaults to 0.95.

        Returns:
            pd.Series: The mean and confidence interval bounds for the specified metric,
                optionally grouped by the specified columns.
        """
        if metric is None:
            metric = list(self.metrics.keys())[0]

        def _ci(series):
            boots = [series.sample(frac=1, replace=True).mean() for _ in range(n_boot)]
            lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
            return pd.Series({"mean": series.mean(), "ci_lo": lo, "ci_hi": hi})

        if group_cols:
            return self.df.groupby(group_cols)[metric].apply(_ci).unstack()
        return _ci(self.df[metric])

    def pareto_front(self, metric_a: str, metric_b: str) -> pd.DataFrame:
        """Returns the rows that are on the Pareto front for the two given metrics.
        A row is on the Pareto front if there is no other row that is better in both metrics.

        Args:
            metric_a (str): The first metric/objective to consider.
            metric_b (str): The second metric/objective to consider.

        Returns:
            pd.DataFrame: The rows that are on the Pareto front for the two given metrics
        """
        a_metric_min = not self._metric_direction_max(metric_a)
        b_metric_min = not self._metric_direction_max(metric_b)
        a = self.df[metric_a] if a_metric_min else -self.df[metric_a]
        b = self.df[metric_b] if b_metric_min else -self.df[metric_b]
        dominated = [
            (
                (a <= a.iloc[i]) & (b <= b.iloc[i]) & ((a < a.iloc[i]) | (b < b.iloc[i]))
            ).any()
            for i in range(len(self.df))
        ]
        return self.df[~pd.Series(dominated, index=self.df.index)]

    def search_space_coverage(self):
        """Returns a dictionary summarizing the coverage of the search space for each
        hyperparameter. For numeric hyperparameters, it includes the number of unique
        values, minimum, and maximum. For categorical hyperparameters, it includes the
        number of unique values and the counts of each category.
        """
        cov = {}
        for p in self.numeric_params:
            cov[p] = {
                "n_unique": self.df[p].nunique(),
                "min": self.df[p].min(),
                "max": self.df[p].max(),
            }
        for p in self.categorical_params:
            cov[p] = {
                "n_unique": self.df[p].nunique(),
                "counts": self.df[p].value_counts().to_dict(),
            }
        return pd.DataFrame(cov).T

    ##################################################################################
    ### Plotting
    def parallel_coordinates_plot(
        self,
        axes: list[str] | None = None,
        objectives: list[str] | None = None,
        color: ColorSpec | str | None = None,
        log_axes: list[str] | None = None,
        **kwargs,
    ) -> Figure:
        """Creates a parallel coordinates plot for the given axes and
        objectives

        Args:
            axes (list[str] | None, optional): The parameters that
                should be shown in the plot. Defaults to None yielding all of them.
            objectives (list[str] | None, optional): The objectives that should be
                shown in the plot. Defaults to None showing all of them.
            color (ColorSpec | str | None, optional): The data that should be used
                for coloring the plot. Defaults to None creating a selection over
                all objective/metrics.
            log_axes (list[str], optional): Which axes should be shown in a log
                scale. Defaults to [].

        Raises:
            ValueError: If the color column is not found in the axes or
                objectives.

        Returns:
            Figure: A Figure object containing the parallel coordinates plot.
                Call `Figure.show()` to display it.
        """
        if log_axes is None:
            log_axes = []
        if axes is None:
            axes = self.hp_names
        if objectives is None:
            objectives = list(self.metrics.keys())
        if color is None:
            if len(objectives) > 1:
                color = ColorSpec(VariableSpec(candidates=objectives))
            else:
                color = objectives[0]
        elif isinstance(color, str):
            if color not in objectives + axes:
                raise ValueError(f"Color column {color} not found in axes or objectives.")
        para_plot = ParallelCoordinatesPlot(
            columns=self.df,
            axes=axes + objectives,
            color=color,
            log_axes=log_axes,
            **kwargs,
        )
        return Figure(para_plot)

    def scatter_table_plot(
        self,
        log_axes: list[str] | None = None,
        **kwargs,
    ) -> Figure:
        """Creates a scatter table plot for the given axes and objectives.

        Args:
            log_axes (list[str], optional): Which axes should be shown in a
                log scale. Defaults to [].

        Returns:
            Figure: A Figure object containing the scatter table plot.
                Call `Figure.show()` to display it.
        """
        if log_axes is None:
            log_axes = []
        combined_objective = self.combined_score()
        table_data = self.df.copy()
        table_data["combined_score"] = combined_objective
        scatter_plot = TableScatter(
            data=table_data,
            axis_keys=self.hp_names,
            objective_keys=list(self.metrics.keys()) + ["combined_score"],
            log_axes=log_axes,
            **kwargs,
        )
        return Figure(scatter_plot)

    def heatmap_plot(
        self,
        bins: int,
        axis_keys: list[str] | None = None,
        log_axes: list[str] | None = None,
        **kwargs,
    ) -> Figure:
        """Creates a heatmap plot for the given axes and objectives.
        Note that non numeric hyperparameters will be ignored in the
        heatmap plot, as they cannot be represented in a 2D grid.

        Args:
            bins (int): The number of bins to use for the heatmap.
            axis_keys (list[str] | None, optional): The parameters that
                should be shown in the plot. Defaults to None yielding all of them.
            log_axes (list[str], optional): Which axes should be shown in a log
                scale. Defaults to [].

        Returns:
            Figure: A Figure object containing the heatmap plot.
                Call `Figure.show()` to display it.
        """
        if log_axes is None:
            log_axes = []
        if axis_keys is None:
            axis_keys = self.numeric_params
        else:
            assert all(
                key in self.numeric_params for key in axis_keys
            ), "Heatmap plot only supports numeric hyperparameters."
        combined_objective = self.combined_score()
        table_data = self.df.copy()
        table_data["combined_score"] = combined_objective
        heatmap_plot = TableHeatMap(
            data=table_data,
            bins=bins,
            axis_keys=axis_keys,
            objective_keys=list(self.metrics.keys()) + ["combined_score"],
            log_axes=log_axes,
            **kwargs,
        )
        return Figure(heatmap_plot)
