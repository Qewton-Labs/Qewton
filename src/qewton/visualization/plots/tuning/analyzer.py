# import os
# import json
# from typing import Set

# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go

# from qewton.optim.tuner.base import TunerLoggingKeys


# class TuningAnalyzer:
#     def __init__(self, folder_path: str):
#         self.folder_path = folder_path
#         self.json_files = []
#         self.csv_files = []
#         self.parameters: Set[str] = set()
#         self.metrics: Set[str] = set()
#         self.df: pd.DataFrame = pd.DataFrame()

#         self.metric_data = {}

#         self._scan_folder()
#         self._collect_keys()
#         self._merge_csvs()

#     #######################################################
#     ### Loading
#     # TODO: Better format for saving results?
#     #######################################################
#     def _scan_folder(self):
#         """Find all JSON and CSV files in the folder."""
#         for f in os.listdir(self.folder_path):
#             full_path = os.path.join(self.folder_path, f)
#             if os.path.isfile(full_path):
#                 if f.endswith(".json"):
#                     self.json_files.append(full_path)
#                 elif f.endswith(".csv"):
#                     self.csv_files.append(full_path)

#     def _collect_keys(self):
#         """Collect parameter and metric names from JSON and CSV."""
#         param_key = TunerLoggingKeys.TUNABLEPARAMS.value
#         metric_key = TunerLoggingKeys.TUNEMETRICS.value
#         for f in self.json_files:
#             try:
#                 with open(f, encoding="utf-8") as jf:
#                     data = json.load(jf)
#                 # Parameters from config
#                 if param_key in data:
#                     self.parameters.update(data[param_key].keys())
#                 # Metrics from config
#                 if metric_key in data:
#                     self.metric_data = data[metric_key]
#                     self.metrics.update(data[metric_key].keys())
#             except (OSError, json.JSONDecodeError) as e:
#                 print(f"Failed to read JSON {f}: {e}")

#     def _merge_csvs(self):
#         """Merge all CSV files into a single DataFrame."""
#         dfs = []
#         for f in self.csv_files:
#             try:
#                 df = pd.read_csv(f)
#                 dfs.append(df)
#             except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
#                 print(f"Failed to read CSV {f}: {e}")
#         if dfs:
#             self.df = pd.concat(dfs, ignore_index=True)
#             # Ensure parameters/metrics exist in the DF
#             self.parameters = self.parameters.intersection(self.df.columns)
#             self.metrics = self.metrics.intersection(self.df.columns)

#     def print_tuning_overview(self):
#         print("Found JSON files:", self.json_files)
#         print("Found CSV files:", self.csv_files)
#         print("Detected parameters:", sorted(self.parameters))
#         print("Detected metrics:", sorted(self.metrics))

#     #######################################################
#     ### Statistics
#     # TODO: More statistical analysis possible?
#     #######################################################
#     def summary_statistics(self):
#         if self.df.empty:
#             print("No data loaded.")
#             return
#         print("------ Summary Statistics ------")
#         print("\nTunable parameters:")
#         for p in self.parameters:
#             print(f"{p}: unique values = {self.df[p].nunique()}")

#         print("\nMetrics:")
#         print(self.df[list(self.metrics)].describe())

#     def parameter_metric_correlation(self):
#         param_cols = list(self.parameters)
#         metric_cols = list(self.metrics)

#         param_df = self.df[param_cols].copy()
#         metric_df = self.df[metric_cols].select_dtypes(include=np.number)

#         # Map categorical params to integers
#         for col in param_cols:
#             if not np.issubdtype(param_df[col].dtype, np.number):
#                 param_df[col] = pd.Categorical(param_df[col]).codes

#         # Full correlation matrix
#         full_corr = pd.DataFrame(index=param_cols, columns=metric_cols)
#         for m in metric_cols:
#             objective = self.metric_data.get(m)["objective"]  # type: ignore
#             for p in param_cols:
#                 x = param_df[p]
#                 y = metric_df[m]
#                 if x.nunique() <= 1 or y.nunique() <= 1:
#                     full_corr.loc[p, m] = np.nan
#                 else:
#                     if objective == "minimize":
#                         y = -y
#                     full_corr.loc[p, m] = x.corr(y)
#         print("Correlation of parameters vs metrics:")
#         print(full_corr)
#         return full_corr

#     #######################################################
#     ### Visualization
#     #######################################################
#     def plot_heatmap(self, param_x: str, param_y: str, metric: str):
#         """Heatmap of metric values for two numeric parameters."""
#         pivot = self.df.pivot_table(
#             values=metric, index=param_y, columns=param_x, aggfunc="mean"
#         )

#         fig = go.Figure(
#             data=go.Heatmap(
#                 z=pivot.values,
#                 x=pivot.columns.astype(float),
#                 y=pivot.index.astype(float),
#                 colorbar=dict(title=metric),  # tickformat=".2e"
#             )
#         )

#         fig.update_layout(
#             title=f"{metric}: {param_x} vs {param_y}",
#             xaxis_title=param_x,
#             yaxis_title=param_y,
#         )

#         fig.show()

#     def plot_scatter(self, param: str, metric: str):
#         """Scatter plot of a numeric parameter vs metric with trendline."""
#         x = self.df[param]
#         y = self.df[metric]

#         # Fit linear model manually
#         coeffs = np.polyfit(x, y, 1)
#         poly = np.poly1d(coeffs)

#         fig = go.Figure()

#         fig.add_scatter(x=x, y=y, mode="markers", name="Data")
#         fig.add_scatter(x=x, y=poly(x), mode="lines", name="Trend")

#         fig.update_layout(title=f"{metric} vs {param}")
#         # fig.update_yaxes(tickformat=".2e")
#         fig.show()

#     def plot_parallel_coordinates(
#         self,
#         params: str | list[str] | None = None,
#         metric: str | list[str] | None = None,
#         color_metric: str | None = None,
#         top_k: int | None = None,
#     ):
#         if params is None:
#             params = list(self.parameters)
#         if metric is None:
#             metric = list(self.metrics)
#         if isinstance(params, str):
#             params = [params]
#         if isinstance(metric, str):
#             metric = [metric]

#         if color_metric is None:
#             color_metric = metric[0]

#         cols = params + metric
#         df_plot = self.df[cols].copy()

#         # TODO: Encode categorical values in plot?

#         # Select top_k best results
#         if top_k is not None:
#             objective = self.metric_data.get(color_metric)["objective"]  # type: ignore

#             if objective == "minimize":
#                 df_plot = df_plot.nsmallest(top_k, color_metric)
#             else:
#                 df_plot = df_plot.nlargest(top_k, color_metric)

#         fig = px.parallel_coordinates(
#             df_plot,
#             color=color_metric,
#             color_continuous_scale=px.colors.sequential.Viridis,
#         )

#         fig.show()
