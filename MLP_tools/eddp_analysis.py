#!/usr/bin/env python3
r"""Analyze EDDP ensembles and plot their generation-dependent accuracy.

Requirements
------------
Python 3.11+ (or Python 3.9+ with tomli), NumPy and Matplotlib for plotting,
and Julia with EDDPotentials, JSON and YAML in its active project for analysis.
The Julia backend is accessed through juliacall and EDDPotentials.

Python examples
---------------
    from eddp_analysis import Model_analyzer

    analysis = Model_analyzer(
        "link.toml",
        julia_command="/path/to/julia --project=/path/to/EDDPotentials.jl",
    )
    report = analysis.analyze()  # Keep results in memory; no saved report needed.
    fig, axes = analysis.plot()  # Reuse the results from analyze().
    analysis.analyze(output="ensemble-analysis.json")  # Optionally save JSON.
    fig, axes = plot_ensemble_accuracies("ensemble-analysis.json", properties="test")
    # On a fresh instance, plot() calls analyze() when source is omitted.

Command-line examples
---------------------
    python3 eddp_analysis.py --toml link.toml --plot
    python3 eddp_analysis.py --toml link.toml --output ensemble-analysis.json
    python3 eddp_analysis.py --toml link.toml --plot --source ensemble-analysis.json
    python3 eddp_analysis.py --toml link.toml --plot --figure accuracy.png --no-show
    python3 eddp_analysis.py --toml link.toml --plot \
        --project=/path/to/EDDPotentials.jl

TOML filenames are relative to the caller's directory. [state].workdir is
relative to each TOML's directory, matching the Julia backend. Output paths
are relative to the caller's directory. Plotting an existing report needs no
Julia runtime. Constructor checks verify file presence, while Julia checks
model contents and dataset compatibility during analysis.
"""

import argparse
import csv
import json
import os
import re
import shlex
import warnings
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path

# Shared TOML parsing function

def _read_toml(path):
    """Read a TOML file; install tomli when using Python earlier than 3.11."""
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError as error:
            raise ImportError("Use Python 3.11+ or install tomli to read TOML files.") from error
    with path.open("rb") as handle:
        return tomllib.load(handle)

# Model analysis class, for trained ensemble models

class Model_analyzer:
    """Manage analysis and plotting for one builder TOML configuration.

    Parameters
    ----------
    toml : path-like
        Builder configuration. Ensemble files and analyzable datasets must
        exist in its configured [state].workdir before instantiation.
    julia_command : str
        Explicit Julia executable path plus options, parsed with shlex.
    model_generation : int, optional
        Restrict detailed results to this generation. The overall result
        still evaluates the latest reportable generation, as in Julia.

    Attributes
    ----------
    toml_file : Path
        Absolute builder configuration path.
    config : dict
        Parsed TOML settings; treat these as read-only after construction.
    workdir : Path
        Absolute directory containing the ensemble and training artifacts.
    julia_command : list[str]
        Julia executable and arguments, including an optional --project path.
    model_generation : int or None
        Requested detailed model generation; None means all generations.
    required_inputs : dict
        Paths and generation numbers found by _validate_inputs().
    report : dict or None
        Latest successfully analyzed or loaded report. plot() reuses it.
    """

    def __init__(
        self, toml, *, julia_command, model_generation=None,
    ):
        """Resolve settings and check input files without launching Julia."""
        # Resolve the TOML independently of the configured dataset directory.
        self.toml_file = Path(toml).expanduser().resolve()
        if not self.toml_file.is_file():
            raise FileNotFoundError(f"EDDP configuration not found: {self.toml_file}")
        if self.toml_file.suffix != ".toml":
            raise ValueError("The EDDP configuration must have a .toml extension.")
        # Retain parsed settings so paths and generation discovery are inspectable.
        self.config = _read_toml(self.toml_file)
        state = self.config.get("state", {})
        self.workdir = Path(state.get("workdir", "."))
        if not self.workdir.is_absolute():
            self.workdir = self.toml_file.parent / self.workdir
        self.workdir = self.workdir.resolve()
        # Keep subprocess arguments separate so filenames with spaces stay intact.
        self.julia_command = shlex.split(julia_command)
        if not self.julia_command:
            raise ValueError("julia_command must not be empty.")
        executable = Path(self.julia_command[0]).expanduser()
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(
                "julia_command must begin with an executable Julia path; "
                f"not found or not executable: {executable}"
            )
        project_options = [
            argument.partition("=")[2]
            for argument in self.julia_command
            if argument.startswith("--project=")
        ]
        if len(project_options) > 1:
            raise ValueError("Specify only one --project path in julia_command.")
        if not project_options:
            raise ValueError(
                "julia_command must include --project=/path/to/EDDPotentials.jl "
                "so JuliaCall can initialize the correct environment."
            )
        project_path = Path(project_options[0]).expanduser().resolve()
        if not (project_path / "Project.toml").is_file():
            raise FileNotFoundError(f"Julia Project.toml not found in: {project_path}")
        self.julia_command[
            next(index for index, argument in enumerate(self.julia_command)
                 if argument.startswith("--project="))
        ] = f"--project={project_path}"
        self.julia_project = project_path
        if model_generation is not None and (
            isinstance(model_generation, bool)
            or not isinstance(model_generation, int) or model_generation < 0
        ):
            raise ValueError("model_generation must be a non-negative integer or None.")
        self.model_generation = model_generation
        # No persistent output is necessary; reports can remain in memory.
        self.report = None
        self.required_inputs = self._validate_inputs()

    @staticmethod
    def load_ensemble_analysis(source):
        """Load a serialized analyzer report or an already loaded mapping."""
        if isinstance(source, Mapping):
            data = source
        else:
            with Path(source).open(encoding="utf-8") as handle:
                data = json.load(handle)
        if isinstance(data, Mapping) and isinstance(data.get("report"), Mapping):
            data = data["report"]
        if not isinstance(data, Mapping) or not isinstance(data.get("generations"), list):
            raise ValueError("Analysis data must contain a 'generations' list.")
        return data

    @classmethod
    def from_json(cls, source):
        """Reconstruct an analyzer and its cached report from a JSON file."""
        path = Path(source).expanduser().resolve()
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("report"), Mapping):
            raise ValueError("Analyzer JSON must contain a 'report' object.")
        configuration = payload.get("configuration", {})
        instance = cls.__new__(cls)
        instance.toml_file = Path(configuration["toml_file"]).expanduser()
        instance.config = configuration.get("config", {})
        instance.workdir = Path(configuration.get("workdir", instance.toml_file.parent))
        instance.julia_command = list(configuration.get("julia_command", []))
        project_options = [
            argument.partition("=")[2]
            for argument in instance.julia_command
            if argument.startswith("--project=")
        ]
        instance.julia_project = Path(project_options[0]).expanduser().resolve()
        instance.model_generation = configuration.get("model_generation")
        instance.required_inputs = payload.get("required_inputs", {})
        instance.report = cls.load_ensemble_analysis(payload["report"])
        return instance

    def _json_payload(self, report):
        """Return the complete machine-readable analyzer state."""
        return {
            "schema_version": 1,
            "configuration": {
                "toml_file": str(self.toml_file),
                "workdir": str(self.workdir),
                "julia_command": self.julia_command,
                "model_generation": self.model_generation,
            },
            "required_inputs": self.required_inputs,
            "report": report,
        }

    def _validate_inputs(self):
        """Check the files used by Julia, returning paths and generation lists.

        Require an ensemble and analyzable data for at least one common
        generation. A dataset snapshot OR raw RES structures is sufficient;
        YAML sidecars, seed cells and scheduler scripts are not required by
        this analysis. Julia checks snapshot compatibility and performs raw
        fallback when needed. Call again after adding model/data artifacts.
        """
        if not self.workdir.is_dir():
            raise FileNotFoundError(f"EDDP working directory not found: {self.workdir}")
        prefix = self.config.get("trainer", {}).get("prefix", "")
        pattern = re.compile(re.escape(prefix) + r"ensemble-gen(\d+)\.(?:json|jld2)")
        models = {}
        for path in self.workdir.iterdir():
            matched = pattern.fullmatch(path.name)
            if matched and path.is_file():
                models.setdefault(int(matched[1]), []).append(path)
        if not models:
            raise FileNotFoundError(f"No {prefix}ensemble-gen*.json or .jld2 models in {self.workdir}")

        # Match Builder's current iteration when filtering raw generations.
        max_iterations = self.config.get("state", {}).get("max_iterations", 5)
        iteration = max_iterations
        for generation in range(max_iterations + 1):
            iteration = generation
            if generation not in models:
                break
        snapshots = {}
        training_dir = self.workdir / "training"
        if training_dir.is_dir():
            for path in training_dir.iterdir():
                matched = re.fullmatch(r"gen-(\d+)-dataset\.jld2", path.name)
                if matched and path.is_file():
                    snapshots[int(matched[1])] = path
        raw_directories = {}
        for path in self.workdir.iterdir():
            matched = re.fullmatch(r"gen(\d+)-dft", path.name)
            if matched and int(matched[1]) <= iteration and path.is_dir():
                if any(item.is_file() for item in path.glob("*.res")):
                    raw_directories[int(matched[1])] = path
        data_generations = sorted(set(snapshots) | set(raw_directories))
        if not data_generations:
            raise FileNotFoundError(
                f"No training/gen-*-dataset.jld2 or gen*-dft/*.res data in {self.workdir}"
            )
        reportable = sorted(set(models) & set(data_generations))
        if not reportable:
            raise ValueError("No generation has both an ensemble model and analyzable data.")
        if self.model_generation is not None and self.model_generation not in reportable:
            raise ValueError(
                f"Generation {self.model_generation} needs both an ensemble and data; "
                f"available generations: {reportable}"
            )
        return {
            "models": models, "snapshots": snapshots,
            "raw_directories": raw_directories,
            "reportable_generations": reportable,
        }

    def analyze(self, output=None, *, silent=False):
        """Run Julia, cache and return its report; optionally write a JSON file.

        output=None uses a temporary JSON file that is removed after loading.
        A previous report is replaced only after a successful Julia run and
        successful JSON loading. Julia logs are streamed to the terminal.
        """
        output_path = None if output is None else Path(output).expanduser().resolve()
        if output_path is not None and output_path.suffix != ".json":
            raise ValueError("output must have a .json extension.")
        self.required_inputs = self._validate_inputs()
        try:
            # JuliaCall requires these two variables to be configured as a pair.
            os.environ["PYTHON_JULIACALL_EXE"] = self.julia_command[0]
            os.environ["PYTHON_JULIACALL_PROJECT"] = str(self.julia_project)
            from juliacall import Main as jl
        except ImportError as error:
            raise ImportError("Install juliacall to run ensemble analysis.") from error
        jl.seval("using EDDPotentials")
        # Pass the resolved path directly; do not change Python or Julia's cwd.
        builder = jl.Builder(str(self.toml_file))
        model_generations = [
            generation for generation in self.required_inputs["reportable_generations"]
            if bool(jl.has_ensemble(builder, generation))
        ]
        selected = [self.model_generation] if self.model_generation is not None else model_generations
        reports = []
        data_generations = sorted(
            set(self.required_inputs.get("snapshots", {}))
            | set(self.required_inputs.get("raw_directories", {}))
        )

        def summarize_results(results):
            fc = results.fc
            errors = [
                abs(float(predicted) - float(target)) / float(size)
                for predicted, target, size in zip(results.H_pred, results.H_target, fc.sizes)
            ]
            maximum_error = max(errors) if errors else None
            formulas = [
                str(metadata["formula"] if isinstance(metadata, Mapping) else metadata.formula)
                for metadata in fc.metadata
            ]
            return {
                "number_of_structures": len(fc),
                "number_of_compositions": len(set(formulas)),
                "energy_mae_ev_per_atom": float(jl.mae_per_atom(results)),
                "energy_rmsd_ev_per_atom": float(jl.rmse_per_atom(results)),
                "spearman_rank": float(jl.EDDPotentials.Training.spearman(results)),
                "max_absolute_error_ev_per_atom": maximum_error,
                "max_absolute_error_structure_ids": [
                    str(fc.labels[index]) for index, value in enumerate(errors)
                    if value == maximum_error
                ],
            }

        for generation in selected:
            if not silent:
                print(f"Loading ensemble model generation {generation}", flush=True)
            model = jl.load_ensemble(builder, generation)
            dataset = jl.load_training_dataset(builder, generation, combined=True)
            _, _, test = jl.load_training_dataset(builder, generation, combined=False)
            results = jl.TrainingResults(model, dataset)
            quality = summarize_results(results)
            test_quality = summarize_results(jl.TrainingResults(model, test))
            if not silent:
                print(
                    f"Generation {generation}: cumulative MAE="
                    f"{quality['energy_mae_ev_per_atom']:.8g} eV/atom, "
                    f"test MAE={test_quality['energy_mae_ev_per_atom']:.8g} eV/atom",
                    flush=True,
                )
            report = {
                "model_generation": generation,
                "cumulative_quality": quality,
                "test_quality": test_quality,
                "forward_generation_quality": [],
            }
            for data_generation in data_generations:
                if generation >= data_generation:
                    continue
                if not silent:
                    print(
                        f"Evaluating model generation {generation} on forward generation {data_generation}",
                        flush=True,
                    )
                combined = jl.load_training_dataset(builder, data_generation, combined=True)
                raw_directory = self.required_inputs.get("raw_directories", {}).get(data_generation)
                labels = {path.stem for path in Path(raw_directory).glob("*.res")} if raw_directory else set()
                indices = [
                    index + 1 for index, label in enumerate(map(str, combined.labels))
                    if label in labels
                ]
                if not indices:
                    warnings.warn(
                        f"No structures found for forward generation {data_generation}; skipping.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                julia_indices = jl.Vector[jl.Int64](indices)
                forward_features = combined[julia_indices]
                forward_results = jl.TrainingResults(model, forward_features)
                summary = summarize_results(forward_results)
                summary.update({
                    "structure_generation": data_generation,
                    "scope": "structures_first_introduced_in_generation",
                })
                report["forward_generation_quality"].append(summary)
            reports.append(report)
        report = self.load_ensemble_analysis({"generations": reports})
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(self._json_payload(report), indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        self.report = report
        return report

    def summarize(self, output="ensemble-summary.csv"):
        """Write cumulative accuracy metrics with model generations as columns."""
        report = self.report if self.report is not None else self.analyze()
        generations = sorted(report["generations"], key=lambda item: item["model_generation"])
        columns = [str(item["model_generation"]) for item in generations]
        metrics = (
            ("MAE (meV/atom)", "energy_mae_ev_per_atom", 1000.0),
            ("RMSD (meV/atom)", "energy_rmsd_ev_per_atom", 1000.0),
            ("Spearman correlation", "spearman_rank", 1.0),
            ("Max AE (meV/atom)", "max_absolute_error_ev_per_atom", 1000.0),
            ("Max AE structure ID", "max_absolute_error_structure_ids", 1.0),
        )
        rows = []
        for generation in generations:
            quality = generation.get("cumulative_quality", {})
            row = [generation["model_generation"]]
            for _, key, scale in metrics:
                value = quality.get(key, "")
                if isinstance(value, (int, float)) and key != "max_absolute_error_structure_ids":
                    value *= scale
                elif isinstance(value, list):
                    value = ";".join(map(str, value))
                row.append("" if value is None else value)
            rows.append(row)
        output_path = Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model_generation", *(label for label, _, _ in metrics)])
            writer.writerows(rows)
        return output_path

    def plot(
            self, properties="all", *, output=None, show=False,
        figsize=(12, 9),
        x_range=None,
        y_ranges=None,
        forward_error_bar=True,
        legend_loc="best",
    ):
        """Plot ensemble accuracy versus model generation.

        Parameters
        ----------
        properties : str or sequence of str, default 'all'
            Any of 'cumulative', 'forward', 'test', or 'all'. Forward points are
            averages over later structure generations and their error bars span
            the minimum-to-maximum range.
        figsize : tuple
            Matplotlib figure size.
        x_range : (float, float), optional
            Shared minimum and maximum model generation shown on the x axes.
        y_ranges : (float, float) or mapping, optional
            One shared y range, or ranges keyed by 'mae', 'rmsd', 'spearman',
            and 'max_error'. These apply to the primary axes.
        legend_loc : str, default 'best'
            Use 'best' to choose each subplot's legend position automatically,
            or select upper/lower left or upper/lower right explicitly.

        Returns
        -------
        fig, axes
            The Matplotlib figure and 2 x 2 array of axes.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        property_order = ("cumulative", "forward", "test")
        property_colors = {"cumulative": "tab:blue", "forward": "tab:orange", "test": "tab:green"}
        property_labels = {"cumulative": "Cumulative", "forward": "Forward", "test": "Test"}
        metrics = (
            ("mae", "energy_mae_ev_per_atom", "Energy MAE", "MAE (meV/atom)", 1000.0),
            ("rmsd", "energy_rmsd_ev_per_atom", "Energy RMSD", "RMSD (meV/atom)", 1000.0),
            ("spearman", "spearman_rank", "Energy Spearman correlation", "Spearman correlation", 1.0),
            ("max_error", "max_absolute_error_ev_per_atom", "Energy maximum AE", "Max AE (meV/atom)", 1000.0),
        )
        legend_corners = ("upper left", "upper right", "lower left", "lower right")

        def finite_value(record, metric, scale):
            value = record.get(metric) if isinstance(record, Mapping) else None
            return np.nan if value is None or not np.isfinite(value) else float(value) * scale

        def axis_range(value, name):
            if value is None:
                return None
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
                raise ValueError(f"{name} must be a (minimum, maximum) pair.")
            if any(limit is not None and not isinstance(limit, Real) for limit in value):
                raise ValueError(f"{name} limits must be numbers or None.")
            return tuple(value)

        def direct_series(generations, quality_key, metric, scale):
            x = np.asarray([generation["model_generation"] for generation in generations])
            y = np.asarray([finite_value(generation.get(quality_key, {}), metric, scale) for generation in generations])
            valid = np.isfinite(y)
            return x[valid], y[valid]

        def forward_series(generations, metric, scale):
            values_by_generation = []
            for generation in generations:
                values = np.asarray([finite_value(record, metric, scale) for record in generation.get("forward_generation_quality", [])])
                values = values[np.isfinite(values)]
                if values.size:
                    mean = values.mean()
                    values_by_generation.append((generation["model_generation"], mean, mean - values.min(), values.max() - mean))
            if not values_by_generation:
                return np.asarray([]), np.asarray([]), np.asarray([[], []])
            x, means, lower, upper = zip(*values_by_generation)
            return np.asarray(x), np.asarray(means), np.asarray([lower, upper])

        def normalise_properties(value):
            if value is None or value == "all":
                return list(property_order)
            requested = [value] if isinstance(value, str) else list(value) if isinstance(value, Sequence) else None
            if requested is None:
                raise TypeError("properties must be a string or a sequence of strings.")
            if "all" in requested:
                if len(requested) != 1:
                    raise ValueError("Use 'all' by itself, or list individual properties.")
                return list(property_order)
            unknown = [name for name in requested if name not in property_order]
            if unknown:
                raise ValueError(f"Unknown properties {unknown}. Choose from: {', '.join((*property_order, 'all'))}.")
            if not requested:
                raise ValueError("Select at least one property.")
            return list(dict.fromkeys(requested))

        metric_names = [metric[0] for metric in metrics]
        data = self.report if self.report is not None else self.analyze()
        selected = normalise_properties(properties)
        for property_name in selected:
            quality_key = (
                "forward_generation_quality"
                if property_name == "forward"
                else f"{property_name}_quality"
            )
            if not any(generation.get(quality_key) for generation in data["generations"]):
                warnings.warn(
                    f"Requested '{property_name}' data is unavailable in the report.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        x_range = axis_range(x_range, "x_range")
        if y_ranges is None:
            y_ranges = dict.fromkeys(metric_names)
        elif isinstance(y_ranges, Mapping):
            unknown = set(y_ranges) - set(metric_names)
            if unknown:
                raise ValueError(f"Unknown y-range metrics: {sorted(unknown)}.")
            y_ranges = {name: axis_range(y_ranges.get(name), f"y_ranges[{name!r}]") for name in metric_names}
        else:
            y_ranges = dict.fromkeys(metric_names, axis_range(y_ranges, "y_ranges"))
        # Kept as an accepted argument for compatibility with existing callers.
        # Forward data is plotted on the same axis as the other properties.
        if legend_loc != "best" and legend_loc not in legend_corners:
            valid_locations = ("best", *legend_corners)
            raise ValueError(f"legend_loc must be one of: {', '.join(valid_locations)}.")
        generations = sorted(data["generations"], key=lambda item: item["model_generation"])
        if not generations:
            raise ValueError("The analysis contains no generations to plot.")

        fig, axes = plt.subplots(2, 2, figsize=figsize, layout='tight')
        axes = np.asarray(axes)
        missing_forward_metrics = []

        for ax, (metric_name, metric, title, ylabel, scale) in zip(axes.flat, metrics):
            has_forward_data = False
            for property_name in selected:
                color = property_colors[property_name]
                label = property_labels[property_name]

                if property_name == "forward":
                    x, y, yerr = forward_series(generations, metric, scale)
                    if y.size:
                        has_forward_data = True
                        if forward_error_bar:
                            ax.errorbar(
                                x, y, yerr=yerr, fmt="o-", color=color, label=label,
                                linewidth=1.8, markersize=4, capsize=3, elinewidth=1.1,
                            )
                        else:
                            ax.plot(
                                x, y, "o-", color=color, label=label,
                                linewidth=1.8, markersize=4,
                            )
                    else:
                        missing_forward_metrics.append(title)
                    continue

                quality_key = f"{property_name}_quality"
                x, y = direct_series(generations, quality_key, metric, scale)
                if y.size:
                    ax.plot(
                        x, y, "o-", color=color, label=label,
                        linewidth=1.8, markersize=4,
                    )

            ax.set_title(title)
            ax.set_ylabel(ylabel)
            # Use logarithmic scaling for error panels when forward data is
            # available. Spearman correlation remains linear because it is signed.
            ax.set_yscale(
                "log" if metric_name != "spearman" and has_forward_data else "linear"
            )
            ax.grid(False)
            if y_ranges[metric_name] is not None:
                ax.set_ylim(y_ranges[metric_name])

            handles, labels = ax.get_legend_handles_labels()
            handles_by_label = dict(zip(labels, handles))
            legend_labels = [
                property_labels[name]
                for name in selected
                if property_labels[name] in handles_by_label
            ]
            if legend_labels:
                ax.legend(
                    [handles_by_label[label] for label in legend_labels], legend_labels,
                    loc=legend_loc,
                )

        for ax in axes[-1, :]:
            ax.set_xlabel("Model generation")

        model_generations = [generation["model_generation"] for generation in generations]
        for ax in axes.flat:
            ax.set_xticks(model_generations)
            if x_range is not None:
                ax.set_xlim(x_range)

        if missing_forward_metrics:
            missing = ", ".join(dict.fromkeys(missing_forward_metrics))
            fig.text(
                0.5, 0.005, f"Forward data unavailable for: {missing}",
                ha="center", fontsize=9, color="0.4",
            )

        if output is not None:
            output_path = Path(output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, bbox_inches="tight")
        if show:
            plt.show()
        self.report = data
        return fig, axes

# A wrapper to plot quality metrics from previously dumped data

def plot_ensemble_accuracies(
    source="ensemble-analysis.json", properties="all", *, summary=False, **plot_kwargs
):
    """Plot a serialized analyzer report without modifying its JSON source.

    When ``summary=True``, write ``<source-stem>-summary.csv`` alongside the
    source JSON and return the figure and axes as usual.
    """
    analyzer = Model_analyzer.from_json(source)
    figure = analyzer.plot(properties=properties, **plot_kwargs)
    if summary:
        source_path = Path(source).expanduser().resolve()
        analyzer.summarize(source_path.with_name(f"{source_path.stem}-summary.csv"))
    return figure
# def main(argv=None):
#     """Run the CLI; --plot analyzes as needed, and --source plots saved JSON.
#
#     Pass an argv list for programmatic invocation. Multiple TOMLs are
#     processed independently; explicit output/source/figure paths require
#     exactly one TOML to avoid overwriting or mixing reports.
#     """
#     parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
#     parser.add_argument("--toml", nargs="+", required=True, help="One or more builder TOML files")
#     parser.add_argument("--plot", action="store_true", help="Plot cached or newly calculated analysis")
#     parser.add_argument("--source", help="Existing JSON report to plot, bypassing Julia")
#     parser.add_argument("--output", help="Optional JSON analysis output file")
#     parser.add_argument("--figure", help="Optional figure output file, e.g. accuracy.png")
#     parser.add_argument("--no-show", action="store_true", help="Do not display plot windows")
#     parser.add_argument("--figsize", nargs=2, type=float, default=(12, 9), metavar=("WIDTH", "HEIGHT"))
#     parser.add_argument("--x-range", nargs=2, type=float, metavar=("MIN", "MAX"))
#     parser.add_argument("--y-ranges", type=json.loads, help='JSON ranges, e.g. \'{"mae": [0, 20]}\'')
#     parser.add_argument(
#         help="Deprecated compatibility option; forward data uses the primary y-axis",
#     )
#     parser.add_argument("--generation", type=int, help="Optional model generation for detailed analysis")
#     parser.add_argument("--julia-command", required=True, help="Explicit Julia executable path plus options")
#     args = parser.parse_args(argv)
#     if (args.source or args.figure) and not args.plot:
#         parser.error("--source and --figure require --plot")
#     if args.source and (args.output or args.generation is not None):
#         parser.error("--source cannot be combined with --output or --generation")
#     if len(args.toml) > 1 and any((args.source, args.output, args.figure)):
#         parser.error("--source, --output and --figure require exactly one --toml file")
#     try:
#         for toml in args.toml:
#             analysis = Model_analyzer(
#                 toml, julia_command=args.julia_command,
#                 model_generation=args.generation,
#             )
#             if args.output or not args.plot:
#                 report = analysis.analyze(output=args.output)
#                 print(f"Analyzed {analysis.toml_file}: {len(report['generations'])} model generations")
#                 if args.output:
#                     print(f"Wrote report to {Path(args.output).expanduser().resolve()}")
#             if args.plot:
#                 analysis.plot(
#                     source=args.source, properties=args.properties,
#                     output=args.figure, show=not args.no_show,
#                     figsize=args.figsize, x_range=args.x_range,
#                     legend_loc=args.legend_loc,
#                 )
#                 if args.figure:
#                     print(f"Wrote figure to {Path(args.figure).expanduser().resolve()}")
#     except (OSError, ValueError, ImportError) as error:
#         parser.exit(1, f"Error: {error}\n")
#     return 0
#
#
# if __name__ == "__main__":
#     raise SystemExit(main())
