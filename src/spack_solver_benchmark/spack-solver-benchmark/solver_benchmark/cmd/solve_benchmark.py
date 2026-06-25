# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
import argparse
import io
import os
import pathlib
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spack.cmd
import spack.cmd.solve
import spack.hash_types as ht
import spack.llnl.util.tty.color as color
import spack.package_base
import spack.solver.asp as asp
import spack.spec
import spack.util.parallel
import spack.util.spack_json as sjson
import spack.util.timer
from scipy.stats import wilcoxon
from spack.cmd.common.arguments import add_concretizer_args
from spack.llnl.util import tty

SOLUTION_PHASES = "setup", "load", "ground", "solve"
TIMING_COLS = [*SOLUTION_PHASES, "total"]
COLUMNS = ["spec", "hash", "iteration", *TIMING_COLS, "deps"]
ALPHA = 0.05
COMPILER_VIRTUALS = {"c", "cxx", "fortran"}


level = "long"
section = "developer"
description = "benchmark concretization speed"


def setup_parser(subparser: argparse.ArgumentParser):
    sp = subparser.add_subparsers(metavar="SUBCOMMAND", dest="subcommand")

    run_parser = sp.add_parser("run", help=run.__doc__)
    run_parser.add_argument(
        "-r",
        "--repetitions",
        type=int,
        help="number of repetitions for each spec",
        default=1,
    )
    run_parser.add_argument("-o", "--output", help="CSV output file", required=True)
    run_parser.add_argument(
        "-n",
        "--nprocess",
        help="number of processes to use to produce the results",
        default=os.cpu_count(),
        type=int,
    )
    run_parser.add_argument(
        "--no-shuffle",
        help="do not shuffle the input specs before running the benchmark",
        action="store_true",
    )
    run_parser.add_argument(
        "--clear-repo-modules",
        help="clear spack_repo.* modules from sys.modules before each solve to include package "
        "load time",
        action="store_true",
    )
    add_concretizer_args(run_parser)
    run_parser.add_argument(
        "-j",
        "--json",
        metavar="FILE",
        dest="spec_output",
        default=None,
        help="output concretized specs as JSON to the specified file",
    )
    run_parser.add_argument(
        "specfile",
        help="text file with one spec per line, can be one of the predefined benchmarks",
    )

    compare_parser = sp.add_parser("compare", help=compare.__doc__)
    compare_parser.add_argument(
        "before",
        help="first CSV file to compare (e.g., develop.csv)",
    )
    compare_parser.add_argument(
        "after",
        help="second CSV file to compare (e.g., pr.csv)",
    )
    compare_parser.add_argument(
        "-o",
        "--output",
        help="output plot file (default: comparison.png)",
        default="comparison.png",
    )
    compare_parser.add_argument(
        "--no-share-y",
        help="do not share y-axis between phase plots",
        action="store_true",
    )

    analyze_parser = sp.add_parser("analyze", help=analyze.__doc__)
    analyze_parser.add_argument(
        "before",
        help="first JSON spec file to analyze (e.g., develop.json)",
    )
    analyze_parser.add_argument(
        "after",
        help="second JSON spec file to analyze (e.g., pr.json)",
    )
    analyze_parser.add_argument(
        "-s",
        "--spec",
        help="specific spec to analyze in detail (e.g., 'zlib')",
        default=None,
    )
    analyze_parser.add_argument(
        "--show-dag",
        help="show DAG trees for both before and after (only with --spec)",
        action="store_true",
    )
    analyze_parser.add_argument(
        "--show-opt",
        help="show optimization criteria (cost vectors) for both before and after (only with --spec)",
        action="store_true",
    )


Record = Tuple[str, str, int, float, float, float, float, float, int]


def _clear_repo_modules():
    """Clear all spack_repo.* modules from sys.modules to force reimport."""
    to_delete = [name for name in sys.modules if name.startswith("spack_repo.")]
    for name in to_delete:
        del sys.modules[name]


def _run_single_solve(
    inputs: Tuple[List[spack.spec.Spec], int, bool, bool],
) -> Tuple[Record, Optional[Dict]]:
    specs, i, clear_repo_modules, include_spec = inputs
    if clear_repo_modules:
        _clear_repo_modules()
    solver = asp.Solver()
    result, timer, _ = solver.driver.solve(
        asp.SpackSolverSetup(),
        specs,
        reuse=solver.selector.reusable_specs(specs),
    )
    assert isinstance(timer, spack.util.timer.Timer)
    timer.stop()
    spec_hash = result.specs[0].dag_hash() if result.specs else ""

    record = (
        str(specs[0]),
        spec_hash,
        i,
        timer.duration("setup"),
        timer.duration("load"),
        timer.duration("ground"),
        timer.duration("solve"),
        timer.duration(),
        len(result.possible_dependencies),
    )

    # Conditionally serialize spec and cost to avoid overhead in CSV mode
    spec_dict = None
    if include_spec and result.specs:
        spec_dict = result.specs[0].to_dict(hash=ht.dag_hash)
        # Extract cost vector from the best answer
        spec_dict["cost"] = result.answers[0][0]  # (cost, index, spec_dict)

        # Capture formatted optimization criteria output using spack's own code
        # This ensures consistency with whatever version of spack is running
        try:
            # Redirect stdout to capture the formatted output with color codes
            old_stdout = sys.stdout
            old_color = color.get_color_when()
            sys.stdout = captured_output = io.StringIO()
            color.set_color_when(True)

            try:
                spack.cmd.solve._process_result(result, ["opt"], None, {})
                spec_dict["opt_output"] = captured_output.getvalue()
            finally:
                sys.stdout = old_stdout
                color.set_color_when(old_color)
        except Exception as e:
            # If capture fails, store a simple fallback
            spec_dict["opt_output"] = f"Cost vector: {result.answers[0][0]}\n(Could not format: {e})"

    return record, spec_dict


def _warmup():
    specs = spack.cmd.parse_specs("hdf5")
    solver = asp.Solver()
    solver.driver.solve(asp.SpackSolverSetup(), specs, reuse=solver.selector.reusable_specs(specs))


def _validate_and_load_csv_files(
    before_file: str, after_file: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate two CSV files for comparison/plotting."""
    # Load the data using the header row
    try:
        before_df = pd.read_csv(before_file)
        after_df = pd.read_csv(after_file)
    except FileNotFoundError as e:
        raise RuntimeError(f"Could not read CSV file: {e}") from e

    # Verify the expected columns exist
    for df, name in [(before_df, before_file), (after_df, after_file)]:
        if list(df.columns) != COLUMNS:
            raise RuntimeError(f"Unexpected CSV format in {name}. Expected columns: {COLUMNS}")

    # Check that both files have the same specs (validate once upfront)
    before_specs = set(before_df["spec"].unique())
    after_specs = set(after_df["spec"].unique())

    if before_specs != after_specs:
        raise RuntimeError(
            f"Specs in {before_file} and {after_file} do not match: "
            f"{before_specs.symmetric_difference(after_specs)}"
        )

    return before_df, after_df


def run(args):
    """run benchmarks and produce a CSV file of timing results"""
    # Solver performance depends on the order of facts. Randomization per spec ensures that the
    # median time is a more reliable measure for comparison of benchmarks.
    os.environ["SPACK_SOLVER_RANDOMIZATION"] = "1"
    input_file = pathlib.Path(args.specfile)
    if not input_file.exists():
        current_dir = pathlib.Path(__file__).parent
        input_file = current_dir / "data" / f"{args.specfile}.txt"

    try:
        spec_strs = [line.strip() for line in input_file.read_text().split("\n") if line.strip()]
    except OSError as e:
        raise RuntimeError(f"Could not read the input spec file: {e}") from e

    tty.info("Warm up...")
    _warmup()

    include_spec = args.spec_output is not None
    input_list = [
        (spack.cmd.parse_specs(spec_str), i, args.clear_repo_modules, include_spec)
        for spec_str in spec_strs
        for i in range(args.repetitions)
    ]

    if not args.no_shuffle:
        random.shuffle(input_list)

    start = time.time()
    pkg_stats: List[Record] = []
    spec_dicts: List[Dict] = []

    if args.nprocess > 1:
        record_iterator = spack.util.parallel.imap_unordered(
            _run_single_solve,
            input_list,
            processes=args.nprocess,
            debug=tty.is_debug(),
            maxtaskperchild=1,
        )
    else:
        record_iterator = map(_run_single_solve, input_list)

    # Process records with unified progress reporting
    tty.info("Benchmarking...")

    for idx, (record, spec_dict) in enumerate(record_iterator):
        pkg_stats.append(record)
        if spec_dict:
            spec_dicts.append(spec_dict)
        tty.msg(f"{record[7]:6.1f}s [{(idx + 1)/len(input_list)*100:3.0f}%] {record[0]}")
        sys.stdout.flush()

    finish = time.time()
    tty.msg(f"Total elapsed time: {finish - start:.2f} seconds")

    # Create DataFrame and write to CSV
    pd.DataFrame(pkg_stats, columns=COLUMNS).to_csv(args.output, index=False)

    # Optionally output specs to a separate file
    if args.spec_output:
        with open(args.spec_output, "w") as f:
            sjson.dump(spec_dicts, f)
        tty.msg(f"Specs written to {args.spec_output}")


def _collect_hash_warnings(
    before_df: pd.DataFrame, after_df: pd.DataFrame, before_file: str, after_file: str
) -> List[str]:
    warnings = []

    # Warning 1: same spec has multiple distinct hashes within the same file
    for df, name in [(before_df, before_file), (after_df, after_file)]:
        for spec, group in df.groupby("spec"):
            hashes = group["hash"].unique()
            if len(hashes) > 1:
                hashes_str = ", ".join(f"`{h}`" for h in sorted(hashes))
                warnings.append(
                    f"`{spec}` has multiple hashes in `{name}` (multiple optimal solutions): "
                    f"{hashes_str}"
                )

    # Warning 2: same spec has a different hash set between the two files
    before_hashes = before_df.groupby("spec")["hash"].apply(set)
    after_hashes = after_df.groupby("spec")["hash"].apply(set)
    for spec in before_hashes.index:
        before_set = before_hashes[spec]
        after_set = after_hashes[spec]
        if before_set.isdisjoint(after_set):
            before_str = ", ".join(f"`{h}`" for h in sorted(before_set))
            after_str = ", ".join(f"`{h}`" for h in sorted(after_set))
            warnings.append(
                f"`{spec}` hash changed between files: "
                f"`{before_file}` has {{{before_str}}}, `{after_file}` has {{{after_str}}}"
            )

    return warnings


def _get_root_spec_key(spec_entry: Dict) -> Tuple[str, str]:
    """Extract (name, version) tuple from root node of a spec."""
    nodes = spec_entry.get("spec", {}).get("nodes", [])
    root = nodes[0]
    name = root.get("name")
    version = root.get("version")
    return (name, version)


def _group_specs_by_root(specs: List[Dict]) -> Dict[Tuple[str, str], List[Dict]]:
    """Group specs by their root (name, version)."""
    groups = {}
    for spec_entry in specs:
        key = _get_root_spec_key(spec_entry)
        if key not in groups:
            groups[key] = []
        groups[key].append(spec_entry)
    return groups


def _validate_matching_specs(
    before_specs: List[Dict], after_specs: List[Dict], before_file: str, after_file: str
) -> None:
    """Validate that both files contain the same set of root specs."""
    def get_root_spec_str(key: Tuple[str, str]) -> str:
        return f"{key[0]}@{key[1]}"

    before_grouped = _group_specs_by_root(before_specs)
    after_grouped = _group_specs_by_root(after_specs)

    if before_grouped.keys() != after_grouped.keys():
        before_spec_strs = {get_root_spec_str(k) for k in before_grouped.keys()}
        after_spec_strs = {get_root_spec_str(k) for k in after_grouped.keys()}
        raise RuntimeError(
            f"Specs in {before_file} and {after_file} do not match: "
            f"{before_spec_strs.symmetric_difference(after_spec_strs)}"
        )


def _find_changed_specs(
    before_specs: List[Dict], after_specs: List[Dict]
) -> Tuple[List[Tuple[Dict, Dict, str]], List[str]]:
    before_grouped = _group_specs_by_root(before_specs)
    after_grouped = _group_specs_by_root(after_specs)

    changed_pairs = []
    unchanged_specs = []

    for key in before_grouped:
        before_spec = before_grouped[key][0]
        after_spec = after_grouped[key][0]
        spec_str = f"{key[0]}@{key[1]}"

        before_hash = before_spec.get("spec", {}).get("nodes", [{}])[0].get("hash")
        after_hash = after_spec.get("spec", {}).get("nodes", [{}])[0].get("hash")

        if before_hash != after_hash:
            changed_pairs.append((before_spec, after_spec, spec_str))
        else:
            unchanged_specs.append(spec_str)

    return changed_pairs, unchanged_specs


def _find_node_compiler(node: Dict, node_map: Dict[str, Dict]) -> str:
    for dep in node.get("dependencies", []):
        virtuals = dep.get("parameters", {}).get("virtuals", [])
        if COMPILER_VIRTUALS & set(virtuals):
            compiler_node = node_map.get(dep.get("hash"))
            if compiler_node:
                return f"{compiler_node.get('name')}@{compiler_node.get('version')}"
    return ""


def _target_to_str(target) -> str:
    if isinstance(target, str):
        return target
    elif isinstance(target, dict):
        return target.get("name", "")
    return ""


def _compare_spec_dags(before_spec_dict: Dict, after_spec_dict: Dict) -> Dict:
    before_nodes = before_spec_dict.get("spec", {}).get("nodes", [])
    after_nodes = after_spec_dict.get("spec", {}).get("nodes", [])

    before_by_hash = {node.get("hash"): node for node in before_nodes}
    after_by_hash = {node.get("hash"): node for node in after_nodes}

    # Use (name, version) as key to handle multiple packages with same name
    before_by_name_version = {(node.get("name"), node.get("version")): node for node in before_nodes}
    after_by_name_version = {(node.get("name"), node.get("version")): node for node in after_nodes}

    before_names = {name for name, _ in before_by_name_version.keys()}
    after_names = {name for name, _ in after_by_name_version.keys()}

    result = {
        "added_packages": [],
        "removed_packages": [],
        "version_changes": [],
        "variant_changes": [],
        "compiler_changes": [],
        "target_changes": [],
    }

    # Find added/removed packages (with versions) as (name, version) tuples
    for name in after_names - before_names:
        versions = [v for n, v in after_by_name_version.keys() if n == name]
        for ver in versions:
            result["added_packages"].append((name, ver))

    for name in before_names - after_names:
        versions = [v for n, v in before_by_name_version.keys() if n == name]
        for ver in versions:
            result["removed_packages"].append((name, ver))

    # For common package names, compare each (name, version) combination
    common_names = before_names & after_names
    for pkg_name in common_names:
        before_versions = {v for n, v in before_by_name_version.keys() if n == pkg_name}
        after_versions = {v for n, v in after_by_name_version.keys() if n == pkg_name}

        removed_versions = before_versions - after_versions
        added_versions = after_versions - before_versions

        # If there's exactly one version in each and they differ, report as version change
        if len(before_versions) == 1 and len(after_versions) == 1 and removed_versions and added_versions:
            old_ver = list(removed_versions)[0]
            new_ver = list(added_versions)[0]
            result["version_changes"].append((pkg_name, old_ver, new_ver))
        else:
            # Report individual additions/removals
            for ver in removed_versions:
                result["version_changes"].append((pkg_name, ver, None))
            for ver in added_versions:
                result["version_changes"].append((pkg_name, None, ver))

        # For versions that exist in both, compare other attributes
        for ver in before_versions & after_versions:
            before_node = before_by_name_version[(pkg_name, ver)]
            after_node = after_by_name_version[(pkg_name, ver)]

            before_params = before_node.get("parameters", {})
            after_params = after_node.get("parameters", {})
            for param_key in set(before_params.keys()) | set(after_params.keys()):
                if param_key in ["cflags", "cppflags", "cxxflags", "fflags", "ldflags", "ldlibs"]:
                    continue
                if before_params.get(param_key) != after_params.get(param_key):
                    result["variant_changes"].append(((pkg_name, ver), param_key, before_params.get(param_key), after_params.get(param_key)))

            before_compiler = _find_node_compiler(before_node, before_by_hash)
            after_compiler = _find_node_compiler(after_node, after_by_hash)
            if before_compiler != after_compiler and (before_compiler or after_compiler):
                result["compiler_changes"].append(((pkg_name, ver), before_compiler, after_compiler))

            before_target = _target_to_str(before_node.get("arch", {}).get("target"))
            after_target = _target_to_str(after_node.get("arch", {}).get("target"))
            if before_target != after_target and before_target and after_target:
                result["target_changes"].append(((pkg_name, ver), before_target, after_target))

    return result


def _print_spec_changes(spec_name: str, changes: Dict) -> None:
    tty.msg(f"Changes in {spec_name}")

    if changes["added_packages"]:
        color.cprint("  @G{Added packages (%d)}" % len(changes['added_packages']))
        for name, ver in sorted(changes["added_packages"]):
            color.cprint("    @G{+} %s@@@c{%s}" % (name, ver))
        print()

    if changes["removed_packages"]:
        color.cprint("  @R{Removed packages (%d)}" % len(changes['removed_packages']))
        for name, ver in sorted(changes["removed_packages"]):
            color.cprint("    @R{-} %s@@@c{%s}" % (name, ver))
        print()

    if changes["version_changes"]:
        color.cprint("  @C{Version changes (%d)}" % len(changes['version_changes']))
        sorted_changes = sorted(changes["version_changes"], key=lambda x: (x[0], x[1] or "", x[2] or ""))
        for pkg, old_ver, new_ver in sorted_changes:
            if old_ver is None:
                color.cprint("    - %s@@@c{%s} @G{(added)}" % (pkg, new_ver))
            elif new_ver is None:
                color.cprint("    - %s@@@c{%s} @R{(removed)}" % (pkg, old_ver))
            else:
                color.cprint("    - %s: @c{%s} → @c{%s}" % (pkg, old_ver, new_ver))
        print()

    if changes["variant_changes"]:
        color.cprint("  @C{Variant changes (%d)}" % len(changes['variant_changes']))
        for (name, ver), variant, old_val, new_val in sorted(changes["variant_changes"]):
            color.cprint("    - %s@@@c{%s}: %s=%s → %s=%s" % (name, ver, variant, old_val, variant, new_val))
        print()

    if changes["compiler_changes"]:
        color.cprint("  @C{Compiler changes (%d)}" % len(changes['compiler_changes']))
        for (name, ver), old_compiler, new_compiler in sorted(changes["compiler_changes"]):
            color.cprint("    - %s@@@c{%s}: %s → %s" % (name, ver, color.cescape(old_compiler), color.cescape(new_compiler)))
        print()

    if changes["target_changes"]:
        color.cprint("  @C{Target changes (%d)}" % len(changes['target_changes']))
        for (name, ver), old_target, new_target in sorted(changes["target_changes"]):
            color.cprint("    - %s@@@c{%s}: %s → %s" % (name, ver, old_target, new_target))
        print()

    if not any([changes["added_packages"], changes["removed_packages"], changes["version_changes"],
                changes["variant_changes"], changes["compiler_changes"], changes["target_changes"]]):
        print("  No changes detected")
        print()


def compare(args) -> None:
    """Compare two CSV files to see whether one is faster than the other and generate a plot."""
    before_df, after_df = _validate_and_load_csv_files(args.before, args.after)

    print("## Warnings\n")
    warnings = _collect_hash_warnings(before_df, after_df, args.before, args.after)
    if warnings:
        for w in warnings:
            print(f"* {w}")
    else:
        print("No warnings.")
    print()

    significant: Dict[str, Tuple[str, float]] = {}
    before = before_df.groupby("spec")[TIMING_COLS].median()
    after = after_df.groupby("spec")[TIMING_COLS].median()

    print("## Performance comparison\n")

    for field in TIMING_COLS:
        # Calculate change in median time
        comparison = pd.DataFrame({"median_before": before[field], "median_after": after[field]})
        comparison["ratio"] = comparison["median_after"] / comparison["median_before"]
        comparison["change_percent"] = (comparison["ratio"] - 1) * 100

        # Statistical Testing using Wilcoxon signed-rank test
        # Null hypothesis: median of log of ratios is 0 (i.e., median of ratios is 1, no change)
        # alternative="less" tests if things got faster (ratios < 1, log of ratios < 0)
        log_ratios = np.log(comparison["ratio"].to_numpy())

        test_improvement = wilcoxon(log_ratios, alternative="less")
        test_regression = wilcoxon(log_ratios, alternative="greater")

        if test_improvement.pvalue < ALPHA:
            result = "improvement"
            significant[field] = (result, test_improvement.pvalue)
        elif test_regression.pvalue < ALPHA:
            result = "regression"
            significant[field] = (result, test_regression.pvalue)
        else:
            result = "not significant"

        print(
            f"**{field}**: {result} (p_improvement = {test_improvement.pvalue:.4f}, "
            f"p_regression = {test_regression.pvalue:.4f})"
        )
        print(comparison.round(2).to_markdown())
        print()

    print("## Summary\n" f"Statistically significant ({len(significant)} fields):")
    for field, (result, p_value) in significant.items():
        print(f"* {field}: {result} (p = {p_value:.4f})")

    # Generate plot
    print(f"\n<!-- generating plot: {args.output} -->")

    # Add source column and combine dataframes
    before_df["source"] = 0
    after_df["source"] = 1
    combined = pd.concat([before_df, after_df])

    # Group by spec and source, calculate statistics
    df = combined.groupby(["spec", "source"])[TIMING_COLS].describe()

    fig = plt.figure(figsize=(20, 10), layout="constrained")
    setup_ax = plt.subplot2grid((2, 3), (1, 0), fig=fig)
    sharey = None if args.no_share_y else setup_ax

    axes = {
        "total": plt.subplot2grid((2, 3), (0, 0), colspan=3, fig=fig),
        "setup": setup_ax,
        "ground": plt.subplot2grid((2, 3), (1, 1), fig=fig, sharey=sharey),
        "solve": plt.subplot2grid((2, 3), (1, 2), fig=fig, sharey=sharey),
    }

    for col, ax in axes.items():
        col_stats = df.xs(col, level=0, axis=1)
        medians = col_stats["50%"].unstack(level="source")
        mins = col_stats["min"].unstack(level="source")
        maxs = col_stats["max"].unstack(level="source")
        error_bars = np.stack(((medians - mins).T, (maxs - medians).T), axis=1)
        details = significant.get(col, None)
        if details is not None:
            title = f"**{col.capitalize()} ({details[0]})**"
        else:
            title = f"{col.capitalize()} (not significant)"

        medians.plot(
            ax=ax,
            kind="bar",
            width=0.9,
            title=title,
            yerr=error_bars,
            capsize=3 if col == "total" else 1,
            error_kw={"capthick": 1, "elinewidth": 0.5},
            alpha=0.7,
        )

        ax.set(xlabel=None, ylabel="Time [s]")
        ax.legend(["before", "after"])
        ax.grid(True, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=90)

    plt.savefig(args.output)


def analyze(args) -> None:
    """Analyze two JSON to see what changed between spec concretization"""
    def load_spec_file(filepath: str) -> List[Dict]:
        spec_path = pathlib.Path(filepath)
        if not spec_path.exists():
            raise RuntimeError(f"File not found: {filepath}")

        with open(spec_path, "r") as f:
            return sjson.load(f)

    def check_unique(specs: List[Dict], label: str) -> None:
        # Group specs by (name, version)
        spec_groups = {}
        for idx, spec_entry in enumerate(specs):
            # Get root node from the spec dict
            nodes = spec_entry.get("spec", {}).get("nodes", [])
            spec_node = nodes[0]
            name = spec_node.get("name")
            version = spec_node.get("version")

            if name and version:
                key = (name, version)
                if key not in spec_groups:
                    spec_groups[key] = []
                spec_groups[key].append((idx, spec_node))

        tty.msg(f"{label}")

        not_unique = False
        for (name, version), entries in sorted(spec_groups.items()):
            # Compare package_hashes for this (name, version) combination
            package_hashes = set()
            for idx, spec_node in entries:
                package_hash = spec_node.get("package_hash")
                if package_hash:
                    package_hashes.add(package_hash)

            if len(package_hashes) > 1:
                not_unique = True
                print(f"{name}@{version}: Different specs across {len(entries)} trials")
                for idx, spec_node in entries:
                    package_hash = spec_node.get("package_hash")
                    print(f"  Trial {idx}: {package_hash}")

        if not not_unique:
            print("All specs are the same across trials.")
        print()

    # Load and analyze both files
    before_specs = load_spec_file(args.before)
    after_specs = load_spec_file(args.after)

    if args.spec:
        # Detailed mode: analyze specific spec
        if args.show_dag and not args.spec:
            raise RuntimeError("--show-dag requires --spec to be specified")
        if args.show_opt and not args.spec:
            raise RuntimeError("--show-opt requires --spec to be specified")

        _validate_matching_specs(before_specs, after_specs, args.before, args.after)
        changed_pairs, _ = _find_changed_specs(before_specs, after_specs)

        # Find the requested spec
        found = False
        for before_spec, after_spec, spec_str in changed_pairs:
            spec_name = _get_root_spec_key(before_spec)[0]
            if spec_name == args.spec or spec_str.startswith(f"{args.spec}@"):
                found = True

                if args.show_dag:
                    # Convert to Spack Spec objects and print DAG with non-default highlighting
                    before_spack_spec = spack.spec.Spec.from_dict(before_spec)
                    after_spack_spec = spack.spec.Spec.from_dict(after_spec)

                    tree_kwargs = {
                        "color": sys.stdout.isatty(),
                        "format": spack.spec.DISPLAY_FORMAT,
                        "hashes": False,
                        "highlight_version_fn": spack.package_base.non_preferred_version,
                        "highlight_variant_fn": spack.package_base.non_default_variant,
                    }

                    tty.msg(f"Before DAG: {spec_str}")
                    sys.stdout.write(spack.spec.tree([before_spack_spec], **tree_kwargs))
                    print()

                    tty.msg(f"After DAG: {spec_str}")
                    sys.stdout.write(spack.spec.tree([after_spack_spec], **tree_kwargs))
                    print()

                if args.show_opt:
                    # Print optimization criteria for before
                    before_opt = before_spec.get("opt_output")
                    if before_opt:
                        tty.msg(f"Before optimization criteria: {spec_str}")
                        print(before_opt, end="")
                    else:
                        tty.warn(f"No optimization output stored for before spec {spec_str}")
                        before_cost = before_spec.get("cost", [])
                        if before_cost:
                            color.cprint(f"Cost vector: {before_cost}")
                        print()

                    # Print optimization criteria for after
                    after_opt = after_spec.get("opt_output")
                    if after_opt:
                        tty.msg(f"After optimization criteria: {spec_str}")
                        print(after_opt, end="")
                    else:
                        tty.warn(f"No optimization output stored for after spec {spec_str}")
                        after_cost = after_spec.get("cost", [])
                        if after_cost:
                            color.cprint(f"Cost vector: {after_cost}")
                        print()

                changes = _compare_spec_dags(before_spec, after_spec)
                _print_spec_changes(spec_str, changes)
                break

        if not found:
            # Check if spec exists but unchanged
            for key in _group_specs_by_root(before_specs).keys():
                if key[0] == args.spec:
                    tty.msg(f"No changes detected for {key[0]}@{key[1]}")
                    return

            raise RuntimeError(f"Spec '{args.spec}' not found in the input files")
    else:
        # High-level mode: show which specs changed
        check_unique(before_specs, f"Before ({args.before})")
        check_unique(after_specs, f"After ({args.after})")

        _validate_matching_specs(before_specs, after_specs, args.before, args.after)
        changed_pairs, unchanged_specs = _find_changed_specs(before_specs, after_specs)

        tty.msg("Comparison")
        if changed_pairs:
            print(f"Changed specs ({len(changed_pairs)} of {len(changed_pairs) + len(unchanged_specs)}):")
            for _, _, spec_str in sorted(changed_pairs, key=lambda x: x[2]):
                print(f"  - {spec_str}")
            print()

        if unchanged_specs:
            print(f"Unchanged specs ({len(unchanged_specs)} of {len(changed_pairs) + len(unchanged_specs)}):")
            for spec_str in sorted(unchanged_specs):
                print(f"  - {spec_str}")
            print()

        if changed_pairs:
            print(f"Use --spec <name> to see detailed changes for a specific spec.")


def solve_benchmark(parser, args):
    action = {"run": run, "compare": compare, "analyze": analyze}
    return action[args.subcommand](args)
