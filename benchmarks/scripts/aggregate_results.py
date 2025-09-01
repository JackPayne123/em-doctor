import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def gather_capability(cap_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not cap_root.exists():
        return pd.DataFrame()

    for model_dir in cap_root.glob("*/*/*"):  # capability/<tag>/<model>
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        tag = model_dir.parent.name
        # lm-eval writes a summary json
        summary_json = model_dir / "results.json"
        if not summary_json.exists():
            # new harness sometimes uses eval_logs or main json; attempt fallback
            for alt in ["results.json", "eval_results.json", "summary.json"]:
                altp = model_dir / alt
                if altp.exists():
                    summary_json = altp
                    break
        if not summary_json.exists():
            continue
        try:
            data = json.loads(summary_json.read_text())
        except Exception:
            continue

        # Harmonize metrics: expect dict of tasks with accuracy-like metrics
        for task, task_result in data.get("results", {}).items():
            # Try common keys
            for metric_key in [
                "acc",
                "exact_match",
                "acc_norm",
                "f1",
                "accuracy",
                "aggregate",
            ]:
                if metric_key in task_result:
                    rows.append({
                        "tag": tag,
                        "model": model,
                        "task": task,
                        "metric": metric_key,
                        "value": task_result[metric_key],
                    })
                    break
    return pd.DataFrame(rows)


def gather_performance(perf_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not perf_root.exists():
        return pd.DataFrame()

    for model_dir in perf_root.glob("*/*/*"):  # performance/<tag>/<model>
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        tag = model_dir.parent.name
        res_path = model_dir / "results.json"
        if not res_path.exists():
            continue
        try:
            data = json.loads(res_path.read_text())
        except Exception:
            continue
        lat = data.get("latency_s", {})
        rows.append({
            "tag": tag,
            "model": model,
            "throughput_rps": data.get("throughput_rps"),
            "throughput_tokens_per_s": data.get("throughput_tokens_per_s"),
            "latency_p50_s": lat.get("p50"),
            "latency_p95_s": lat.get("p95"),
            "latency_mean_s": lat.get("mean"),
            "num_requests": data.get("num_requests"),
            "wall_time_s": data.get("wall_time_s"),
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate capability and performance results")
    parser.add_argument("--results_root", type=str, default="/workspace/em-doctor/benchmarks/results")
    args = parser.parse_args()

    root = Path(args.results_root)

    cap_df = gather_capability(root / "capability")
    perf_df = gather_performance(root / "performance")

    # Save CSVs
    out_cap = root / "capability_aggregated.csv"
    out_perf = root / "performance_aggregated.csv"
    if not cap_df.empty:
        cap_df.to_csv(out_cap, index=False)
    if not perf_df.empty:
        perf_df.to_csv(out_perf, index=False)

    # Produce simple Markdown summaries by latest tag
    def latest_tag(df: pd.DataFrame) -> str | None:
        if df.empty:
            return None
        tags = df["tag"].unique().tolist()
        tags.sort()
        return tags[-1]

    md = []

    lt_cap = latest_tag(cap_df)
    if lt_cap:
        md.append(f"## Capability (latest tag: {lt_cap})")
        cap_latest = cap_df[cap_df["tag"] == lt_cap]
        pivot = cap_latest.pivot_table(index=["model", "task"], columns="metric", values="value", aggfunc="mean").reset_index()
        md.append(pivot.to_markdown(index=False))

    lt_perf = latest_tag(perf_df)
    if lt_perf:
        md.append(f"\n## Performance (latest tag: {lt_perf})")
        perf_latest = perf_df[perf_df["tag"] == lt_perf]
        md.append(perf_latest.to_markdown(index=False))

    summary_md = "\n\n".join(md) if md else "No results found. Run benchmarks first."
    (root / "SUMMARY.md").write_text(summary_md)

    print(f"Wrote: {out_cap if not cap_df.empty else '(no capability CSV)'}")
    print(f"Wrote: {out_perf if not perf_df.empty else '(no performance CSV)'}")
    print(f"Wrote: {root / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
