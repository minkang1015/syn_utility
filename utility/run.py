from __future__ import annotations

import argparse
from pathlib import Path

from utility.datasets import get_dataset_loader
from utility.evaluation import evaluate
from utility.utils import load_json, parse_int_list, parse_ratios_from_bin


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Synthetic utility evaluator (JSON config-driven)")
    p.add_argument("--config", type=str, required=True, help="Path to config .json")
    return p


def main():
    args = build_parser().parse_args()
    cfg = load_json(args.config)

    data_root = Path(cfg.get("data_root", "./datas"))
    outdir = Path(cfg.get("outdir", "./outputs"))
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = cfg["dataset"]
    method = cfg["method"]
    run = str(cfg.get("run", "1"))
    sample = str(cfg.get("sample", "sample1"))

    mode = cfg.get("mode", "tmtr")
    seed = int(cfg.get("seed", 42))
    test_ratio = float(cfg.get("test_ratio", 0.2))
    granularity = int(cfg.get("granularity", 10))

    dates_cfg = cfg.get("dates", "")
    dates = parse_int_list(dates_cfg) if str(dates_cfg).strip() else None

    bin_size = float(cfg.get("bin", 0.1))
    ratio_max = float(cfg.get("ratio_max", 1.0))
    ratios = parse_ratios_from_bin(bin_size, ratio_max=ratio_max)

    loader = get_dataset_loader(dataset)
    real_df, info = loader(data_root, split="original", dates=dates)
    syn_df, _ = loader(data_root, split="synthetic", method=method, run=run, sample=sample, dates=dates)

    metrics_df = evaluate(
        real_df=real_df,
        syn_df=syn_df,
        ratios=ratios,
        info=info,
        seed=seed,
        test_ratio=test_ratio,
        mode=mode,
        granularity=granularity,
    )

    # meta columns for aggregation later
    metrics_df.insert(0, "dataset", dataset)
    metrics_df.insert(1, "method", method)
    metrics_df.insert(2, "mode", mode)
    metrics_df.insert(3, "run", run)
    metrics_df.insert(4, "sample", sample)
    metrics_df.insert(5, "seed", seed)
    metrics_df.insert(6, "test_ratio", test_ratio)

    out_path = outdir / f"{dataset}__{method}__{mode}__run{run}__{sample}__seed{seed}.csv"
    metrics_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path.resolve()}")
    print(metrics_df)

if __name__ == "__main__":
    main()
