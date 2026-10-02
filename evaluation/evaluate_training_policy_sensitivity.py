#!/usr/bin/env python3
"""Training-policy sensitivity: Explicit-only vs retrained U-Zero / U-One.

This is not the evaluation-only recoding analysis
(evaluate_uncertainty_endpoint_sensitivity.py). Models here were retrained
with CheXpert U-Zero or U-One labels. The outer-test radiographs and their
explicit 0/1 labels stay the ones used in the original 70-run.

Deltas use a paired patient-level cluster bootstrap (2,000 resamples):
each replicate draws the same patients for both policies.
"""

from __future__ import annotations

import os

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_cluster_bootstrap_delta import (  # noqa: E402
    bootstrap_p_value,
    draw,
    fast_ap,
    fast_auc,
    image_key,
    percentile_ci,
    subject_blocks,
)

STUDY_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", STUDY_DIR / "results_pneumonia")).resolve()
N_BOOT = 2000
BOOT_SEED = 42
POLICIES = ("u_zero", "u_one")


def load_outer_table(run_dir: Path) -> dict:
    pred_path = run_dir / "outer_test" / "outer_test_predictions.csv"
    if not pred_path.is_file():
        raise RuntimeError(f"outer-test predictions가 없습니다: {pred_path}")
    table = {}
    with pred_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = image_key(row["img_path"])
            if key in table:
                raise RuntimeError(f"예측 CSV에 같은 영상이 두 번 있습니다: {pred_path} {key}")
            table[key] = (int(float(row["label"])), float(row["prob_ensemble"]))
    if not table:
        raise RuntimeError(f"예측 CSV가 비어 있습니다: {pred_path}")
    return table


def run_config(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"summary.json이 없습니다: {run_dir}")
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return summary.get("config") or {}


def iter_run_dirs(root: Path):
    if not root.is_dir():
        raise FileNotFoundError(f"결과 폴더가 없습니다: {root}")
    for summary_path in sorted(root.glob("*/summary.json")):
        yield summary_path.parent


def select_runs(root: Path, *, views, modes, arches, policy: str) -> dict:
    """(view, mode, arch) -> run dir. 같은 키가 둘이면 멈춥니다."""
    chosen = {}
    for run_dir in iter_run_dirs(root):
        cfg = run_config(run_dir)
        view = str(cfg.get("view") or "")
        mode = str(cfg.get("data_mode") or "")
        arch = str(cfg.get("arch") or "")
        if view not in views or mode not in modes or arch not in arches:
            continue
        got = str(cfg.get("uncertainty_policy") or "explicit_only")
        if got != policy:
            continue
        missing = [key for key in ("seed", "split_seed") if key not in cfg]
        if missing:
            raise RuntimeError(f"{run_dir / 'summary.json'} config에 {', '.join(missing)}가 없습니다.")
        if int(cfg["seed"]) != 42 or int(cfg["split_seed"]) != 42:
            continue
        key = (str(cfg["view"]), str(cfg["data_mode"]), str(cfg["arch"]))
        if key in chosen:
            raise RuntimeError(
                f"{policy} run이 중복입니다: {key}\n  {chosen[key]}\n  {run_dir}"
            )
        chosen[key] = run_dir
    return chosen


def marginal_ci(y, prob, blocks, n_boot: int, seed: int):
    order, counts, offsets, n_subjects = blocks
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot)
    aps = np.empty(n_boot)
    filled = 0
    for _ in range(n_boot):
        picked = draw(order, counts, offsets, n_subjects, rng)
        y_b = y[picked]
        if y_b.min() == y_b.max():
            continue
        aucs[filled] = fast_auc(y_b, prob[picked])
        aps[filled] = fast_ap(y_b, prob[picked])
        filled += 1
    if filled == 0:
        raise RuntimeError("bootstrap 표본이 모두 한 클래스라서 CI를 계산할 수 없습니다.")
    return percentile_ci(aucs[:filled]), percentile_ci(aps[:filled]), filled


def paired_delta(y, prob_base, prob_new, blocks, n_boot: int, seed: int):
    order, counts, offsets, n_subjects = blocks
    rng = np.random.default_rng(seed)
    d_auc = np.empty(n_boot)
    d_ap = np.empty(n_boot)
    filled = 0
    for _ in range(n_boot):
        picked = draw(order, counts, offsets, n_subjects, rng)
        y_b = y[picked]
        if y_b.min() == y_b.max():
            continue
        d_auc[filled] = fast_auc(y_b, prob_new[picked]) - fast_auc(y_b, prob_base[picked])
        d_ap[filled] = fast_ap(y_b, prob_new[picked]) - fast_ap(y_b, prob_base[picked])
        filled += 1
    if filled == 0:
        raise RuntimeError("paired bootstrap 표본이 모두 한 클래스라서 CI를 계산할 수 없습니다.")
    return d_auc[:filled], d_ap[:filled]


def aligned_arrays(base_table: dict, other_table: dict, *, label: str):
    if set(base_table) != set(other_table):
        only_base = sorted(set(base_table) - set(other_table))[:5]
        only_other = sorted(set(other_table) - set(base_table))[:5]
        raise RuntimeError(
            f"{label}: outer-test 영상이 Explicit-only와 다릅니다. "
            f"only_in_explicit={only_base}, only_in_policy={only_other}"
        )
    keys = sorted(base_table)
    y_base = np.asarray([base_table[k][0] for k in keys], dtype=np.int8)
    y_other = np.asarray([other_table[k][0] for k in keys], dtype=np.int8)
    if not np.array_equal(y_base, y_other):
        raise RuntimeError(f"{label}: outer-test 라벨이 Explicit-only와 다릅니다. 평가 endpoint가 바뀌었습니다.")
    if set(np.unique(y_base).tolist()) - {0, 1}:
        raise RuntimeError(f"{label}: outer-test 라벨에 0/1 이외의 값이 있습니다.")
    prob_base = np.asarray([base_table[k][1] for k in keys], dtype=np.float64)
    prob_other = np.asarray([other_table[k][1] for k in keys], dtype=np.float64)
    return keys, y_base, prob_base, prob_other


def blank_delta():
    return {
        "delta_auroc": "",
        "delta_auroc_lo": "",
        "delta_auroc_hi": "",
        "delta_auroc_p": "",
        "delta_auprc": "",
        "delta_auprc_lo": "",
        "delta_auprc_hi": "",
        "delta_auprc_p": "",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicit-only 대비 U-Zero/U-One 재학습의 paired bootstrap 표"
    )
    parser.add_argument("--baseline-root", type=Path, default=RESULTS_DIR)
    parser.add_argument("--policy-root", type=Path, default=STUDY_DIR / "results_pneumonia_trainpolicy")
    parser.add_argument("--views", nargs="+", default=["AP", "PA"], choices=["AP", "PA"])
    parser.add_argument("--modes", nargs="+", default=["raw", "medsam3_crop"])
    parser.add_argument("--arches", nargs="+", default=["resnet152", "eva_x_base"])
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=BOOT_SEED)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="기본: {policy-root}/analysis_outputs/training_policy_sensitivity",
    )
    args = parser.parse_args()

    views = set(args.views)
    modes = set(args.modes)
    arches = set(args.arches)
    baseline = select_runs(
        args.baseline_root, views=views, modes=modes, arches=arches, policy="explicit_only"
    )
    policies = {
        policy: select_runs(args.policy_root, views=views, modes=modes, arches=arches, policy=policy)
        for policy in POLICIES
    }
    expected = {(view, mode, arch) for view in args.views for mode in args.modes for arch in args.arches}
    missing_base = sorted(expected - set(baseline))
    if missing_base:
        raise RuntimeError(f"Explicit-only 기준선 run이 없습니다: {missing_base}")
    for policy, found in policies.items():
        missing = sorted(expected - set(found))
        if missing:
            raise RuntimeError(f"{policy} run이 없습니다: {missing}")

    rows = []
    contrast_rows = []
    for view, mode, arch in sorted(expected):
        base_dir = baseline[(view, mode, arch)]
        base_table = load_outer_table(base_dir)
        keys = sorted(base_table)
        y = np.asarray([base_table[k][0] for k in keys], dtype=np.int8)
        prob = np.asarray([base_table[k][1] for k in keys], dtype=np.float64)
        blocks = subject_blocks(keys)
        auc_ci, ap_ci, n_eff = marginal_ci(y, prob, blocks, args.n_boot, args.seed)
        n_subjects = int(blocks[3])
        rows.append({
            "analysis": "training_policy_sensitivity",
            "view": view,
            "model": arch,
            "input": mode,
            "training_policy": "Explicit-only",
            "n_images": len(keys),
            "n_subjects": n_subjects,
            "auroc": f"{fast_auc(y, prob):.6f}",
            "auroc_lo": f"{auc_ci[0]:.6f}",
            "auroc_hi": f"{auc_ci[1]:.6f}",
            "auprc": f"{fast_ap(y, prob):.6f}",
            "auprc_lo": f"{ap_ci[0]:.6f}",
            "auprc_hi": f"{ap_ci[1]:.6f}",
            **blank_delta(),
            "delta_reference": "Ref.",
            "n_boot_effective": n_eff,
            "run_dir": str(base_dir),
        })

        for policy in POLICIES:
            policy_dir = policies[policy][(view, mode, arch)]
            other = load_outer_table(policy_dir)
            label = f"{view}/{arch}/{mode}/{policy}"
            keys_p, y_p, prob_base, prob_new = aligned_arrays(base_table, other, label=label)
            blocks_p = subject_blocks(keys_p)
            auc_ci, ap_ci, n_eff = marginal_ci(y_p, prob_new, blocks_p, args.n_boot, args.seed)
            d_auc, d_ap = paired_delta(y_p, prob_base, prob_new, blocks_p, args.n_boot, args.seed)
            auc_lo, auc_hi = percentile_ci(d_auc)
            ap_lo, ap_hi = percentile_ci(d_ap)
            point_auc = float(fast_auc(y_p, prob_new) - fast_auc(y_p, prob_base))
            point_ap = float(fast_ap(y_p, prob_new) - fast_ap(y_p, prob_base))
            policy_name = "U-Zero" if policy == "u_zero" else "U-One"
            rows.append({
                "analysis": "training_policy_sensitivity",
                "view": view,
                "model": arch,
                "input": mode,
                "training_policy": policy_name,
                "n_images": len(keys_p),
                "n_subjects": int(blocks_p[3]),
                "auroc": f"{fast_auc(y_p, prob_new):.6f}",
                "auroc_lo": f"{auc_ci[0]:.6f}",
                "auroc_hi": f"{auc_ci[1]:.6f}",
                "auprc": f"{fast_ap(y_p, prob_new):.6f}",
                "auprc_lo": f"{ap_ci[0]:.6f}",
                "auprc_hi": f"{ap_ci[1]:.6f}",
                "delta_auroc": f"{point_auc:.6f}",
                "delta_auroc_lo": f"{auc_lo:.6f}",
                "delta_auroc_hi": f"{auc_hi:.6f}",
                "delta_auroc_p": f"{bootstrap_p_value(d_auc):.6f}",
                "delta_auprc": f"{point_ap:.6f}",
                "delta_auprc_lo": f"{ap_lo:.6f}",
                "delta_auprc_hi": f"{ap_hi:.6f}",
                "delta_auprc_p": f"{bootstrap_p_value(d_ap):.6f}",
                "delta_reference": "Explicit-only",
                "n_boot_effective": int(d_auc.size),
                "run_dir": str(policy_dir),
            })
            contrast_rows.append(rows[-1])

    out_dir = args.out_dir or (args.policy_root / "analysis_outputs" / "training_policy_sensitivity")
    table_path = out_dir / "training_policy_sensitivity_table.csv"
    contrast_path = out_dir / "training_policy_sensitivity_contrasts.csv"
    write_csv(table_path, rows)
    write_csv(contrast_path, contrast_rows)
    note = {
        "analysis": "training_policy_sensitivity",
        "not_this_analysis": "evaluation-only U-Zero/U-One recoding (evaluate_uncertainty_endpoint_sensitivity.py)",
        "question": "U-Zero/U-One으로 모델을 다시 학습하면 Explicit-only 대비 outer-test 성능이 얼마나 달라지는가",
        "endpoint": "original explicit 0/1 outer-test radiographs, same patients",
        "delta_interval": "paired patient-level cluster bootstrap",
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "n_rows": len(rows),
        "table": str(table_path),
    }
    with (out_dir / "training_policy_sensitivity_meta.json").open("w", encoding="utf-8") as f:
        json.dump(note, f, indent=2, ensure_ascii=False)
    print(f"[DONE] {table_path} rows={len(rows)}")
    print(f"[DONE] {contrast_path} rows={len(contrast_rows)}")


if __name__ == "__main__":
    main()
