# -*- coding: utf-8 -*-
"""
experiments.py
--------------
اجرای آزمایش‌ها روی بنچمارک‌ها و تولید خروجی قابل ارائه.

این فایل باید:
- EPC را روی Sphere و Rosenbrock اجرا کند
- تاریخچه best_f را برای نمودار همگرایی نگه دارد
- (اختیاری) چند اجرا با seedهای مختلف بزند و آمار بدهد
"""

from __future__ import annotations

from typing import Callable, Any, List, Dict
import numpy as np
import matplotlib.pyplot as plt

from epc import EPCConfig, epc_optimize, EPCResult
from objectives import sphere, rosenbrock


# ------------------------------------------------------------
# اجرای یک آزمایش و نمایش نتایج
# ------------------------------------------------------------
def run_single_experiment(
    name: str,
    obj_func: Callable[[np.ndarray], float],
    D: int,
    lb: Any,
    ub: Any,
    cfg: EPCConfig,
    seed: int
) -> EPCResult:
    """
    یک اجرا از EPC برای یک تابع هدف.

    کارهایی که انجام می‌دهد:
    - epc_optimize را صدا می‌زند
    - نتیجه را چاپ می‌کند
    - history را برمی‌گرداند تا بتوانیم plot کنیم
    """
    res = epc_optimize(obj_func=obj_func, D=D, lb=lb, ub=ub, config=cfg, seed=seed)

    print(f"\n=== {name} ===")
    print("best_f:", res.best_f)
    print("best_x (first 5 dims):", res.best_x[:5] if res.best_x.size >= 5 else res.best_x)
    print("elapsed_sec:", res.elapsed_sec)
    print("meta:", res.meta)

    return res


# ------------------------------------------------------------
# چند اجرا (برای گزارش آماری)
# ------------------------------------------------------------
def run_multiple_trials(
    name: str,
    obj_func: Callable[[np.ndarray], float],
    D: int,
    lb: Any,
    ub: Any,
    cfg: EPCConfig,
    seeds: List[int]
) -> Dict[str, Any]:
    """
    چند بار اجرای EPC با seedهای مختلف و گرفتن آمار.

    خروجی مناسب برای گزارش پروژه:
    - mean/std/min/max از best_f
    """
    best_fs = []
    results: List[EPCResult] = []

    for s in seeds:
        res = epc_optimize(obj_func=obj_func, D=D, lb=lb, ub=ub, config=cfg, seed=s)
        results.append(res)
        best_fs.append(res.best_f)

    best_fs = np.array(best_fs, dtype=float)

    summary = {
        "name": name,
        "seeds": seeds,
        "mean_best_f": float(np.mean(best_fs)),
        "std_best_f": float(np.std(best_fs)),
        "min_best_f": float(np.min(best_fs)),
        "max_best_f": float(np.max(best_fs)),
        "best_f_values": best_fs,
        "results": results
    }
    return summary


# ------------------------------------------------------------
# رسم نمودار همگرایی
# ------------------------------------------------------------
def plot_convergence(title: str, history: np.ndarray) -> None:
    """
    رسم نمودار best_f برحسب iteration.
    """
    plt.figure()
    plt.plot(history)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Best f(x)")
    plt.grid(True)


# ------------------------------------------------------------
# نقطه شروع اجرای فایل
# ------------------------------------------------------------
if __name__ == "__main__":
    # تنظیمات الگوریتم (می‌توانی تغییر بدهی)
    cfg = EPCConfig(
        N=25,
        Tmax=150,
        epsilon=1e-10,
        verbose=True  # برای دیدن روند
    )

    # 1) Sphere
    res_sphere = run_single_experiment(
        name="Sphere",
        obj_func=sphere,
        D=10,
        lb=-5.0,
        ub=5.0,
        cfg=cfg,
        seed=42
    )
    plot_convergence("EPC Convergence - Sphere", res_sphere.history_best_f)

    # 2) Rosenbrock
    res_rosen = run_single_experiment(
        name="Rosenbrock",
        obj_func=rosenbrock,
        D=5,
        lb=-2.0,
        ub=2.0,
        cfg=cfg,
        seed=7
    )
    plot_convergence("EPC Convergence - Rosenbrock", res_rosen.history_best_f)

    # (اختیاری) چند اجرا برای آمار
    summary = run_multiple_trials(
        name="Sphere (multi-trial)",
        obj_func=sphere,
        D=10,
        lb=-5.0,
        ub=5.0,
        cfg=cfg,
        seeds=[0, 1, 2, 3, 4]
    )
    print("\n=== Multi-trial summary ===")
    print("mean_best_f:", summary["mean_best_f"])
    print("std_best_f :", summary["std_best_f"])
    print("min_best_f :", summary["min_best_f"])
    print("max_best_f :", summary["max_best_f"])

    plt.show()
