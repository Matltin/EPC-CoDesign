# -*- coding: utf-8 -*-
"""
experiments.py
--------------
اجرای آزمایش‌ها و چاپ لاگ مطابق نمونه out.txt
خروجی شامل:
1) چند بلاک EPC (برای هر run)
2) در انتها: EPC Benchmark Summary
"""
from __future__ import annotations
from typing import Any, List, Tuple
import numpy as np
from epc import EPCConfig, epc_optimize, EPCResult
from utils import SimpleLogger, epc_summary_banner
# ------------------------------------------------------------
# اجرای چند Run برای یک (Objective, D)
# ------------------------------------------------------------
def run_case(
    name: str,
    D: int,
    lb: Any,
    ub: Any,
    cfg: EPCConfig,
    seeds: List[int],
) -> List[EPCResult]:
    """
    یک کیس (مثلاً Sphere با D=10) را چند بار اجرا می‌کند.
    چاپ‌های Iter و هدر داخل epc_optimize انجام می‌شود.
    """
    results: List[EPCResult] = []
    for s in seeds:
        res = epc_optimize(D=D, lb=lb, ub=ub, config=cfg, seed=s)
        results.append(res)
    return results
# ------------------------------------------------------------
# ساخت یک خط Summary مثل نمونه
# ------------------------------------------------------------
def summary_line(name: str, D: int, results: List[EPCResult]) -> str:
    best_fs = np.array([r.best_f for r in results], dtype=float)
    mean = float(best_fs.mean())
    std = float(best_fs.std(ddof=0))
    best = float(best_fs.min())
    worst = float(best_fs.max())
    total_time = float(sum(r.elapsed_sec for r in results))
    # دقیقاً شبیه نمونه: mean به صورت e با 6 رقم، std با 3 رقم، time با 3 رقم اعشار
    return (
        f"{name:<10} | D={D:4d} | runs={len(results):2d} | "
        f"mean={mean:.6e} ± {std:.3e} | best={best:.6e} | worst={worst:.6e} | time={total_time:.3f}s"
    )
# ------------------------------------------------------------
# main
# ------------------------------------------------------------
if __name__ == "__main__":
    # اگر می‌خواهی دقیقاً مثل out.txt خروجی در فایل ذخیره شود:
    # - log_path را "out.txt" بگذار
    # - توجه: append می‌کند (هر بار اجرا، ته فایل اضافه می‌شود)
    LOG_PATH = "out.txt"   # یا None برای فقط کنسول
    # تنظیمات مطابق سبک نمونه out.txt
    # فقط متود B_R استفاده می‌شود
    cfg = EPCConfig(
        N=20,
        Tmax=100,
        mu0=0.5,
        m0=0.5,
        mu_decay=0.99,
        m_decay=0.99,
        pairs_per_penguin=15,
        log_enabled=True,
        log_every=1,
        log_path=LOG_PATH,
    )
    # Logger برای Summary انتهایی (هم کنسول هم فایل)
    final_logger = SimpleLogger(LOG_PATH)
    # تعریف کیس‌ها (فقط Sphere استفاده می‌شود)
    seeds = [0, 1, 2]  # runs=3 مثل out.txt
    cases: List[Tuple[str, int, Any, Any]] = [
        ("Sphere", 10, -5.0, 5.0),
    ]
    all_results: List[Tuple[str, int, List[EPCResult]]] = []
    for name, D, lb, ub in cases:
        results = run_case(name, D, lb, ub, cfg, seeds)
        all_results.append((name, D, results))
    # چاپ Summary مثل نمونه
    final_logger("")
    final_logger(epc_summary_banner())
    for name, D, results in all_results:
        final_logger(summary_line(name, D, results))
    final_logger.close()