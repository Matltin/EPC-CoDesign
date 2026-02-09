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
from typing import Callable, Any, Dict, List, Tuple
import numpy as np
from epc import EPCConfig, epc_optimize, EPCResult
from objectives import sphere, rosenbrock
from utils import SimpleLogger, epc_summary_banner
# ------------------------------------------------------------
# اجرای چند Run برای یک (Objective, D)
# ------------------------------------------------------------
def run_case(
    name: str,
    obj_func: Callable[[np.ndarray], float],
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
        res = epc_optimize(obj_func=obj_func, D=D, lb=lb, ub=ub, config=cfg, seed=s)
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
    # برای انتخاب متود، update_method را به یکی از موارد زیر تغییر دهید:
    # "A" - مارپیچ روی همه جفت‌بعدها + شعاع ثابت
    # "A_R" - مارپیچ روی همه جفت‌بعدها + شعاع آپدیت می‌شود
    # "B" - مارپیچ روی K جفت تصادفی + شعاع ثابت
    # "B_R" - مارپیچ روی K جفت تصادفی + شعاع آپدیت می‌شود
    cfg = EPCConfig(
        N=20,
        Tmax=100,
        mu0=0.5,
        m0=0.5,
        mu_decay=0.99,
        m_decay=0.99,
        update_method="B_R",  # <-- اینجا متود مورد نظر را انتخاب کنید
        pairs_per_penguin=6,
        log_enabled=True,
        log_every=1,
        log_path=LOG_PATH,
    )
    # Logger برای Summary انتهایی (هم کنسول هم فایل)
    final_logger = SimpleLogger(LOG_PATH)
    # تعریف کیس‌ها (مثل نمونه می‌تونی D=10 و D=100 بزنی)
    seeds = [0, 1, 2]  # runs=3 مثل out.txt
    cases: List[Tuple[str, Callable[[np.ndarray], float], int, Any, Any]] = [
        ("Sphere", sphere, 10, -5.0, 5.0),
        # ("Sphere", sphere, 100, -5.0, 5.0),
        # ("Rosenbrock", rosenbrock, 10, -2.0, 2.0),
        # ("Rosenbrock", rosenbrock, 100, -2.0, 2.0),
    ]
    all_results: List[Tuple[str, int, List[EPCResult]]] = []
    for name, func, D, lb, ub in cases:
        results = run_case(name, func, D, lb, ub, cfg, seeds)
        all_results.append((name, D, results))
    # چاپ Summary مثل نمونه
    final_logger("")
    final_logger(epc_summary_banner())
    for name, D, results in all_results:
        final_logger(summary_line(name, D, results))
    final_logger.close()
