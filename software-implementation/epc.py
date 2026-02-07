# -*- coding: utf-8 -*-
"""
epc.py
------
پیاده‌سازی الگوریتم EPC (Emperor Penguin Colony).

نکته:
- این نسخه از «روش A» استفاده می‌کند: آپدیت مارپیچی روی تمام جفت‌بعدها (p<q).
- بنابراین پیچیدگی زمانی حدوداً O(T * N * D^2) است.
- این برای دیدن همه حالت‌ها/رفتارها خوب است، ولی برای D بزرگ کند می‌شود.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
import time
import numpy as np

from utils import as_bounds, clip_to_bounds, evaluate_population


# ------------------------------------------------------------
# کانفیگ EPC (پارامترهای الگوریتم)
# ------------------------------------------------------------
@dataclass
class EPCConfig:
    # اندازه جمعیت (تعداد پنگوئن‌ها)
    N: int = 30

    # تعداد تکرارهای حداکثر
    Tmax: int = 200

    # شرط توقف: اگر best_f <= epsilon شد، متوقف شو
    epsilon: float = 1e-12

    # پارامترهای مارپیچ
    a: float = 1.0
    b: float = 0.5

    # گرما/جذابیت
    mu0: float = 0.05
    mu_decay: float = 0.99

    # نویز/انحراف تصادفی
    m0: float = 0.5
    m_decay: float = 0.99

    # برای جلوگیری از log(0)
    tiny: float = 1e-300

    # چاپ وضعیت
    verbose: bool = False


# ------------------------------------------------------------
# خروجی EPC
# ------------------------------------------------------------
@dataclass
class EPCResult:
    best_x: np.ndarray
    best_f: float
    history_best_f: np.ndarray
    elapsed_sec: float
    meta: Dict[str, Any]


# ------------------------------------------------------------
# مقداردهی اولیه جمعیت
# ------------------------------------------------------------
def initialize_population(
    rng: np.random.Generator,
    N: int,
    D: int,
    LB: np.ndarray,
    UB: np.ndarray
) -> np.ndarray:
    """
    ساخت جمعیت اولیه به شکل یکنواخت داخل بازه [LB, UB].

    Parameters
    ----------
    rng : np.random.Generator
        مولد اعداد تصادفی
    N : int
        تعداد افراد
    D : int
        تعداد ابعاد
    LB, UB : np.ndarray
        bounds با شکل (D,)

    Returns
    -------
    np.ndarray
        جمعیت X با شکل (N, D)
    """
    # تولید U(0,1) و نگاشت به بازه
    return LB + rng.random((N, D)) * (UB - LB)


# ------------------------------------------------------------
# آپدیت یک پنگوئن - روش A (تمام جفت‌بعدها)
# ------------------------------------------------------------
def update_penguin_method_A(
    rng: np.random.Generator,
    x_i: np.ndarray,
    x_best: np.ndarray,
    mu: float,
    m: float,
    a: float,
    b: float,
    LB: np.ndarray,
    UB: np.ndarray,
    tiny: float
) -> np.ndarray:
    """
    آپدیت یک فرد با روش A (تمام جفت‌بعدها).

    مراحل اصلی:
    1) فاصله تا بهترین: dist = ||x_best - x_i||
    2) جذابیت: Q = exp(-mu * dist)
    3) برای هر جفت‌بعد (p,q):
         - theta_i = atan2(x[q], x[p])
         - theta_b = atan2(best[q], best[p])
         - theta_k = (1/b)*ln((1-Q)*exp(b*theta_b) + Q*exp(b*theta_i))
         - r_k = a*exp(b*theta_k)
         - x[p]=r_k*cos(theta_k), x[q]=r_k*sin(theta_k)
    4) افزودن نویز: x += m*u  (u~U(-1,1))
    5) clip به bounds

    Returns
    -------
    np.ndarray
        x جدید با شکل (D,)
    """
    D = x_i.shape[0]
    x_new = x_i.copy()

    # --- (1) فاصله تا بهترین ---
    dist = float(np.linalg.norm(x_best - x_new, ord=2))

    # --- (2) Q ---
    Q = float(np.exp(-mu * dist))

    # --- (3) مارپیچ روی تمام جفت‌بعدها ---
    if D >= 2:
        for p in range(D - 1):
            for q in range(p + 1, D):
                # زاویه‌ها روی صفحه (p,q)
                theta_i = float(np.arctan2(x_new[q], x_new[p]))
                theta_b = float(np.arctan2(x_best[q], x_best[p]))

                # term = (1-Q)*exp(b*theta_b) + Q*exp(b*theta_i)
                term = (1.0 - Q) * np.exp(b * theta_b) + Q * np.exp(b * theta_i)

                # جلوگیری از log(0)
                if term < tiny:
                    term = tiny

                theta_k = float((1.0 / b) * np.log(term))
                r_k = float(a * np.exp(b * theta_k))

                # آپدیت مختصات در همان صفحه
                x_new[p] = r_k * np.cos(theta_k)
                x_new[q] = r_k * np.sin(theta_k)

    # --- (4) نویز تصادفی ---
    u = rng.uniform(-1.0, 1.0, size=D)
    x_new = x_new + (m * u)

    # --- (5) clip ---
    x_new = clip_to_bounds(x_new, LB, UB)

    return x_new


# ------------------------------------------------------------
# تابع اصلی EPC
# ------------------------------------------------------------
def epc_optimize(
    obj_func: Callable[[np.ndarray], float],
    D: int,
    lb: Any,
    ub: Any,
    config: EPCConfig,
    seed: Optional[int] = None
) -> EPCResult:
    """
    اجرای EPC برای مینیمم کردن یک تابع هدف.

    Parameters
    ----------
    obj_func : Callable
        تابع هدف
    D : int
        تعداد ابعاد
    lb, ub : Any
        bounds (اسکالر یا بردار طول D)
    config : EPCConfig
        تنظیمات
    seed : Optional[int]
        برای تکرارپذیری

    Returns
    -------
    EPCResult
        شامل بهترین جواب، بهترین مقدار، تاریخچه، زمان اجرا و meta
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    # bounds را استاندارد می‌کنیم
    LB, UB = as_bounds(lb, ub, D)

    # --- init population ---
    X = initialize_population(rng, config.N, D, LB, UB)

    # --- evaluate initial ---
    fitness = evaluate_population(obj_func, X)
    best_idx = int(np.argmin(fitness))
    g_best_x = X[best_idx].copy()
    g_best_f = float(fitness[best_idx])

    # پارامترهای پویا
    mu = float(config.mu0)
    m = float(config.m0)

    # تاریخچه بهترین مقدار برای نمودار همگرایی
    history = np.empty(config.Tmax + 1, dtype=float)
    history[0] = g_best_f

    if config.verbose:
        print(f"[init] best_f={g_best_f:.6e}")

    # --- main loop ---
    it = 0
    for it in range(1, config.Tmax + 1):
        # کاهش تدریجی پارامترها
        mu *= config.mu_decay
        m *= config.m_decay

        # آپدیت هر فرد
        for i in range(config.N):
            X[i] = update_penguin_method_A(
                rng=rng,
                x_i=X[i],
                x_best=g_best_x,
                mu=mu,
                m=m,
                a=config.a,
                b=config.b,
                LB=LB,
                UB=UB,
                tiny=config.tiny
            )

        # ارزیابی جمعیت جدید
        fitness = evaluate_population(obj_func, X)

        # آپدیت بهترین جهانی
        curr_best_idx = int(np.argmin(fitness))
        curr_best_f = float(fitness[curr_best_idx])
        if curr_best_f < g_best_f:
            g_best_f = curr_best_f
            g_best_x = X[curr_best_idx].copy()

        history[it] = g_best_f

        # چاپ روند
        if config.verbose and (it % max(1, config.Tmax // 10) == 0):
            print(f"[iter {it:4d}] best_f={g_best_f:.6e}  mu={mu:.4e}  m={m:.4e}")

        # شرط توقف
        if g_best_f <= config.epsilon:
            history = history[: it + 1]
            break

    elapsed = time.perf_counter() - t0

    meta = {
        "iters_done": it,
        "N": config.N,
        "D": D,
        "seed": seed,
        "mu_final": mu,
        "m_final": m,
        "method": "A (all pairs p<q)",
        "complexity_note": "O(T * N * D^2)"
    }

    return EPCResult(
        best_x=g_best_x,
        best_f=g_best_f,
        history_best_f=history,
        elapsed_sec=float(elapsed),
        meta=meta
    )
