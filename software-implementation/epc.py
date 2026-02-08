# -*- coding: utf-8 -*-
"""
epc.py
------
پیاده‌سازی الگوریتم EPC (Emperor Penguin Colony) با روش A:
- آپدیت مارپیچی روی تمام جفت‌بعدها (p<q)
- لاگ خروجی مطابق نمونه out.txt

پیچیدگی زمانی (روش A):
    O(T * N * D^2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import time
import numpy as np

from utils import (
    as_bounds,
    clip_to_bounds,
    evaluate_population,
    SimpleLogger,
    epc_header_block,
    epc_iter_line,
)


# ------------------------------------------------------------
# کانفیگ EPC
# ------------------------------------------------------------
@dataclass
class EPCConfig:
    # اندازه جمعیت
    N: int = 20

    # تعداد تکرارهای حداکثر
    Tmax: int = 100

    # شرط توقف
    epsilon: float = 1e-12

    # پارامترهای مارپیچ
    a: float = 1.0
    b: float = 0.5

    # گرما/جذابیت
    mu0: float = 0.05
    mu_decay: float = 0.99

    # نویز/Mutation
    m0: float = 0.05
    m_decay: float = 0.99

    # جلوگیری از log(0)
    tiny: float = 1e-300

    # فعال/غیرفعال کردن لاگ
    log_enabled: bool = True

    # هر چند iteration یکبار چاپ کند (برای نمونه out.txt باید 1 باشد)
    log_every: int = 1

    # اگر فایل بدهی، علاوه بر کنسول در فایل هم ذخیره می‌شود (append)
    log_path: Optional[str] = None


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
    """
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
    آپدیت یک فرد با روش A (تمام جفت‌بعدها) طبق فرمول‌های EPC.
    """
    D = x_i.shape[0]
    x_new = x_i.copy()

    # (1) فاصله تا بهترین
    dist = float(np.linalg.norm(x_best - x_new, ord=2))

    # (2) Q = exp(-mu * dist)  => در (0,1]
    Q = float(np.exp(-mu * dist))
    # محافظ عددی (اختیاری ولی خوب)
    Q = float(np.clip(Q, tiny, 1.0))

    # (3) مارپیچ روی تمام جفت‌بعدها
    if D >= 2:
        for p in range(D - 1):
            for q in range(p + 1, D):
                theta_i = float(np.arctan2(x_new[q], x_new[p]))
                theta_b = float(np.arctan2(x_best[q], x_best[p]))

                term = (1.0 - Q) * np.exp(b * theta_b) + Q * np.exp(b * theta_i)
                if term < tiny:
                    term = tiny

                theta_k = float((1.0 / b) * np.log(term))
                r_k = float(a * np.exp(b * theta_k))

                x_new[p] = r_k * np.cos(theta_k)
                x_new[q] = r_k * np.sin(theta_k)

    # (4) نویز/Mutation
    u = rng.uniform(-1.0, 1.0, size=D)
    x_new = x_new + (m * u)

    # (5) clip
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
    اجرای EPC با لاگ خروجی شبیه نمونه out.txt.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    logger = SimpleLogger(config.log_path)

    # bounds
    LB, UB = as_bounds(lb, ub, D)

    # init population
    X = initialize_population(rng, config.N, D, LB, UB)

    # evaluate initial
    fitness = evaluate_population(obj_func, X)
    best_idx = int(np.argmin(fitness))
    best_x = X[best_idx].copy()
    best_f = float(fitness[best_idx])

    # پارامترهای پویا
    mu = float(config.mu0)
    m = float(config.m0)

    # history
    history = np.empty(config.Tmax + 1, dtype=float)
    history[0] = best_f

    # چاپ هدر مثل نمونه
    if config.log_enabled:
        logger(epc_header_block(config.N, D, config.Tmax, init_f=best_f))

    # main loop
    it = 0
    for it in range(1, config.Tmax + 1):
        # --- آپدیت جمعیت با mu,m فعلی ---
        for i in range(config.N):
            if i == best_idx:
                continue

            X[i] = update_penguin_method_A(
                rng=rng,
                x_i=X[i],
                x_best=best_x,
                mu=mu,
                m=m,
                a=config.a,
                b=config.b,
                LB=LB,
                UB=UB,
                tiny=config.tiny
            )

        # --- ارزیابی ---
        fitness = evaluate_population(obj_func, X)

        # --- آپدیت بهترین (فقط بهترین همین نسل، بدون مقایسه با گذشته) ---
        curr_best_idx = int(np.argmin(fitness))
        curr_best_f = float(fitness[curr_best_idx])
        best_idx = curr_best_idx
        best_f = curr_best_f
        best_x = X[curr_best_idx].copy()

        history[it] = best_f

        # --- چاپ لاگ Iter مثل نمونه ---
        if config.log_enabled and (it % max(1, config.log_every) == 0):
            logger(epc_iter_line(it, config.Tmax, best=best_f, mu=mu, mutation=m))

        # --- شرط توقف ---
        if best_f <= config.epsilon:
            history = history[: it + 1]
            break

        # --- decay برای iteration بعدی (برای اینکه Iter1 دقیقاً mu0 چاپ شود) ---
        mu *= config.mu_decay
        m *= config.m_decay

    elapsed = time.perf_counter() - t0

    meta = {
        "iters_done": it,
        "N": config.N,
        "D": D,
        "seed": seed,
        "mu_final": mu,
        "m_final": m,
        "method": "A (all pairs p<q)",
        "total_pairs": D * (D - 1) // 2,
    }

    logger.close()

    return EPCResult(
        best_x=best_x,
        best_f=best_f,
        history_best_f=history,
        elapsed_sec=float(elapsed),
        meta=meta
    )
