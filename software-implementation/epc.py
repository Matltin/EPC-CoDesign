# -*- coding: utf-8 -*-
"""
epc.py
------
پیاده‌سازی الگوریتم EPC (Emperor Penguin Colony) با متود B_R:

Method B_R:
- آپدیت مارپیچی فقط روی K جفت‌بعد تصادفی در هر پنگوئن
- شعاع آپدیت می‌شود (بر اساس فاصله و Q)
- پیچیدگی تقریبی: O(T * N * K)

این نسخه با ساختار لاگ و منطق کدی که خودت فرستادی سازگار است:
- بهترین فرد (best_idx) در هر iteration آپدیت نمی‌شود (elitism)
- best هر iteration از روی جمعیت فعلی محاسبه و ثبت می‌شود
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple
import time
import numpy as np

from utils import (
    as_bounds,
    clip_to_bounds,
    SimpleLogger,
    epc_header_block,
    epc_iter_line,
)

from objectives import sphere


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

    # جلوگیری از log(0) / خطای عددی
    tiny: float = 1e-300

    # ----------------------------
    # تنظیمات متود B_R
    # ----------------------------
    # تعداد جفت‌بعدهایی که در هر iteration برای هر پنگوئن آپدیت می‌کنیم
    pairs_per_penguin: int = 6

    # ----------------------------
    # تنظیمات لاگ
    # ----------------------------
    log_enabled: bool = True
    log_every: int = 1
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


# ============================================================
# بخش‌های کمکی برای Method B_R
# ============================================================

def _sample_one_pair(rng: np.random.Generator, D: int) -> Tuple[int, int]:
    """
    نمونه‌برداری تصادفی از یک جفت‌بعد (p,q) با شرط p<q.

    روش:
    - p را از [0..D-1] انتخاب می‌کنیم
    - q را از [0..D-2] انتخاب می‌کنیم و اگر q >= p بود، q را یک واحد شیفت می‌دهیم
      تا q != p تضمین شود.
    - سپس (p,q) را مرتب می‌کنیم تا p<q شود.
    """
    p = int(rng.integers(0, D))
    q = int(rng.integers(0, D - 1))
    if q >= p:
        q += 1
    # حالا p و q قطعاً متفاوت‌اند
    if p < q:
        return p, q
    return q, p


def _sample_k_unique_pairs(rng: np.random.Generator, D: int, k: int) -> List[Tuple[int, int]]:
    """
    تولید k جفت‌بعد *منحصر‌به‌فرد* برای Method B_R.

    نکته:
    - اگر k >= تعداد کل جفت‌ها، عملاً برابر Method A می‌شود.
    """
    if D < 2 or k <= 0:
        return []

    total_pairs = D * (D - 1) // 2
    if k >= total_pairs:
        # همه‌ی جفت‌ها
        pairs: List[Tuple[int, int]] = []
        for p in range(D - 1):
            for q in range(p + 1, D):
                pairs.append((p, q))
        return pairs

    pairs_set: set[Tuple[int, int]] = set()
    while len(pairs_set) < k:
        pairs_set.add(_sample_one_pair(rng, D))

    return list(pairs_set)


def _spiral_update_on_pair_with_radius(
    x_new: np.ndarray,
    x_best: np.ndarray,
    p: int,
    q: int,
    Q: float,
    a: float,
    b: float,
    tiny: float
) -> None:
    """
    انجام آپدیت مارپیچی روی صفحه‌ی (p,q) با آپدیت شعاع.

    فرمول‌ها (مطابق EPC با تغییر شعاع):
    theta_i = atan2(x[q], x[p])
    theta_b = atan2(best[q], best[p])
    
    r_i = sqrt(x[p]^2 + x[q]^2)  (شعاع فعلی)
    r_b = sqrt(best[p]^2 + best[q]^2)  (شعاع بهترین)

    theta_k = (1/b) * ln( (1-Q)*exp(b*theta_b) + Q*exp(b*theta_i) )
    r_k     = (1-Q) * r_b + Q * r_i  (میانگین وزن‌دار شعاع‌ها)

    x[p] = r_k * cos(theta_k)
    x[q] = r_k * sin(theta_k)
    """
    theta_i = float(np.arctan2(x_new[q], x_new[p]))
    theta_b = float(np.arctan2(x_best[q], x_best[p]))

    # محاسبه شعاع فعلی و شعاع بهترین
    r_i = float(np.sqrt(x_new[p]**2 + x_new[q]**2))
    r_b = float(np.sqrt(x_best[p]**2 + x_best[q]**2))

    term = (1.0 - Q) * np.exp(b * theta_b) + Q * np.exp(b * theta_i)
    if term < tiny:
        term = tiny

    theta_k = float((1.0 / b) * np.log(term))
    
    # آپدیت شعاع: میانگین وزن‌دار بر اساس Q
    r_k = float((1.0 - Q) * r_b + Q * r_i)

    x_new[p] = r_k * np.cos(theta_k)
    x_new[q] = r_k * np.sin(theta_k)


# ============================================================
# Method B_R - شعاع آپدیت می‌شود
# ============================================================

def update_penguin_method_B_R(
    rng: np.random.Generator,
    x_i: np.ndarray,
    x_best: np.ndarray,
    mu: float,
    m: float,
    a: float,
    b: float,
    LB: np.ndarray,
    UB: np.ndarray,
    tiny: float,
    k_pairs: int
) -> np.ndarray:
    """
    آپدیت یک فرد با روش B_R:
    - محاسبه Q
    - انتخاب K جفت‌بعد تصادفی (منحصر به فرد)
    - مارپیچ فقط روی همان K جفت (شعاع آپدیت می‌شود)
    - نویز
    - clip

    مزیت:
    - برای D بزرگ، خیلی سریع‌تر از روش A_R است.
    """
    D = x_i.shape[0]
    x_new = x_i.copy()

    # (1) فاصله تا بهترین
    dist = float(np.linalg.norm(x_best - x_new, ord=2))

    # (2) Q
    Q = float(np.exp(-mu * dist))
    Q = float(np.clip(Q, tiny, 1.0))

    # (3) انتخاب K جفت‌بعد
    pairs = _sample_k_unique_pairs(rng, D, k_pairs)

    # (4) مارپیچ فقط روی همان جفت‌ها با آپدیت شعاع
    for (p, q) in pairs:
        _spiral_update_on_pair_with_radius(x_new, x_best, p, q, Q, a, b, tiny)

    # (5) نویز/Mutation
    u = rng.uniform(-1.0, 1.0, size=D)
    x_new = x_new + (m * u)

    # (6) clip
    x_new = clip_to_bounds(x_new, LB, UB)
    return x_new


# ------------------------------------------------------------
# تابع اصلی EPC
# ------------------------------------------------------------
def epc_optimize(
    D: int,
    lb: Any,
    ub: Any,
    config: EPCConfig,
    seed: Optional[int] = None
) -> EPCResult:
    """
    اجرای EPC با لاگ خروجی شبیه نمونه out.txt.

    نکته‌ی مهم (طبق کدی که خودت فرستادی):
    - بهترین فرد (best_idx) در هر iteration آپدیت نمی‌شود (elitism)
    - سپس دوباره fitness کل جمعیت محاسبه می‌شود و best_idx جدید انتخاب می‌شود.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    logger = SimpleLogger(config.log_path)

    # bounds
    LB, UB = as_bounds(lb, ub, D)

    # init population
    X = initialize_population(rng, config.N, D, LB, UB)

    # evaluate initial
    fitness = np.array([sphere(x) for x in X], dtype=float)
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
        # برای اینکه معلوم باشد روش کدام است
        logger(f"Update Method: B_R | pairs_per_penguin(k)={config.pairs_per_penguin}")

    # main loop
    it = 0

    for it in range(1, config.Tmax + 1):
        # --- آپدیت جمعیت با mu,m فعلی ---
        for i in range(config.N):
            # elitism: بهترین فرد این iteration را دست نمی‌زنیم
            if i == best_idx:
                continue

            X[i] = update_penguin_method_B_R(
                rng=rng,
                x_i=X[i],
                x_best=best_x,
                mu=mu,
                m=m,
                a=config.a,
                b=config.b,
                LB=LB,
                UB=UB,
                tiny=config.tiny,
                k_pairs=config.pairs_per_penguin
            )

        # --- ارزیابی ---
        fitness = np.array([sphere(x) for x in X], dtype=float)

        # --- آپدیت بهترین این نسل ---
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

        # --- decay برای iteration بعدی ---
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
        "method": "B_R",
        "pairs_per_penguin": config.pairs_per_penguin,
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