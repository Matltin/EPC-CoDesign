# -*- coding: utf-8 -*-
"""
utils.py
---------
ابزارهای کمکی عمومی برای پروژه.

این فایل کارهای تکراری/زیرساختی را نگه می‌دارد:
- تبدیل bounds به شکل استاندارد (بردار طول D)
- clip کردن به bounds
- ارزیابی جمعیت (fitness)
- ابزارهای کوچک کمکی

این جداسازی باعث می‌شود epc.py روی منطق الگوریتم تمرکز کند.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple
import numpy as np


# ------------------------------------------------------------
# تبدیل bounds به بردارهای طول D
# ------------------------------------------------------------
# چرا لازم است؟
# چون گاهی bounds را اسکالر می‌دهیم (مثلاً -5 تا 5)
# و گاهی به‌صورت بردار (برای هر بعد جدا)
def as_bounds(lb: Any, ub: Any, D: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert lb/ub into numpy arrays of shape (D,).

    Parameters
    ----------
    lb, ub : Any
        می‌تواند float باشد یا لیست/آرایه طول D
    D : int
        تعداد ابعاد

    Returns
    -------
    (LB, UB) : Tuple[np.ndarray, np.ndarray]
        کران پایین و کران بالا، هر دو با شکل (D,)
    """
    LB = np.array(lb, dtype=float)
    UB = np.array(ub, dtype=float)

    # اگر اسکالر بود، آن را به بردار طول D تبدیل می‌کنیم
    if LB.ndim == 0:
        LB = np.full((D,), float(LB))
    if UB.ndim == 0:
        UB = np.full((D,), float(UB))

    # بررسی سازگاری شکل
    if LB.shape != (D,) or UB.shape != (D,):
        raise ValueError(f"Bounds must be scalar or shape (D,), got LB={LB.shape}, UB={UB.shape}")

    # بررسی اینکه UB > LB باشد
    if np.any(UB <= LB):
        raise ValueError("All elements must satisfy UB > LB")

    return LB, UB


# ------------------------------------------------------------
# Clip به bounds
# ------------------------------------------------------------
# بعد از هر آپدیت، ممکن است جواب از بازه خارج شود.
# طبق متن پروژه باید به کران‌ها محدود شود.
def clip_to_bounds(X: np.ndarray, LB: np.ndarray, UB: np.ndarray) -> np.ndarray:
    """
    Clip X into [LB, UB] elementwise.

    Parameters
    ----------
    X : np.ndarray
        می‌تواند (D,) یا (N,D) باشد
    LB, UB : np.ndarray
        کران‌ها (D,)

    Returns
    -------
    np.ndarray
        X پس از clip شدن
    """
    return np.minimum(np.maximum(X, LB), UB)


# ------------------------------------------------------------
# ارزیابی fitness برای کل جمعیت
# ------------------------------------------------------------
# این بخش کاندید گلوگاه است (به‌خصوص اگر obj سنگین باشد)
def evaluate_population(obj_func: Callable[[np.ndarray], float], X: np.ndarray) -> np.ndarray:
    """
    Evaluate objective function for each individual in population.

    Parameters
    ----------
    obj_func : Callable[[np.ndarray], float]
        تابع هدف (برای یک بردار)
    X : np.ndarray
        جمعیت با شکل (N,D)

    Returns
    -------
    np.ndarray
        fitness با شکل (N,)
    """
    # ساده‌ترین روش: حلقه پایتونی
    # (بعداً برای سرعت می‌شود vectorize/numba/cython کرد)
    return np.array([obj_func(x) for x in X], dtype=float)
