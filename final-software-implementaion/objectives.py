# -*- coding: utf-8 -*-
"""
objectives.py
----------------
این فایل شامل توابع هدف (Benchmark Functions) است.
هدف:
- جدا کردن "تعریف مسئله" از "الگوریتم" تا:
  1) خوانایی بالا برود
  2) اضافه کردن تابع هدف جدید آسان شود
  3) برای همطراحی HW/SW بعداً مشخص باشد "ارزیابی تابع هدف" کجاست
"""
from __future__ import annotations
import numpy as np
# ------------------------------------------------------------
# Sphere Function
# ------------------------------------------------------------
# توضیح:
# - یک تابع بسیار ساده و استاندارد برای تست الگوریتم‌های بهینه‌سازی
# - کمینه: f=0 در x=0
def sphere(x: np.ndarray) -> float:
    """
    Sphere benchmark function:
        f(x) = sum(x_i^2)
    Parameters
    ----------
    x : np.ndarray
        بردار جواب با شکل (D,)
    Returns
    -------
    float
        مقدار تابع هدف
    """
    return float(np.sum(x * x))
# ------------------------------------------------------------
# Rosenbrock Function
# ------------------------------------------------------------
# توضیح:
# - یکی از معروف‌ترین توابع تست
# - "دره باریک" دارد و برای الگوریتم‌ها چالشی است
# - کمینه: f=0 در x=[1,1,...,1]
def rosenbrock(x: np.ndarray) -> float:
    """
    Rosenbrock benchmark function:
        f(x) = sum_{i=1..D-1} [100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]
    Parameters
    ----------
    x : np.ndarray
        بردار جواب با شکل (D,)
    Returns
    -------
    float
        مقدار تابع هدف
    """
    xi = x[:-1]
    xnext = x[1:]
    return float(np.sum(100.0 * (xnext - xi * xi) ** 2 + (xi - 1.0) ** 2))
