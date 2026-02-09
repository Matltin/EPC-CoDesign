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