# -*- coding: utf-8 -*-
"""
utils.py
---------
ابزارهای کمکی عمومی برای پروژه:
- تبدیل bounds به شکل استاندارد
- clip به bounds
- ارزیابی جمعیت
- Logger ساده برای چاپ + نوشتن در فایل (اختیاری)
- توابع format برای چاپ لاگ شبیه نمونه out.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple
import numpy as np


# ------------------------------------------------------------
# Logger ساده (چاپ روی کنسول + اگر خواستی ذخیره در فایل)
# ------------------------------------------------------------
@dataclass
class SimpleLogger:
    """
    یک Logger خیلی ساده:
    - اگر filepath=None باشد فقط print می‌کند
    - اگر filepath داده شود، علاوه بر print، در فایل هم append می‌کند
    """
    filepath: Optional[str] = None

    def __post_init__(self) -> None:
        self._fh = None
        if self.filepath:
            self._fh = open(self.filepath, "a", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        print(msg)
        if self._fh:
            self._fh.write(msg + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


# ------------------------------------------------------------
# Formatterهای لاگ، شبیه نمونه out.txt
# ------------------------------------------------------------
def epc_banner_line(title: str = "EPC", width: int = 91) -> str:
    """
    تولید خط بنر بالا مثل:
    =========================================== EPC ===========================================
    """
    mid = f" {title} "
    side = (width - len(mid)) // 2
    return ("=" * side) + mid + ("=" * (width - side - len(mid)))


def epc_header_block(N: int, D: int, Tmax: int, init_f: float) -> str:
    """
    تولید هدر کامل مثل نمونه:
    =========================================== EPC ===========================================
    Number of Population=20 | D=10, Max Iter=100 | Total pairs=45 | Initial Fitness=53.193
    ===========================================================================================
    """
    total_pairs = D * (D - 1) // 2
    line1 = epc_banner_line("EPC", width=91)
    line2 = (
        f"Number of Population={N} | D={D}, Max Iter={Tmax} | "
        f"Total pairs={total_pairs} | Initial Fitness={init_f:.3f}"
    )
    line3 = "=" * 91
    return "\n".join([line1, line2, line3])


def epc_iter_line(it: int, Tmax: int, best: float, mu: float, mutation: float) -> str:
    """
    تولید خط هر iteration مثل:
    Iter    1/100 | best=0.0043 | mu=0.050 | mutation=0.050
    """
    return f"Iter {it:4d}/{Tmax} | best={best:.4f} | mu={mu:.3f} | mutation={mutation:.3f}"


def epc_summary_banner(width: int = 56) -> str:
    """
    تولید بنر خلاصه مثل:
    ================ EPC Benchmark Summary ================
    """
    return ("=" * 16) + " EPC Benchmark Summary " + ("=" * 16)


# ------------------------------------------------------------
# تبدیل bounds به بردارهای طول D
# ------------------------------------------------------------
def as_bounds(lb: Any, ub: Any, D: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert lb/ub into numpy arrays of shape (D,).
    """
    LB = np.array(lb, dtype=float)
    UB = np.array(ub, dtype=float)

    if LB.ndim == 0:
        LB = np.full((D,), float(LB))
    if UB.ndim == 0:
        UB = np.full((D,), float(UB))

    if LB.shape != (D,) or UB.shape != (D,):
        raise ValueError(f"Bounds must be scalar or shape (D,), got LB={LB.shape}, UB={UB.shape}")

    if np.any(UB <= LB):
        raise ValueError("All elements must satisfy UB > LB")

    return LB, UB


# ------------------------------------------------------------
# Clip به bounds
# ------------------------------------------------------------
def clip_to_bounds(X: np.ndarray, LB: np.ndarray, UB: np.ndarray) -> np.ndarray:
    """
    Clip X into [LB, UB] elementwise.
    """
    return np.minimum(np.maximum(X, LB), UB)


# ------------------------------------------------------------
# ارزیابی fitness برای کل جمعیت
# ------------------------------------------------------------
def evaluate_population(obj_func: Callable[[np.ndarray], float], X: np.ndarray) -> np.ndarray:
    """
    Evaluate objective function for each individual in population.
    """
    return np.array([obj_func(x) for x in X], dtype=float)
