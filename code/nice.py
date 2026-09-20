"""Run the current process at BELOW_NORMAL priority so the desktop stays usable.

Imported for its side effect by every long-running entry point.
"""
import os
import sys

# bound the BLAS/OpenMP thread pools before torch/numpy are imported
_THREADS = int(os.environ.get("NTHREADS", "4"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, str(_THREADS))

if sys.platform == "win32":
    try:
        import ctypes

        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
else:
    try:
        os.nice(10)
    except Exception:
        pass


def threads():
    return _THREADS
