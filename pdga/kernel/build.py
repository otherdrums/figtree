"""Build script: compile pdga/kernel/boundary_project.cu → .so module.

Usage:
    python build.py

Produces:
    build/ boundary_project.so

The .so is then imported by pdga.kernel.boundary_project at runtime.
"""

from pathlib import Path
from torch.utils.cpp_extension import load

KERNEL_DIR = Path(__file__).parent
CU_FILE = KERNEL_DIR / "boundary_project.cu"

# Compile (or load cached) CUDA extension
_boundary_project_ext = None


def get_extension():
    """Compile/load the boundary_project CUDA extension."""
    global _boundary_project_ext
    if _boundary_project_ext is not None:
        return _boundary_project_ext

    _boundary_project_ext = load(
        name="boundary_project",
        sources=[str(CU_FILE)],
        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
        verbose=True,
    )
    return _boundary_project_ext


if __name__ == "__main__":
    ext = get_extension()
    print(f"Compiled: {ext}")
