"""PDGA v2 CUDA kernels.

boundary_project: project fact boundaries through W_k/W_v weight matrices.
"""

from pdga.kernel.boundary_project import project_boundaries_to_kv

__all__ = ["project_boundaries_to_kv"]
