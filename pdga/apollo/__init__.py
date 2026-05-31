"""Apollo engine — LARQL-compatible residual-injection generation.

Provides ApolloModel, a high-level generator that uses KV caching
and fused CUDA kernels for boundary swap + token injection.
"""

from pdga.apollo.engine import ApolloModel, apollo_generate

__all__ = ["ApolloModel", "apollo_generate"]
