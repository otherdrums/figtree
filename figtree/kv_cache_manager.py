"""KV cache manager: external, on-demand K/V materialization for figments.

Figment K/V caches are large, variable-shape tensors
``(num_layers, seq_len, 2, kv_dim)``. Rather than store them in the LanceDB
row, this manager keeps them as external blobs (local files or object storage)
addressed by ``kv_uri`` on the figment's meta.

Modes:
- **lazy** (default): K/V is not persisted at ingest. ``materialize`` recomputes
  it from text on demand, caches it in an in-memory LRU, and (optionally) writes
  a quantized blob to ``kv_uri`` for future reuse.
- **eager**: K/V is computed at ingest and written (quantized) to ``kv_uri``.

Tiers: LRU (hot) -> external blob at ``kv_uri`` (warm) -> recompute (cold).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from figtree.figment import Figment


def _fs_for_uri(uri: str):
    """Return a pyarrow FileSystem + path for ``uri`` (local or s3://)."""
    from pyarrow import fs

    if uri.startswith(("s3://", "gs://", "az://", "hf://")):
        filesystem, path = fs.FileSystem.from_uri(uri)
        return filesystem, path
    # Local path.
    p = Path(uri)
    return fs.LocalFileSystem(), str(p)


class KVCacheManager:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        kv_root: str = "./figtree_kv",
        mode: str = "lazy",
        quantize: str | None = "fp16",
        lru_capacity: int = 64,
        use_kernel: bool = True,
    ):
        """Create a KV cache manager.

        Args:
            kv_root: directory or object-store URI prefix for persisted blobs.
            mode: ``"lazy"`` (recompute on demand) or ``"eager"`` (persist at
                ingest).
            quantize: ``"fp16"`` / ``"int8"`` / ``None`` when writing blobs.
            lru_capacity: number of figment K/V blobs held in memory.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.kv_root = kv_root
        self.mode = mode
        self.quantize = quantize
        self.use_kernel = use_kernel
        self.num_layers = len(model.model.layers)
        config = model.config
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self.kv_dim = self.num_kv_heads * self.head_dim
        self._lru: dict[str, np.ndarray] = {}
        self._lru_order: list[str] = []
        self._lru_capacity = lru_capacity

    # -- blob path ------------------------------------------------------- #
    def _blob_uri(self, figment_id: str) -> str:
        if self.kv_root.startswith(("s3://", "gs://", "az://")):
            return f"{self.kv_root.rstrip('/')}/{figment_id}.kv.npy"
        Path(self.kv_root).mkdir(parents=True, exist_ok=True)
        return str(Path(self.kv_root) / f"{figment_id}.kv.npy")

    # -- serialization --------------------------------------------------- #
    def _quantize(self, arr: np.ndarray) -> tuple[np.ndarray, dict[str, Any] | None]:
        if self.quantize == "fp16":
            return arr.astype(np.float16), {"quantize": "fp16"}
        if self.quantize == "int8":
            # Symmetric int8 quantization around zero; store scale per tensor.
            scale = float(np.max(np.abs(arr))) or 1.0
            q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
            return q, {"quantize": "int8", "scale": scale}
        return arr.astype(np.float32), None

    @staticmethod
    def _dequantize(arr: np.ndarray, meta: dict[str, Any] | None) -> np.ndarray:
        if meta is None:
            return arr.astype(np.float32)
        if meta.get("quantize") == "fp16":
            return arr.astype(np.float32)
        if meta.get("quantize") == "int8":
            scale = float(meta["scale"])
            return arr.astype(np.float32) * scale
        return arr.astype(np.float32)

    def _write_blob(self, uri: str, kv: np.ndarray, qmeta: dict[str, Any] | None) -> None:
        filesystem, path = _fs_for_uri(uri)
        payload = {
            "kv": kv,
            "meta": qmeta,
            "num_layers": self.num_layers,
            "kv_dim": self.kv_dim,
        }
        buf = _to_bytes(payload)
        parent = path.rsplit("/", 1)[0] if "/" in path else "."
        if parent:
            filesystem.create_dir(parent)
        with filesystem.open_output_stream(path) as stream:
            stream.write(buf)

    def _read_blob(self, uri: str) -> tuple[np.ndarray, dict[str, Any] | None]:
        filesystem, path = _fs_for_uri(uri)
        with filesystem.open_input_stream(path) as stream:
            data = stream.read()
        payload = _from_bytes(data)
        return payload["kv"], payload.get("meta")

    # -- LRU ------------------------------------------------------------- #
    def _lru_put(self, fid: str, kv: np.ndarray) -> None:
        if fid in self._lru:
            self._lru_order.remove(fid)
        self._lru[fid] = kv
        self._lru_order.append(fid)
        while len(self._lru_order) > self._lru_capacity:
            old = self._lru_order.pop(0)
            self._lru.pop(old, None)

    def _lru_get(self, fid: str) -> np.ndarray | None:
        if fid in self._lru:
            self._lru_order.remove(fid)
            self._lru_order.append(fid)
            return self._lru[fid]
        return None

    # -- KV computation -------------------------------------------------- #
    def _compute_kv(self, figments: list[Figment]) -> list[np.ndarray]:
        """Recompute per-figment K/V via a single concatenated forward pass.

        Mirrors ``ingest.py``: figment texts joined with ``\\n\\n`` separators so
        the cached slices reproduce what generation would produce.
        """
        device = self.model.device
        sep_ids = self.tokenizer.encode("\n\n", add_special_tokens=False)
        stream: list[int] = []
        starts: list[int] = []
        kept: list[Figment] = []
        for fig in figments:
            ids = self.tokenizer.encode(fig.text, add_special_tokens=False)
            if not ids:
                continue
            if stream:
                stream.extend(sep_ids)
            starts.append(len(stream))
            stream.extend(ids)
            kept.append(fig)
        if not stream:
            return []

        all_ids = torch.tensor([stream], dtype=torch.long, device=device)
        seq_len_total = all_ids.shape[1]

        layer_outputs: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(idx):
            def hook(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                layer_outputs[idx] = o.detach()
            return hook

        for li in range(self.num_layers):
            handles.append(self.model.model.layers[li].register_forward_hook(make_hook(li)))
        try:
            with torch.no_grad():
                emb_out = self.model.get_input_embeddings()(all_ids)
                self.model(all_ids)
                layer_inputs = [emb_out[0]] + [layer_outputs[li - 1][0] for li in range(1, self.num_layers)]
                from figtree.ingest import _project_kv
                full_kv: list[torch.Tensor] = []
                for li in range(self.num_layers):
                    k, v = _project_kv(layer_inputs[li], self.model.model.layers[li],
                                       self.num_kv_heads, self.head_dim, self.use_kernel)
                    k = k.reshape(seq_len_total, self.kv_dim)
                    v = v.reshape(seq_len_total, self.kv_dim)
                    full_kv.append(torch.stack([k, v], dim=1).float().cpu())
                results: list[np.ndarray] = []
                for i, (fig, start) in enumerate(zip(kept, starts)):
                    end = starts[i + 1] if i + 1 < len(starts) else len(stream)
                    kv_list = [full_kv[li][start:end].numpy() for li in range(self.num_layers)]
                    results.append(np.stack(kv_list))
                return results
        finally:
            for h in handles:
                h.remove()

    # -- public API ------------------------------------------------------ #
    def persist(self, figments: list[Figment]) -> list[Figment]:
        """Eager path: compute + write K/V blobs, patch ``kv_uri`` onto figs."""
        if self.mode != "eager":
            raise RuntimeError("persist() requires mode='eager'")
        kvs = self._compute_kv(figments)
        out = []
        for fig, kv in zip(figments, kvs):
            uri = self._blob_uri(fig.figment_id)
            qkv, qmeta = self._quantize(kv)
            self._write_blob(uri, qkv, qmeta)
            self._lru_put(fig.figment_id, kv)
            fig.meta["kv_uri"] = uri
            fig.meta["has_kv_cache"] = True
            out.append(fig)
        return out

    def materialize(self, figments: list[Figment]) -> dict[str, np.ndarray]:
        """Return {figment_id: kv_array} for generation, using tiers + recompute."""
        result: dict[str, np.ndarray] = {}
        need: list[Figment] = []
        for fig in figments:
            cached = self._lru_get(fig.figment_id)
            if cached is not None:
                result[fig.figment_id] = cached
                continue
            uri = fig.meta.get("kv_uri")
            if uri and fig.meta.get("has_kv_cache"):
                try:
                    qkv, qmeta = self._read_blob(uri)
                    kv = self._dequantize(qkv, qmeta)
                    self._lru_put(fig.figment_id, kv)
                    result[fig.figment_id] = kv
                    continue
                except Exception:
                    pass
            need.append(fig)
        if need:
            computed = self._compute_kv(need)
            for fig, kv in zip(need, computed):
                self._lru_put(fig.figment_id, kv)
                result[fig.figment_id] = kv
                # If eager and not yet persisted, write it now for reuse.
                if self.mode == "eager" and not fig.meta.get("kv_uri"):
                    uri = self._blob_uri(fig.figment_id)
                    qkv, qmeta = self._quantize(kv)
                    self._write_blob(uri, qkv, qmeta)
                    fig.meta["kv_uri"] = uri
                    fig.meta["has_kv_cache"] = True
        return result


def _to_bytes(payload: dict) -> bytes:
    import io

    buf = io.BytesIO()
    np.save(buf, payload["kv"], allow_pickle=False)
    kv_bytes = buf.getvalue()
    meta = payload.get("meta")
    header = json.dumps({
        "num_layers": payload["num_layers"],
        "kv_dim": payload["kv_dim"],
        "meta": meta,
        "kv_len": len(kv_bytes),
    }).encode()
    return len(header).to_bytes(4, "little") + header + kv_bytes


def _from_bytes(data: bytes) -> dict:
    import io

    hlen = int.from_bytes(data[:4], "little")
    header = json.loads(data[4:4 + hlen].decode())
    kv_bytes = data[4 + hlen: 4 + hlen + header["kv_len"]]
    kv = np.load(io.BytesIO(kv_bytes), allow_pickle=False)
    return {"kv": kv, "meta": header.get("meta"),
            "num_layers": header["num_layers"], "kv_dim": header["kv_dim"]}
