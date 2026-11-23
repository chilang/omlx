# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 routed-expert kernels: grouped rows, the one-token expert path and
their fallbacks keep the stock MLX bits."""

from __future__ import annotations

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

if HAS_MLX:
    from tests.test_inkling_vlm_mtp import _inkling2_tiny

_AFFINE4 = {"mode": "affine", "bits": 4, "group_size": 64}


def _gpu_only():
    if mx.default_device() != mx.gpu:
        pytest.skip("the expert kernels are GPU-only")


def _routing(rows, top, experts, seed):
    """Distinct experts per row, as top-k routing picks them."""
    scores = mx.random.uniform(shape=(rows, experts), key=mx.random.key(seed))
    return mx.argpartition(-scores, kth=top - 1, axis=-1)[:, :top].astype(mx.uint32)


def _carrier(mode, experts, n, k, seed):
    w = mx.random.normal((experts, n, k), key=mx.random.key(seed)) * 0.05
    if mode == "nvfp4":
        q, s = mx.quantize(w, mode="nvfp4")
        return q, s, None
    q, s, b = mx.quantize(w, group_size=64, bits=4)
    return q, s.astype(mx.bfloat16), b.astype(mx.bfloat16)


def _stock(x, carrier, inds, mode):
    q, s, b = carrier
    rows, top = inds.shape
    y = mx.gather_qmm(
        mx.expand_dims(x, (-2, -3)),
        q,
        s,
        b,
        rhs_indices=inds,
        transpose=True,
        group_size=16 if mode == "nvfp4" else 64,
        bits=4,
        mode=mode,
    )
    return y.reshape(rows * top, -1)


@pytest.mark.parametrize("mode", ["nvfp4", "affine"])
# K = 512 takes MLX's qmv_fast, the others its qmv; 1152 and 2304 are
# Inkling 2 Mini's expert widths.
@pytest.mark.parametrize("k,n", [(128, 64), (512, 128), (1152, 256), (2304, 128)])
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8])
def test_grouped_expert_rows_match_gather_qmm(mode, k, n, rows):
    """Each pair's row equals the stock per-pair ``gather_qmm`` at M = 1."""
    _gpu_only()
    from omlx.patches.inkling2 import kernels

    experts, top = 8, 4
    carrier = _carrier(mode, experts, n, k, seed=rows)
    x = (mx.random.normal((rows, k), key=mx.random.key(7)) * 0.5).astype(mx.bfloat16)
    inds = _routing(rows, top, experts, seed=rows + 1)
    flat = inds.reshape(-1)
    order = mx.argsort(flat).astype(mx.uint32)
    if mode == "nvfp4":
        got = kernels.nvfp4_gather_rows(
            x, carrier[0], carrier[1], order, flat[order], top, rows
        )
    else:
        got = kernels.affine4_gather_rows(
            x, *carrier, 64, order, flat[order], top, rows
        )
    assert mx.array_equal(got, _stock(x, carrier, inds, mode))


@pytest.mark.parametrize("mode", ["nvfp4", "affine"])
def test_grouped_expert_rows_cover_long_runs_of_one_expert(mode):
    """More pairs on one expert than ``max_rows`` are taken in chunks; none is
    left unwritten."""
    _gpu_only()
    from omlx.patches.inkling2 import kernels

    rows, top, k, n = 3, 4, 256, 64
    carrier = _carrier(mode, 2, n, k, seed=3)
    x = (mx.random.normal((rows, k), key=mx.random.key(4)) * 0.5).astype(mx.bfloat16)
    inds = mx.array([[0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]], dtype=mx.uint32)
    flat = inds.reshape(-1)
    order = mx.argsort(flat).astype(mx.uint32)
    gather = (
        (lambda: kernels.nvfp4_gather_rows(x, carrier[0], carrier[1], order, flat[order], top, rows))
        if mode == "nvfp4"
        else (lambda: kernels.affine4_gather_rows(x, *carrier, 64, order, flat[order], top, rows))
    )
    assert mx.array_equal(gather(), _stock(x, carrier, inds, mode))


def test_routing_kernel_orders_the_pairs_by_expert():
    _gpu_only()
    from omlx.patches.inkling2 import kernels

    logits = mx.random.normal((5, 64), key=mx.random.key(1)) * 2
    bias = mx.random.normal((64,), key=mx.random.key(2)) * 0.05
    inds, w = kernels.route_topk(logits, bias, 4, 8.0)
    inds2, w2, order, sorted_e = kernels.route_topk_sorted_rows(logits, bias, 4, 8.0)
    assert mx.array_equal(inds, inds2) and mx.array_equal(w, w2)
    flat = inds.reshape(-1)
    assert mx.array_equal(flat[order], sorted_e)
    assert mx.array_equal(mx.sort(sorted_e), sorted_e)
    assert mx.array_equal(mx.sort(order), mx.arange(flat.size, dtype=mx.uint32))


def _bf16_run(model, ids, prefill):
    """Eager prefill, then compiled one-token steps, in BF16 activations."""
    from mlx.utils import tree_flatten

    keep = model.cast_predicate
    cast = [
        (k, v.astype(mx.bfloat16))
        for k, v in tree_flatten(model.parameters())
        if v.dtype == mx.float32 and keep(k)
    ]
    model.load_weights(cast, strict=False)
    cache = model.make_cache()
    outs = [model(ids[:, :prefill], cache=cache)]
    outs += [model(ids[:, t : t + 1], cache=cache) for t in range(prefill, ids.shape[1])]
    mx.eval(outs)
    return outs


def _spy(monkeypatch, names):
    from omlx.patches.inkling2 import kernels

    used = dict.fromkeys(names, 0)
    for name in names:
        original = getattr(kernels, name)

        def spy(*args, _original=original, _name=name, **kwargs):
            out = _original(*args, **kwargs)
            used[_name] += out is not None
            return out

        monkeypatch.setattr(kernels, name, spy)
    return used


def _stock_experts(monkeypatch):
    from omlx.patches.inkling2 import model as inkling2

    monkeypatch.setattr(inkling2.Inkling2MoE, "decode_conv", lambda *a, **k: None)
    monkeypatch.setattr(inkling2.Inkling2Experts, "_grouped", lambda *a, **k: None)


_LOOPED_NVFP4 = dict(
    formats=("nvfp4",) * 18,
    loop_config=dict(core_start=6, core_end=12, num_passes=3, gate_rank=8),
)


@pytest.mark.parametrize(
    "dims",
    [dict(formats=("nvfp4",) * 6 + (_AFFINE4,) * 6), _LOOPED_NVFP4],
    ids=["nvfp4+affine", "looped"],
)
def test_expert_kernels_keep_the_stock_bits(dims, monkeypatch):
    """Prefill, then one-token steps: the NVFP4 layers take the two-kernel
    expert path, the affine layers the grouped kernel, and every logit equals
    the stock gather_qmm path's."""
    _gpu_only()
    ids = mx.random.randint(0, 300, (1, 16), key=mx.random.key(5))
    used = _spy(monkeypatch, ("nvfp4_expert_decode", "affine4_gather_rows"))
    fused = _bf16_run(_inkling2_tiny(**dims), ids, 12)
    assert used["nvfp4_expert_decode"] > 0
    assert (used["affine4_gather_rows"] > 0) == (_AFFINE4 in dims["formats"])
    _stock_experts(monkeypatch)
    # A fresh model: compiled layer stages keep the graph of their first trace.
    reference = _bf16_run(_inkling2_tiny(**dims), ids, 12)
    for got, want in zip(fused, reference):
        assert mx.array_equal(got, want)


@pytest.mark.parametrize("width", [2, 3, 4, 8])
@pytest.mark.parametrize(
    "formats",
    [("nvfp4",) * 12, ("nvfp4",) * 6 + (_AFFINE4,) * 6],
    ids=["nvfp4", "nvfp4+affine"],
)
def test_grouped_verify_rows_match_serial_decode(formats, width, monkeypatch):
    """Row-exact verify blocks route their rows through the grouped experts
    (each expert read once) and every row keeps its serial decode bits."""
    import mlx.nn as nn

    from omlx.patches.qwen35_verify_qmm import (
        apply_verify_qmm_patch,
        set_verify_qmm_armed,
    )

    _gpu_only()
    apply_verify_qmm_patch()
    used = _spy(monkeypatch, ("nvfp4_gather_rows", "route_topk_sorted_rows"))
    # 32 experts: the routing kernel covers multiples of 32.
    model = _inkling2_tiny(formats=formats, n_routed_experts=32, num_experts_per_tok=4)
    nn.quantize(
        model,
        group_size=64,
        bits=8,
        class_predicate=lambda _, m: isinstance(m, nn.Linear),
    )
    assert model._omlx_mtp_row_exact_verify
    from mlx.utils import tree_flatten

    keep = model.cast_predicate
    model.load_weights(
        [
            (k, v.astype(mx.bfloat16))
            for k, v in tree_flatten(model.parameters())
            if v.dtype == mx.float32 and keep(k)
        ],
        strict=False,
    )
    ids = mx.random.randint(0, 300, (1, 24 + width), key=mx.random.key(width))

    def prefilled():
        cache = model.make_cache()
        model(ids[:, :24], cache=cache)
        return cache

    cache = prefilled()
    serial = [
        model(ids[:, t : t + 1], cache=cache)[0, -1] for t in range(24, 24 + width)
    ]
    cache = prefilled()
    set_verify_qmm_armed(True, row_exact=True)
    try:
        block = model(ids[:, 24 : 24 + width], cache=cache)[0]
    finally:
        set_verify_qmm_armed(False)
    mx.eval(block, serial)
    assert used["nvfp4_gather_rows"] > 0 and used["route_topk_sorted_rows"] > 0
    for row, expected in enumerate(serial):
        assert mx.array_equal(block[row], expected)


def test_expert_kernels_fall_back_when_they_do_not_fit(monkeypatch):
    """Shapes or dtypes the kernels do not cover take the stock path: float16
    activations, more rows than the grouped kernel holds, MXFP8 experts, and a
    GPU whose threadgroup limit rejects the downs kernel."""
    _gpu_only()
    from omlx.patches.inkling2 import kernels
    from omlx.patches.inkling2 import model as inkling2

    model = _inkling2_tiny(formats=("nvfp4",) * 6 + ("mxfp8",) * 6)
    moe = model.model.layers[0].mlp
    ex = moe.experts
    x = mx.random.normal((1, 128), key=mx.random.key(9))
    inds = _routing(1, 2, 8, seed=3)
    w = mx.full((1, 2), 0.5)
    assert ex._grouped(x.astype(mx.float16), inds, w) is None
    assert model.model.layers[11].mlp.experts._grouped(x.astype(mx.bfloat16), inds, w) is None
    state = mx.zeros((1, 3, 128))
    residual = mx.zeros((1, 1, 128))
    conv_w = mx.zeros((128, 4))
    assert moe.decode_conv(x.astype(mx.float16), None, conv_w, state, 0.5, residual) is None

    called = []
    monkeypatch.setattr(
        inkling2.Inkling2Experts,
        "_grouped",
        lambda self, *a, _orig=inkling2.Inkling2Experts._grouped: called.append(1) or _orig(self, *a),
    )
    many = mx.random.normal((inkling2._GROUPED_MAX_ROWS + 1, 128)).astype(mx.bfloat16)
    many_inds = _routing(many.shape[0], 2, 8, seed=4)
    ex.unreduced(many, many_inds, mx.full(many_inds.shape, 0.5))
    assert not called

    monkeypatch.setattr(kernels, "_fits_threadgroup", lambda key, launch: False)
    one = x.astype(mx.bfloat16)
    assert moe.decode_conv(one, None, conv_w, state, 0.5, residual) is None
