# SPDX-License-Identifier: Apache-2.0
"""Attention and attention-residual primitives for Inkling 2."""

from __future__ import annotations

import mlx.core as mx

from ...custom_kernels.nax import is_nax_available

# Upper bound on one additive mask block (elements) in the general attention
# path; queries are split into sub-blocks so the mask stays below it.
_MASK_BUDGET = 1 << 29


_ATTN_RES_TG = 256
_attn_res_kernels: dict = {}


def _attn_res_kernel(blocks: int, queries: int, raw: bool = False):
    """Fused depth mixing for ``blocks`` completed blocks plus the partial sum.

    One threadgroup per token row: dot products of every source with every
    query, softmax over sources, then the weighted sum, without stacking the
    sources.
    """
    key = (blocks, queries, raw)
    kernel = _attn_res_kernels.get(key)
    if kernel is not None:
        return kernel
    srcs = [f"b{i}" for i in range(blocks)] + ["p"]
    loads = "\n".join(
        f"        float s{i} = {name}[base + h];" for i, name in enumerate(srcs)
    )
    if raw:
        # The partial sum arrives un-normalised; scale by the unit-RMS factor.
        loads = loads.replace(
            f"float s{blocks} = p[base + h];",
            f"float s{blocks} = static_cast<float>(p[base + h]) * inv;",
        )
    dots = "\n".join(
        f"        acc[{i} * Q + qi] += s{i} * qv;" for i in range(len(srcs))
    )
    mix = " + ".join(f"w[{i} * Q + qi] * s{i}" for i in range(len(srcs)))
    source = f"""
    constexpr uint NS = {len(srcs)};
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    size_t base = (size_t)row * H;
{_RAW_RMS_PROLOGUE if raw else ""}
    float acc[NS * Q];
    for (uint i = 0; i < NS * Q; ++i) acc[i] = 0.0f;
    for (uint h = tid; h < H; h += TG) {{
{loads}
        for (uint qi = 0; qi < Q; ++qi) {{
            float qv = query[qi * H + h];
{dots}
        }}
{"        pn[base + h] = s" + str(blocks) + ";" if raw else ""}
    }}
    threadgroup float red[NS * Q][TG / 32];
    for (uint i = 0; i < NS * Q; ++i) {{
        float v = simd_sum(acc[i]);
        if (lane == 0) red[i][sg] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float w[NS * Q];
    for (uint i = 0; i < NS * Q; ++i) {{
        float t = 0.0f;
        for (uint j = 0; j < TG / 32; ++j) t += red[i][j];
        w[i] = t * scale[0];
    }}
    for (uint qi = 0; qi < Q; ++qi) {{
        float m = -INFINITY;
        for (uint n = 0; n < NS; ++n) m = max(m, w[n * Q + qi]);
        float z = 0.0f;
        for (uint n = 0; n < NS; ++n) {{
            w[n * Q + qi] = metal::exp(w[n * Q + qi] - m);
            z += w[n * Q + qi];
        }}
        for (uint n = 0; n < NS; ++n) w[n * Q + qi] /= z;
    }}
    for (uint h = tid; h < H; h += TG) {{
{loads}
        for (uint qi = 0; qi < Q; ++qi) {{
            out[((size_t)row * Q + qi) * H + h] = {mix};
        }}
    }}
"""
    kernel = mx.fast.metal_kernel(
        name=f"omlx_inkling2_attn_res_{blocks}_{queries}{'_raw' if raw else ''}",
        input_names=[*srcs, "query", "scale"],
        output_names=["out", "pn"] if raw else ["out"],
        source=source,
    )
    _attn_res_kernels[key] = kernel
    return kernel


# Unit-RMS factor of the raw partial sum, with MLX's ``rms_single_row``
# arithmetic for an FP32 row of H: 4 consecutive elements per thread over
# ceil(H/4) threads rounded up to whole simdgroups, simd_sum per simdgroup,
# then simd_sum of the (zero-padded) 32 simdgroup sums and precise rsqrt.
# The TG real threads cover those virtual threads in rounds, keeping each
# virtual simdgroup on one real simdgroup, so every sum groups identically.
_RAW_RMS_PROLOGUE = """
    threadgroup float rms_sums[32];
    threadgroup float rms_inv[1];
    if (tid < 32) rms_sums[tid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr uint VT = ((H + 3) / 4 + 31) / 32 * 32;
    for (uint r0 = 0; r0 < VT; r0 += TG) {
        uint vt = r0 + tid;
        float racc = 0.0f;
        if (vt < VT) {
            for (int i = 0; i < 4; i++) {
                uint c = vt * 4 + i;
                if (c < H) {
                    float xi = static_cast<float>(p[base + c]);
                    racc += xi * xi;
                }
            }
        }
        racc = simd_sum(racc);
        uint vsg = (r0 + sg * 32) / 32;
        if (lane == 0 && vsg < 32) rms_sums[vsg] = racc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float tot = simd_sum(rms_sums[lane]);
        if (lane == 0) rms_inv[0] = metal::precise::rsqrt(tot / H + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rms_inv[0];
"""


def attn_res_mix_raw(blocks: list, partial: mx.array, query: mx.array, scale: float):
    """``attn_res_mix(blocks, unit_rms(partial.astype(f32)), ...)`` with the
    norm computed in the same kernel (bit-identical). Returns the mixtures
    and the unit-RMS partial sum [B, L, H] FP32."""
    b, length, hidden = partial.shape
    nq = query.shape[0]
    rows = b * length
    kernel = _attn_res_kernel(len(blocks), nq, raw=True)
    out, pn = kernel(
        inputs=[
            *[x.astype(mx.float32) for x in blocks],
            partial,
            query.astype(mx.float32),
            mx.array([scale], dtype=mx.float32),
        ],
        template=[("H", hidden), ("Q", nq), ("TG", _ATTN_RES_TG)],
        grid=(rows * _ATTN_RES_TG, 1, 1),
        threadgroup=(_ATTN_RES_TG, 1, 1),
        output_shapes=[(b, length, nq, hidden), (b, length, hidden)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return out, pn


def attn_res_mix(blocks: list, partial_n: mx.array, query: mx.array, scale: float):
    """Mix unit-RMS depth sources with learned queries.

    ``blocks`` holds [B, L, H] FP32 sources, ``partial_n`` is the current
    partial sum (also unit-RMS) and ``query`` is [Q, H]. Returns [B, L, Q, H].
    """
    if mx.default_device() != mx.gpu:
        src = mx.stack([*blocks, partial_n], axis=-2)
        logits = (src @ query.T) * scale
        w = mx.softmax(logits, axis=-2, precise=True)
        return w.swapaxes(-1, -2) @ src
    b, length, hidden = partial_n.shape
    nq = query.shape[0]
    rows = b * length
    kernel = _attn_res_kernel(len(blocks), nq)
    out = kernel(
        inputs=[
            *[x.astype(mx.float32) for x in blocks],
            partial_n.astype(mx.float32),
            query.astype(mx.float32),
            mx.array([scale], dtype=mx.float32),
        ],
        template=[("H", hidden), ("Q", nq), ("TG", _ATTN_RES_TG)],
        grid=(rows * _ATTN_RES_TG, 1, 1),
        threadgroup=(_ATTN_RES_TG, 1, 1),
        output_shapes=[(b, length, nq, hidden)],
        output_dtypes=[mx.float32],
    )[0]
    return out


def _decode_attention(q, k, v, rel, sink, scale, window):
    s = k.shape[2]
    lo = max(0, s - window) if window else 0
    if lo:
        k = k[:, :, lo:]
        v = v[:, :, lo:]
    sv = s - lo
    r = rel.shape[-1]
    n = min(r, sv)
    hq = q.shape[1]
    tail = rel[0, 0, :, :n][:, ::-1]
    if n < sv:
        tail = mx.concatenate([mx.zeros((hq, sv - n), dtype=tail.dtype), tail], -1)
    mask = tail.reshape(1, hq, 1, sv).astype(q.dtype)
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=mask, sinks=sink[0, 0].astype(q.dtype)
    )


# Flash attention with the relative-bias band and per-query sinks, for
# prefill-sized BF16 query blocks. One threadgroup runs 32 queries of one head
# in four simdgroups (8 rows each) over 16-key tiles staged in half. Scores
# live in the log2 domain; each row's online softmax starts from its sink
# logit, so the zero-value sink is exact without an extra key.
_FLASH_SOURCE = """
    constexpr int BQ = 32;
    constexpr int BK = 16;
    constexpr int NT = 128;
    constexpr float LOG2E = 1.4426950408889634f;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const int qt = threadgroup_position_in_grid.x;
    const int h = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int L = dims[0];
    const int S = dims[1];
    const int lp = dims[2 + b];
    const int hk = h / (HQ / HKV);
    const int q0 = qt * BQ;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const int row = q0 + int(sg) * 8 + fm;
    const bool row_ok = row < L;
    const int a_row = S - L + row;
    const float qscale = scale[0] * LOG2E;

    // Q, K and V are BF16 values within half's range (L2-clipped q/k,
    // unit-RMS v), so staging them in half is exact; scores accumulate in
    // FP32 and the softmax scale is applied there. K is stored transposed
    // and both tiles are padded by 16 bytes per row against bank conflicts.
    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BK * LDV];

    simdgroup_half8x8 Qf[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        float2 qv = float2(0.0f);
        if (row_ok) {
            const device T* qp = q + b * q_strides[0] + h * q_strides[1]
                + row * q_strides[2] + (db * 8 + fn) * q_strides[3];
            qv = float2(float(qp[0]), float(qp[q_strides[3]]));
        }
        Qf[db].thread_elements()[0] = half(qv.x);
        Qf[db].thread_elements()[1] = half(qv.y);
    }
    simdgroup_float8x8 Of[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        Of[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float m = row_ok ? float(sink[(b * L + row) * HQ + h]) * LOG2E : 0.0f;
    float l = 1.0f;
    const device T* relp = rel + ((size_t)(b * L + (row_ok ? row : 0)) * HQ + h) * R;

    const int a_first = S - L + q0;
    const int a_last = S - L + min(q0 + BQ, L) - 1;
    int lo = WINDOW > 0 ? max(0, a_first - WINDOW + 1) : 0;
    lo = max(lo, lp);
    const int k_begin = (lo / BK) * BK;
    const device T* kb_ptr = k + b * k_strides[0] + hk * k_strides[1];
    const device T* vb_ptr = v + b * v_strides[0] + hk * v_strides[1];
    const bool vec_kv = k_strides[3] == 1 && v_strides[3] == 1;

    for (int k0 = k_begin; k0 <= a_last; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (vec_kv) {
            // 8 BF16 values (16 bytes) per load.
            for (int c = tid; c < BK * D / 8; c += NT) {
                const int r = c / (D / 8);
                const int col = (c % (D / 8)) * 8;
                const int key = k0 + r;
                uint4 kw = uint4(0);
                uint4 vw = uint4(0);
                if (key < S) {
                    kw = *(const device uint4*)(kb_ptr + key * k_strides[2] + col);
                    vw = *(const device uint4*)(vb_ptr + key * v_strides[2] + col);
                }
                threadgroup half* kd = Ks + col * LDK + r;
                threadgroup half* vd = Vs + r * LDV + col;
                for (int j = 0; j < 4; ++j) {
                    kd[(2 * j) * LDK] = half(as_type<float>(kw[j] << 16));
                    kd[(2 * j + 1) * LDK] = half(as_type<float>(kw[j] & 0xFFFF0000u));
                    vd[2 * j] = half(as_type<float>(vw[j] << 16));
                    vd[2 * j + 1] = half(as_type<float>(vw[j] & 0xFFFF0000u));
                }
            }
        } else {
            for (int c = tid; c < BK * D; c += NT) {
                const int r = c / D;
                const int col = c % D;
                const int key = k0 + r;
                float kv = 0.0f;
                float vv = 0.0f;
                if (key < S) {
                    kv = float(kb_ptr[key * k_strides[2] + col * k_strides[3]]);
                    vv = float(vb_ptr[key * v_strides[2] + col * v_strides[3]]);
                }
                Ks[col * LDK + r] = half(kv);
                Vs[r * LDV + col] = half(vv);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + int(sg) * 8 >= L) {
            continue;
        }

        simdgroup_float8x8 Sf[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            Sf[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 Kt;
                simdgroup_load(Kt, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(Sf[kb], Qf[db], Kt, Sf[kb]);
            }
        }

        float rmax = -INFINITY;
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const int key = k0 + kb * 8 + fn + e;
                const int d = a_row - key;
                const bool vis = row_ok && d >= 0 && key >= lp
                    && (WINDOW == 0 || d < WINDOW);
                float s = Sf[kb].thread_elements()[e] * qscale;
                if (vis) {
                    if (d < R) {
                        s += float(relp[d]) * LOG2E;
                    }
                } else {
                    s = -INFINITY;
                }
                Sf[kb].thread_elements()[e] = s;
                rmax = max(rmax, s);
            }
        }
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        rmax = max(rmax, simd_shuffle_xor(rmax, 8));
        const float m_new = max(m, rmax);
        const float factor = fast::exp2(m - m_new);
        float rsum = 0.0f;
        simdgroup_half8x8 Pf[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float p = fast::exp2(Sf[kb].thread_elements()[e] - m_new);
                Pf[kb].thread_elements()[e] = half(p);
                rsum += p;
            }
        }
        rsum += simd_shuffle_xor(rsum, 1);
        rsum += simd_shuffle_xor(rsum, 8);
        l = l * factor + rsum;
        m = m_new;
        for (int db = 0; db < D / 8; ++db) {
            Of[db].thread_elements()[0] *= factor;
            Of[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_half8x8 Vm;
                simdgroup_load(Vm, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(Of[db], Pf[kb], Vm, Of[db]);
            }
        }
    }

    if (row_ok) {
        const float inv = 1.0f / l;
        device T* op = out + ((size_t)(b * L + row) * HQ + h) * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = T(Of[db].thread_elements()[0] * inv);
            op[db * 8 + fn + 1] = T(Of[db].thread_elements()[1] * inv);
        }
    }
"""
_flash_kernel = None
# Query blocks shorter than this keep the SDPA path (decode and verify).
_FLASH_MIN_QUERIES = 32


def _use_flash(length: int, window: int) -> bool:
    """Prefill blocks take the flash kernel on the GPU.

    On hosts with tensor units (NAX) MLX's masked SDPA runs on them, which
    this kernel does not; there only sliding layers, whose cost is dominated
    by building the mask, switch to it.
    """
    if length < _FLASH_MIN_QUERIES or mx.default_device() != mx.gpu:
        return False
    return bool(window) or not is_nax_available()


def flash_banded_attention(q, k, v, rel, sink, *, scale, window, left_padding=None):
    """``banded_sink_attention`` for BF16 prefill blocks; same arguments and result."""
    global _flash_kernel
    b, hq, length, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    if _flash_kernel is None:
        _flash_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_flash_banded",
            input_names=["q", "k", "v", "rel", "sink", "dims", "scale"],
            output_names=["out"],
            source=_FLASH_SOURCE,
            ensure_row_contiguous=False,
        )
    lp = (
        left_padding.astype(mx.int32)
        if left_padding is not None
        else mx.zeros((b,), dtype=mx.int32)
    )
    dims = mx.concatenate([mx.array([length, s], dtype=mx.int32), lp])
    out = _flash_kernel(
        inputs=[
            q,
            k,
            v,
            mx.contiguous(rel),
            mx.contiguous(sink.astype(mx.float32)),
            dims,
            mx.array([scale], dtype=mx.float32),
        ],
        template=[
            ("T", q.dtype),
            ("D", d),
            ("HQ", hq),
            ("HKV", hkv),
            ("R", rel.shape[-1]),
            ("WINDOW", int(window)),
        ],
        grid=(((length + 31) // 32) * 128, hq, b),
        threadgroup=(128, 1, 1),
        output_shapes=[(b, length, hq, d)],
        output_dtypes=[q.dtype],
    )[0]
    return out.transpose(0, 2, 1, 3)


def banded_sink_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rel: mx.array,
    sink: mx.array,
    *,
    scale: float,
    window: int = 0,
    left_padding=None,
) -> mx.array:
    """Attention with a relative bias band and per-query sink logits.

    q: [B, Hq, L, D]; k, v: [B, Hkv, S, D] (queries are the last L keys);
    rel: [B, L, Hq, R] bias by backward distance d (applied for d < R);
    sink: [B, L, Hq] logits of a zero-value sink. ``window`` > 0 limits
    keys to d < window (sliding layers use window == R).
    """
    b, length = q.shape[0], q.shape[2]
    if b == 1 and length == 1 and left_padding is None:
        return _decode_attention(q, k, v, rel, sink, scale, window)
    if q.dtype == mx.bfloat16 and _use_flash(length, window):
        return flash_banded_attention(
            q, k, v, rel, sink, scale=scale, window=window, left_padding=left_padding
        )
    return _masked_attention(q, k, v, rel, sink, scale, window, left_padding)


# Verify-sized query blocks: the band's index arrays depend only on the
# shapes, so one forward builds them once per (keys, rows, band) and every
# layer visit reuses them. Reset at the start of each text forward.
_BAND_MEMO_MAX_ROWS = 16
_band_memo: dict = {}


def reset_band_memo():
    _band_memo.clear()


def _band_index(a0, f, nq, near, r, window, b, hq):
    dist = (a0 - f) + mx.arange(nq)[:, None] - mx.arange(near)[None, :]
    gidx = mx.broadcast_to(mx.clip(dist, 0, r - 1), (b, hq, nq, near))
    vis = dist >= 0
    if window:
        vis = vis & (dist < window)
    return gidx, vis, dist < r


@mx.compile
def _near_mask(bias, vis, in_band, neg):
    return mx.where(vis, mx.where(in_band, bias, 0), neg).astype(neg.dtype)


# Verify-sized blocks: the whole additive mask (sink column, then the keys
# lo..s-1) in one pass, the same selections as the take/where/concat chain.
_VERIFY_MASK_SOURCE = """
    const uint gid = thread_position_in_grid.x;
    const int W = dims[0], nq = dims[1], hq = dims[2], R = dims[3];
    const int a0 = dims[4], lo = dims[5], window = dims[6], L = dims[7], i0 = dims[8];
    const int total = dims[9];
    const int sc = dims[10];  // 1: column 0 carries the sink logit; 0: keys only
    if (int(gid) >= total) return;
    const int c = int(gid) % W + (1 - sc);
    int t = int(gid) / W;
    const int i = t % nq;
    t /= nq;
    const int h = t % hq;
    const int bb = t / hq;
    const int qi = i0 + i;
    T val;
    if (c == 0) {
        val = sink[bb * sink_strides[0] + qi * sink_strides[1] + h * sink_strides[2]];
    } else {
        const int dist = a0 + i - (lo + c - 1);
        if (dist < 0 || (window > 0 && dist >= window)) {
            val = T(-INFINITY);
        } else if (dist < R) {
            val = rel[bb * rel_strides[0] + qi * rel_strides[1] + h * rel_strides[2]
                      + dist * rel_strides[3]];
        } else {
            val = T(0);
        }
    }
    out[gid] = val;
"""
_verify_mask_kernel = None


def _verify_mask(rel, sink, dims, shape):
    global _verify_mask_kernel
    if _verify_mask_kernel is None:
        _verify_mask_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_verify_mask",
            input_names=["rel", "sink", "dims"],
            output_names=["out"],
            source=_VERIFY_MASK_SOURCE,
            ensure_row_contiguous=False,
        )
    total = shape[0] * shape[1] * shape[2] * shape[3]
    return _verify_mask_kernel(
        inputs=[rel, sink, dims],
        template=[("T", rel.dtype)],
        grid=(((total + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape],
        output_dtypes=[rel.dtype],
    )[0]


_FOLDED_VERIFY = True


def _masked_attention(q, k, v, rel, sink, scale, window, left_padding):
    """MLX SDPA over an additive mask that carries the bias band and sinks."""
    b, hq, length, d = q.shape
    s = k.shape[2]
    r = rel.shape[-1]
    if window:
        assert window == r, "sliding layers expect rel_extent == window"
    lo = max(0, s - length - window + 1) if window else 0
    hkv = k.shape[1]
    # A zero key/value row carries the sink: its logit comes from the mask.
    zero = mx.zeros((b, hkv, 1, d), dtype=k.dtype)
    kz = mx.concatenate([zero, k[:, :, lo:]], axis=2)
    vz = mx.concatenate([zero, v[:, :, lo:]], axis=2)
    nk = s - lo
    qb = max(16, min(length, _MASK_BUDGET // max(1, b * hq * (nk + 1))))
    neg = mx.array(-mx.inf, dtype=q.dtype)
    lp_ok = None
    if left_padding is not None:
        lp_ok = (lo + mx.arange(nk))[None, :] >= left_padding[:, None]
        lp_ok = mx.concatenate([mx.ones((b, 1), dtype=mx.bool_), lp_ok], axis=1)
        lp_ok = lp_ok[:, None, None, :]

    outs = []
    for i0 in range(0, length, qb):
        i1 = min(length, i0 + qb)
        nq = i1 - i0
        a0 = s - length + i0
        hi = s - length + i1
        f = max(lo, a0 - r + 1)
        near = hi - f
        if (
            length <= _BAND_MEMO_MAX_ROWS
            and rel.dtype == q.dtype
            and sink.dtype == q.dtype
            and mx.default_device() == mx.gpu
        ):
            if _FOLDED_VERIFY and b == 1 and lp_ok is None:
                # One query per (head, row): MLX's single-query SDPA with a
                # per-(head, row) sink reads K/V in place (no zero sink row
                # concatenated onto the whole window).
                key = ("fold", a0, lo, nq, nk, r, window, b, hq, length, i0)
                dims = _band_memo.get(key)
                if dims is None:
                    total = b * hq * nq * nk
                    dims = _band_memo[key] = mx.array(
                        [nk, nq, hq, r, a0, lo, window, length, i0, total, 0],
                        dtype=mx.int32,
                    )
                mask = _verify_mask(rel, sink, dims, (b, hq, nq, nk))
                sinks = sink[0, i0:i1].transpose(1, 0).reshape(-1)
                out = mx.fast.scaled_dot_product_attention(
                    q[:, :, i0:i1].reshape(b, hq * nq, 1, d),
                    k[:, :, lo:],
                    v[:, :, lo:],
                    scale=scale,
                    mask=mask.reshape(b, hq * nq, 1, nk),
                    sinks=sinks,
                )
                outs.append(out.reshape(b, hq, nq, d))
                continue
            key = ("mask", a0, lo, nq, nk, r, window, b, hq, length, i0)
            dims = _band_memo.get(key)
            if dims is None:
                total = b * hq * nq * (nk + 1)
                dims = _band_memo[key] = mx.array(
                    [nk + 1, nq, hq, r, a0, lo, window, length, i0, total, 1],
                    dtype=mx.int32,
                )
            mask = _verify_mask(rel, sink, dims, (b, hq, nq, nk + 1))
            if lp_ok is not None:
                mask = mx.where(lp_ok, mask, neg)
            out = mx.fast.scaled_dot_product_attention(
                q[:, :, i0:i1], kz, vz, scale=scale, mask=mask
            )
            outs.append(out)
            continue
        relq = rel[:, i0:i1].transpose(0, 2, 1, 3)
        if length <= _BAND_MEMO_MAX_ROWS:
            key = (a0, f, nq, near, r, window, b, hq)
            band = _band_memo.get(key)
            if band is None:
                band = _band_memo[key] = _band_index(*key)
            gidx, vis, in_band = band
            bias = mx.take_along_axis(relq, gidx, axis=-1)
            near_mask = _near_mask(bias, vis, in_band, neg)
        else:
            gidx, vis, in_band = _band_index(a0, f, nq, near, r, window, b, hq)
            bias = mx.take_along_axis(relq, gidx, axis=-1)
            near_mask = mx.where(vis, mx.where(in_band, bias, 0), neg)
        parts = [sink[:, i0:i1].transpose(0, 2, 1)[..., None].astype(q.dtype)]
        if f > lo:
            fill = neg if window else mx.array(0, dtype=q.dtype)
            parts.append(mx.broadcast_to(fill, (b, hq, nq, f - lo)))
        parts.append(near_mask.astype(q.dtype))
        if hi < s:
            parts.append(mx.broadcast_to(neg, (b, hq, nq, s - hi)))
        mask = mx.concatenate(parts, axis=-1)
        if lp_ok is not None:
            mask = mx.where(lp_ok, mask, neg)
        out = mx.fast.scaled_dot_product_attention(
            q[:, :, i0:i1], kz, vz, scale=scale, mask=mask
        )
        if qb < length:
            # A lazy graph would hold every block's mask at once (12 GiB at
            # 64K keys); evaluated block by block, one mask is alive.
            mx.eval(out)
        outs.append(out)
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


_ROUTER_SOURCE = """
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    constexpr uint PER = E / 32;
    float sc[PER];
    float key[PER];
    for (uint j = 0; j < PER; ++j) {
        uint e = lane + 32 * j;
        float s = 1.0f / (1.0f + metal::exp(-logits[row * E + e]));
        sc[j] = s;
        key[j] = s + bias[e];
    }
    float picked[K];
    uint chosen[K];
    float total = 0.0f;
    for (uint k = 0; k < K; ++k) {
        float best = -INFINITY;
        uint bj = 0;
        for (uint j = 0; j < PER; ++j) {
            if (key[j] > best) {
                best = key[j];
                bj = j;
            }
        }
        float top = simd_max(best);
        uint mine = (best == top) ? (lane + 32 * bj) : 0xFFFFFFFFu;
        uint win = simd_min(mine);
        float s = (lane == win % 32) ? sc[win / 32] : 0.0f;
        s = simd_sum(s);
        if (lane == win % 32) key[win / 32] = -INFINITY;
        picked[k] = s;
        chosen[k] = win;
        total += s;
    }
    if (lane == 0) {
        float norm = route_scale[0] / (total + 1e-20f);
        for (uint k = 0; k < K; ++k) {
            inds[row * K + k] = chosen[k];
            weights[row * K + k] = picked[k] * norm;
        }
    }
"""
_router_kernel = None


def route_topk(logits: mx.array, bias: mx.array, top_k: int, route_scale: float):
    """Sigmoid scores, top-k by ``score + bias``, renormalized unbiased weights.

    ``logits``: [N, E] FP32. Returns ([N, K] uint32 indices, [N, K] FP32).
    """
    global _router_kernel
    rows, experts = logits.shape
    if mx.default_device() != mx.gpu or experts % 32:
        scores = mx.sigmoid(logits)
        inds = mx.argpartition(-(scores + bias), kth=top_k - 1, axis=-1)[..., :top_k]
        w = mx.take_along_axis(scores, inds, axis=-1)
        w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20) * route_scale
        return inds, w
    if _router_kernel is None:
        _router_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_route_topk",
            input_names=["logits", "bias", "route_scale"],
            output_names=["inds", "weights"],
            source=_ROUTER_SOURCE,
        )
    inds, weights = _router_kernel(
        inputs=[
            logits.astype(mx.float32),
            bias.astype(mx.float32),
            mx.array([route_scale], dtype=mx.float32),
        ],
        template=[("E", experts), ("K", top_k)],
        grid=(rows * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, top_k), (rows, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return inds, weights


# Routed NVFP4 expert projection for decode/verify-sized row batches.
#
# ``gather_qmm`` (unsorted) runs one mat-vec per (row, expert) pair, so an
# expert picked by several verify rows is streamed from memory once per row.
# Here a threadgroup owns one distinct expert and an 8-column output tile: it
# reads that weight tile once and applies it to every pair routed to the
# expert. Per pair, the arithmetic is MLX 0.32.4's ``fp_qmv`` traversal for
# NVFP4 (``fp_quantized.h``: 8 codes per lane, 256-value blocks, the E4M3
# group scale applied per lane block, then ``simd_sum``), so each output
# should equal the ``gather_qmm`` output bit for bit; the tests check that.
_NVFP4_ROWS_SOURCE = """
    uint p = threadgroup_position_in_grid.y;
    uint e = sorted_e[p];
    if (p > 0 && sorted_e[p - 1] == e) {
        return;
    }
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    int cnt = 1;
    while (cnt < MAXR && p + cnt < P && sorted_e[p + cnt] == e) {
        cnt++;
    }
    constexpr int KW = K / 8;
    constexpr int KG = K / 16;
    int out_row = threadgroup_position_in_grid.x * 8 + simd_gid * 4;
    const device uint8_t* ws =
        (const device uint8_t*)(w + ((size_t)e * N + out_row) * KW) + lane * 4;
    const device uint8_t* sc = scales + ((size_t)e * N + out_row) * KG + lane / 2;
    uint dst[MAXR];
    const device T* xs[MAXR];
    for (int j = 0; j < MAXR; ++j) {
        uint pair = order[p + min(j, cnt - 1)];
        dst[j] = pair;
        xs[j] = x + (size_t)(pair / SRC_DIV) * K + lane * 8;
    }
    float result[MAXR][4];
    for (int j = 0; j < MAXR; ++j) {
        for (int r = 0; r < 4; ++r) {
            result[j][r] = 0.0f;
        }
    }
    for (int k = 0; k < K; k += 256) {
        if (k + (int)lane * 8 < K) {
            float s[4];
            float wd[4][8];
            for (int r = 0; r < 4; ++r) {
                const device ushort* w16 = (const device ushort*)(ws + r * KW * 4);
                for (int i = 0; i < 2; ++i) {
                    ushort q = w16[i];
                    wd[r][4 * i] = fp4v(q);
                    wd[r][4 * i + 1] = fp4v(q >> 4);
                    wd[r][4 * i + 2] = fp4v(q >> 8);
                    wd[r][4 * i + 3] = fp4v(q >> 12);
                }
                uchar sb = sc[r * KG];
                ushort v = sb & 127;
                ushort u = (v << 7) | (((v + 1) >> 7) << 14) | (((ushort)((sb >> 7) & 1)) << 15);
                s[r] = float(as_type<half>(u) * 256.0h);
            }
            for (int j = 0; j < MAXR; ++j) {
                if (j < cnt) {
                    float xt[8];
                    for (int i = 0; i < 8; ++i) {
                        xt[i] = float(xs[j][k + i]);
                    }
                    for (int r = 0; r < 4; ++r) {
                        float accum = 0.0f;
                        for (int i = 0; i < 2; ++i) {
                            accum += (xt[4 * i] * wd[r][4 * i] + xt[4 * i + 1] * wd[r][4 * i + 1] +
                                      xt[4 * i + 2] * wd[r][4 * i + 2] + xt[4 * i + 3] * wd[r][4 * i + 3]);
                        }
                        result[j][r] += s[r] * accum;
                    }
                }
            }
        }
        ws += 128;
        sc += 16;
    }
    for (int j = 0; j < MAXR; ++j) {
        if (j < cnt) {
            for (int r = 0; r < 4; ++r) {
                float v = simd_sum(result[j][r]);
                if (lane == 0) {
                    y[(size_t)dst[j] * N + out_row + r] = static_cast<T>(v);
                }
            }
        }
    }
"""
_NVFP4_ROWS_HEADER = """
inline float fp4v(ushort b) {
    half c = as_type<half>(ushort((b & 7) << 9));
    c *= 16384.0h;
    return float((b & 8) ? -c : c);
}
"""
_nvfp4_rows_kernel = None
_nvfp4_rows_exact_kernel = None
# OMLX_INKLING2_EXACT_VERIFY=1: verify rows (2-8) also take the unfused kernel,
# so they equal stock gather_qmm bit for bit (costs ~0.6 ms per 4-row verify).
import os as _os

_EXACT_VERIFY_EXPERTS = _os.environ.get("OMLX_INKLING2_EXACT_VERIFY", "") == "1"


# Loop-controller projections for decode/verify-sized rows: FP32 inputs times
# BF16 weights (the controllers' FP32 copies hold BF16 values), FP32 out.
# Per output column the arithmetic is MLX's FP32 ``gemv_t`` with SM = 8,
# TM = 4: eight partial sums, partial j over input rows i * 32 + j * 4 + tm in
# order, combined by the simd_shuffle_down tree ((p0+p4)+(p2+p6)) +
# ((p1+p5)+(p3+p7)). Here partial j is simdgroup j and the lanes are 32
# consecutive output columns, so each weight row is read once (half the bytes
# of the FP32 copy) for every row of the batch.
_LOOP_GEMV_SOURCE = """
    uint t = thread_position_in_threadgroup.x;
    uint lane = t % CPT;
    uint j = t / CPT;
    uint col = threadgroup_position_in_grid.x * CPT + lane;
    uint bi = threadgroup_position_in_grid.y;
    const device bfloat16_t* wp = w + (size_t)bi * K * N + col;
    auto zp = z + (size_t)bi * ROWS * K;
    float acc[ROWS];
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;
    // Weight loads for UNROLL blocks are issued before their FMAs (the
    // accumulation order is unchanged), so more loads are in flight.
    constexpr int UNROLL = (K / 32) % 4 == 0 ? 4 : 1;
    for (int i0 = 0; i0 < K / 32; i0 += UNROLL) {
        float wv[UNROLL][4];
        for (int u = 0; u < UNROLL; ++u) {
            for (int tm = 0; tm < 4; ++tm) {
                wv[u][tm] = static_cast<float>(wp[(size_t)((i0 + u) * 32 + j * 4 + tm) * N]);
            }
        }
        for (int u = 0; u < UNROLL; ++u) {
            for (int tm = 0; tm < 4; ++tm) {
                int k = (i0 + u) * 32 + j * 4 + tm;
                for (int r = 0; r < ROWS; ++r) {
                    float vc = static_cast<float>(zp[r * K + k]);
                    acc[r] += vc * wv[u][tm];
                }
            }
        }
    }
    threadgroup float red[8][ROWS][CPT];
    for (int r = 0; r < ROWS; ++r) red[j][r][lane] = acc[r];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (j == 0) {
        for (int r = 0; r < ROWS; ++r) {
            float p0 = red[0][r][lane], p1 = red[1][r][lane], p2 = red[2][r][lane], p3 = red[3][r][lane];
            float p4 = red[4][r][lane], p5 = red[5][r][lane], p6 = red[6][r][lane], p7 = red[7][r][lane];
            float a0 = p0 + p4, a1 = p1 + p5, a2 = p2 + p6, a3 = p3 + p7;
            float b0 = a0 + a2, b1 = a1 + a3;
            out[((size_t)bi * ROWS + r) * N + col] = b0 + b1;
        }
    }
"""
_loop_gemv_kernel = None


def loop_gemv(z, w):
    """``z @ w.astype(float32)`` for ``z`` [B, rows, K] FP32 and ``w`` [B, K, N]
    BF16, bit-equal to MLX's FP32 matrix-vector product per row."""
    global _loop_gemv_kernel
    if _loop_gemv_kernel is None:
        _loop_gemv_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_loop_gemv",
            input_names=["z", "w"],
            output_names=["out"],
            source=_LOOP_GEMV_SOURCE,
        )
    b, rows, k = z.shape
    n = w.shape[-1]
    # Narrow outputs (the down projection) take 8-column threadgroups so the
    # work spreads over more GPU cores; one thread per (partial, column).
    cpt = 8 if n <= 256 else 32
    (out,) = _loop_gemv_kernel(
        inputs=[z, w],
        template=[("K", k), ("N", n), ("ROWS", rows), ("CPT", cpt)],
        grid=((n // cpt) * 8 * cpt, b, 1),
        threadgroup=(8 * cpt, 1, 1),
        output_shapes=[(b, rows, n)],
        output_dtypes=[mx.float32],
    )
    return out


def nvfp4_gather_rows(x, w, scales, order, sorted_e, src_div: int, max_rows: int):
    """``y[pair] = x[pair // src_div] @ W[expert(pair)]^T`` for NVFP4 experts.

    ``x``: [Q, K] activations; ``w``/``scales``: the [E, N, K/8] uint32 codes
    and [E, N, K/16] E4M3 scales; ``order``/``sorted_e``: the pairs in expert
    order and their experts (``argsort`` of the flat routing). ``max_rows``
    bounds how many pairs share one expert. Returns [P, N] in ``x.dtype``,
    without the NVFP4 tensor scale (like ``gather_qmm``).
    """
    global _nvfp4_rows_kernel, _nvfp4_rows_exact_kernel
    if _nvfp4_rows_kernel is None:
        _nvfp4_rows_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_nvfp4_gather_rows",
            input_names=["x", "w", "scales", "sorted_e", "order"],
            output_names=["y"],
            header=_NVFP4_ROWS_HEADER,
            source=_NVFP4_ROWS_SOURCE,
        )
        # MLX's precompiled qmv does not fuse the multiply-adds; this JIT
        # kernel would, and the one-ulp differences then flip a bf16 output
        # now and then (why loop attempt 6 failed the server guard). The
        # single-row decode call uses the unfused variant, bit-identical to
        # gather_qmm; verify rows keep the faster fused one.
        _nvfp4_rows_exact_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_nvfp4_gather_rows_exact",
            input_names=["x", "w", "scales", "sorted_e", "order"],
            output_names=["y"],
            header=_NVFP4_ROWS_HEADER,
            source="#pragma METAL fp contract(off)\n" + _NVFP4_ROWS_SOURCE,
        )
    exact = max_rows == 1 or _EXACT_VERIFY_EXPERTS
    kernel = _nvfp4_rows_exact_kernel if exact else _nvfp4_rows_kernel
    pairs = order.shape[0]
    n = w.shape[1]
    k = x.shape[-1]
    (y,) = kernel(
        inputs=[x, w, scales, sorted_e, order],
        template=[
            ("T", x.dtype),
            ("K", k),
            ("N", n),
            ("P", pairs),
            ("MAXR", max_rows),
            ("SRC_DIV", src_div),
        ],
        grid=(64 * (n // 8), pairs, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(pairs, n)],
        output_dtypes=[x.dtype],
    )
    return y


# Fused L2 clip for decode/verify rows: ``xf = x * scale`` (FP32),
# ``out = xf * min(thr / (||xf|| + 1e-6), 1)`` cast to ``x.dtype``, in one
# kernel instead of multiply + square + row sum + elementwise tail. The row
# sum follows MLX's own row reduction for a contiguous last-axis sum, so the
# result is bit-identical: rows of R <= 32 follow ``row_reduce_small`` (one
# thread, sequential; here lanes load and lane order is kept by shuffles); longer rows use ``row_reduce_simple`` (32 lanes, four
# consecutive elements each, summed in order, then ``simd_sum``).
_L2_CLIP_SOURCE = """
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    const float t = thr[0];
    size_t base = elem_to_loc((size_t)row * R, x_shape, x_strides, x_ndim);
    uint srow = SH ? (row % (SH ? SH : 1)) * R : 0;
    if (R <= 32) {
        // Lane i loads element i; the squares are then summed in order
        // (as one MLX thread would) through shuffles, all lanes in lockstep.
        float v = 0.0f;
        if ((int)lane < R) {
            v = static_cast<float>(x[base + lane]) * static_cast<float>(scale[SH ? srow + lane : 0]);
        }
        float sq = v * v;
        float total = 0.0f;
        for (int i = 0; i < R; ++i) {
            float si = simd_shuffle(sq, (ushort)i);
            total = si + total;
        }
        float m = metal::min(t / (metal::precise::sqrt(total) + 1e-6f), 1.0f);
        if ((int)lane < R) {
            out[(size_t)row * R + lane] = static_cast<T>(v * m);
        }
    } else {
        float xf[R / 32 > 0 ? R / 32 : 1];
        float total = 0.0f;
        for (int b = 0; b < R / 128; ++b) {
            for (int j = 0; j < 4; ++j) {
                int i = b * 128 + lane * 4 + j;
                float v = static_cast<float>(x[base + i]) * static_cast<float>(scale[SH ? srow + i : 0]);
                xf[b * 4 + j] = v;
                float sq = v * v;
                total = sq + total;
            }
        }
        total = simd_sum(total);
        float m = metal::min(t / (metal::precise::sqrt(total) + 1e-6f), 1.0f);
        for (int b = 0; b < R / 128; ++b) {
            for (int j = 0; j < 4; ++j) {
                int i = b * 128 + lane * 4 + j;
                out[(size_t)row * R + i] = static_cast<T>(xf[b * 4 + j] * m);
            }
        }
    }
"""
_l2_clip_kernel = None


def fused_l2_clip(x, scale, threshold):
    """``_l2_clip`` in one kernel. ``x`` [..., R] (last axis contiguous),
    ``scale`` [SH, R] broadcast over the leading rows or a scalar, ``threshold``
    a scalar FP32 array. R must be <= 32 or a multiple of 128."""
    global _l2_clip_kernel
    if _l2_clip_kernel is None:
        _l2_clip_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_l2_clip",
            input_names=["x", "scale", "thr"],
            output_names=["out"],
            source=_L2_CLIP_SOURCE,
            ensure_row_contiguous=False,
        )
    r = x.shape[-1]
    rows = x.size // r
    sh = 0 if scale.ndim == 0 else scale.size // r
    (out,) = _l2_clip_kernel(
        inputs=[x, scale.reshape(-1), threshold.reshape(1)],
        template=[("T", x.dtype), ("R", r), ("SH", sh)],
        grid=(32 * rows, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return out


# Routed-expert combine for decode/verify rows:
# ``out[r] = sum_j z[r, j] * (w[r, j] * (gs[e[r, j]] * inv_den))`` in FP32,
# summed over the top-k axis in order (MLX's small column reduction: one
# thread per output, sequential), then cast to ``T``. Replaces the scale
# gather, the scale/weight multiplies, the product, the sum and the cast.
_MOE_COMBINE_SOURCE = """
    uint n = thread_position_in_grid.x;
    uint r = thread_position_in_grid.y;
    if (n >= N) return;
    float total = 0.0f;
    for (int j = 0; j < TOP; ++j) {
        float wj = w[r * TOP + j];
        if (HAS_GS) {
            float s = gs[inds[r * TOP + j]] * inv_den[0];
            wj = wj * s;
        }
        float p = static_cast<float>(z[((size_t)r * TOP + j) * N + n]) * wj;
        total = p + total;
    }
    out[(size_t)r * N + n] = static_cast<T>(total);
"""
_moe_combine_kernel = None


def moe_combine(z, weights, inds, global_scale, inv_den, out_dtype):
    """``z`` [rows, top, N] (row-contiguous), ``weights`` [rows, top] FP32,
    ``inds`` [rows, top] expert ids, ``global_scale`` [E] FP32 or None."""
    global _moe_combine_kernel
    if _moe_combine_kernel is None:
        _moe_combine_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_moe_combine",
            input_names=["z", "w", "inds", "gs", "inv_den"],
            output_names=["out"],
            source=_MOE_COMBINE_SOURCE,
        )
    rows, top, n = z.shape
    has_gs = global_scale is not None
    gs = global_scale if has_gs else mx.zeros((1,), mx.float32)
    (out,) = _moe_combine_kernel(
        inputs=[z, weights, inds, gs, mx.array([inv_den], dtype=mx.float32)],
        template=[("T", out_dtype), ("TOP", top), ("N", n), ("HAS_GS", int(has_gs))],
        grid=(n, rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[out_dtype],
    )
    return out


# Loop-controller message mix for decode/verify rows, in three kernels:
#   mix_z: z = bf16(unit_rms(x)), bf16(unit_rms(message)) stacked as
#     [N, 2, rows, H] (``rms_single_row`` arithmetic for an FP32 row);
#   loop_gemv(z, down) (BF16 z reads equal the FP32 copy);
#   loop_mix_up: low = bf16(sum of the two halves) on the fly, the gain/forget
#     projection (``loop_gemv`` arithmetic), tanh and the gated combine.
# Every step matches the stock ops bit for bit (tests/test_ops.py).
_MIX_Z_SOURCE = """
    constexpr int NR = 4;
    uint q = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint simd_lane_id = thread_index_in_simdgroup;
    uint simd_group_id = simdgroup_index_in_threadgroup;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    uint r = q % ROWS;
    uint half_ = (q / ROWS) % 2;
    uint i = q / (2 * ROWS);
    size_t flat = ((size_t)i * ROWS + r) * D;
    float xv[NR];
    float acc = 0;
    size_t xb = elem_to_loc(flat, x_shape, x_strides, x_ndim);
    size_t mb = elem_to_loc(flat, m_shape, m_strides, m_ndim);
    int64_t xs = x_strides[x_ndim - 1], ms = m_strides[m_ndim - 1];
    for (int k = 0; k < NR; k++) {
        int c = lid * NR + k;
        if (c < D) {
            xv[k] = half_ == 0 ? x[xb + c * xs] : m[mb + c * ms];
            acc += xv[k] * xv[k];
        }
    }
    acc = simd_sum(acc);
    if (simd_group_id == 0) {
        local_sums[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        local_sums[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(acc / D + 1e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = local_inv_mean[0];
    for (int k = 0; k < NR; k++) {
        int c = lid * NR + k;
        if (c < D) {
            z[(size_t)q * D + c] = static_cast<bfloat16_t>(xv[k] * inv);
        }
    }
"""
_mix_z_kernel = None


def mix_z(x, message):
    """``stack([unit_rms(x), unit_rms(message)], 1).astype(bf16)`` reshaped to
    [2N, rows, H]; ``x``/``message`` [N, B, L, H] FP32 (strided views ok)."""
    global _mix_z_kernel
    if _mix_z_kernel is None:
        _mix_z_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_mix_z",
            input_names=["x", "m"],
            output_names=["z"],
            source=_MIX_Z_SOURCE,
            ensure_row_contiguous=False,
        )
    n, b, length, d = x.shape
    rows = b * length
    tg = ((d + 3) // 4 + 31) // 32 * 32
    (z,) = _mix_z_kernel(
        inputs=[x, message],
        template=[("D", d), ("ROWS", rows)],
        grid=(tg * 2 * n * rows, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(2 * n, rows, d)],
        output_dtypes=[mx.bfloat16],
    )
    return z


_LOOP_MIX_UP_SOURCE = """
    uint t = thread_position_in_threadgroup.x;
    uint lane = t % CPT;
    uint j = t / CPT;
    uint col = threadgroup_position_in_grid.x * CPT + lane;
    uint bi = threadgroup_position_in_grid.y;
    const device bfloat16_t* wp = w + (size_t)bi * K * N2 + col;
    const device float* pa = parts + (size_t)(2 * bi) * ROWS * K;
    const device float* pb = parts + (size_t)(2 * bi + 1) * ROWS * K;
    // low = bf16(sum of the two halves), as the stock axis-1 sum + casts.
    threadgroup float low[ROWS * K];
    for (uint e = t; e < ROWS * K; e += 8 * CPT) {
        float s = 0.0f;
        s = pa[e] + s;
        s = pb[e] + s;
        low[e] = static_cast<float>(static_cast<bfloat16_t>(s));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ag[ROWS];
    float af[ROWS];
    for (int r = 0; r < ROWS; ++r) { ag[r] = 0.0f; af[r] = 0.0f; }
    for (int i0 = 0; i0 < K / 32; ++i0) {
        float wg[4], wf[4];
        for (int tm = 0; tm < 4; ++tm) {
            int k = i0 * 32 + j * 4 + tm;
            wg[tm] = static_cast<float>(wp[(size_t)k * N2]);
            wf[tm] = static_cast<float>(wp[(size_t)k * N2 + H]);
        }
        for (int tm = 0; tm < 4; ++tm) {
            int k = i0 * 32 + j * 4 + tm;
            for (int r = 0; r < ROWS; ++r) {
                float vc = low[r * K + k];
                ag[r] += vc * wg[tm];
                af[r] += vc * wf[tm];
            }
        }
    }
    threadgroup float red[2][8][ROWS][CPT];
    for (int r = 0; r < ROWS; ++r) { red[0][j][r][lane] = ag[r]; red[1][j][r][lane] = af[r]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int r = j; r < ROWS; r += 8) {
        float g2[2];
        for (int u = 0; u < 2; ++u) {
            float p0 = red[u][0][r][lane], p1 = red[u][1][r][lane], p2 = red[u][2][r][lane], p3 = red[u][3][r][lane];
            float p4 = red[u][4][r][lane], p5 = red[u][5][r][lane], p6 = red[u][6][r][lane], p7 = red[u][7][r][lane];
            float a0 = p0 + p4, a1 = p1 + p5, a2 = p2 + p6, a3 = p3 + p7;
            float b0 = a0 + a2, b1 = a1 + a3;
            g2[u] = b0 + b1;
        }
        float gain = metal::precise::tanh(g2[0]);
        float forget = metal::precise::tanh(g2[1]);
        size_t flat = ((size_t)bi * ROWS + r) * H;
        float xv = x[elem_to_loc(flat, x_shape, x_strides, x_ndim) + col * x_strides[x_ndim - 1]];
        float mv = m[elem_to_loc(flat, m_shape, m_strides, m_ndim) + col * m_strides[m_ndim - 1]];
        // Two products and a separate add, as the stock elementwise ops
        // round them (no fused multiply-add).
        {
#pragma METAL fp contract(off)
            float keep = (1.0f - forget) * xv;
            float add = gain * mv;
            out[flat + col] = keep + add;
        }
    }
"""
_loop_mix_up_kernel = None


def loop_mix_up(parts, w, x, message):
    """Second half of the loop-controller mix. ``parts`` [2N, rows, K] FP32
    (down projection of the x and message halves), ``w`` [N, K, 2H] BF16,
    ``x``/``message`` [N, B, L, H] FP32. Returns [N, B, L, H] FP32."""
    global _loop_mix_up_kernel
    if _loop_mix_up_kernel is None:
        _loop_mix_up_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_loop_mix_up",
            input_names=["parts", "w", "x", "m"],
            output_names=["out"],
            source=_LOOP_MIX_UP_SOURCE,
            ensure_row_contiguous=False,
        )
    n, b, length, h = x.shape
    rows = b * length
    k = parts.shape[-1]
    cpt = 32 if rows == 1 else 16  # fastest per width in a dependent-chain microbench
    (out,) = _loop_mix_up_kernel(
        inputs=[parts, w, x, message],
        template=[("K", k), ("H", h), ("N2", 2 * h), ("ROWS", rows), ("CPT", cpt)],
        grid=((h // cpt) * 8 * cpt, n, 1),
        threadgroup=(8 * cpt, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[mx.float32],
    )
    return out


# Prefill (L > 16 rows, batch 1) depthwise causal conv of ``model._static_conv``
# in one pass: per element the FP32 tap products summed in tap order, the
# weight-dtype rounding, the residual add and scale, the cast; rows >= L - (K-1)
# also write the next conv state (the FP32 input). No contraction, so every
# step rounds like the stock elementwise kernels.
_STATIC_CONV_SOURCE = """
#pragma METAL fp contract(off)
    uint c = thread_position_in_grid.x;
    uint i = thread_position_in_grid.y;
    if (c >= C) return;
    const int L = dims[0];
    const size_t xr = x_strides[1], xc = x_strides[2];
    float rr[K];
    for (int j = 0; j < K; ++j) {
        int p = int(i) + j;
        rr[j] = p < K - 1 ? state[(size_t)p * C + c]
                          : static_cast<float>(x[(size_t)(p - (K - 1)) * xr + c * xc]);
    }
    float o = taps[c] * rr[0];
    for (int j = 1; j < K; ++j) {
        float t = taps[(size_t)j * C + c] * rr[j];
        o = o + t;
    }
    float xf = rr[K - 1];
    float y = (static_cast<float>(static_cast<W>(o)) + xf) * scale[0];
    out[(size_t)i * C + c] = static_cast<T>(y);
    if (i == 0) {
        // Next state = the last K-1 rows of the window [state, x].
        for (int t = 0; t < K - 1; ++t) {
            int w = L + t;
            next_state[(size_t)t * C + c] = w < K - 1
                ? state[(size_t)w * C + c]
                : static_cast<float>(x[(size_t)(w - (K - 1)) * xr + c * xc]);
        }
    }
"""
_static_conv_kernel = None


def static_conv_prefill(x, taps, state, scale, wdtype):
    """``x`` [1, L, C] (any strides, any L >= 1), ``taps`` [K, C] FP32 (tap j at
    row j), ``state`` [1, K-1, C] FP32. Returns (out [1, L, C] x.dtype,
    next state [1, K-1, C] FP32)."""
    global _static_conv_kernel
    if _static_conv_kernel is None:
        _static_conv_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_static_conv_prefill",
            input_names=["x", "taps", "state", "scale", "dims"],
            output_names=["out", "next_state"],
            source=_STATIC_CONV_SOURCE,
            ensure_row_contiguous=False,
        )
    _, length, c = x.shape
    k = taps.shape[0]
    out, nxt = _static_conv_kernel(
        inputs=[x, taps, mx.contiguous(state), mx.array([scale], dtype=mx.float32),
                mx.array([length], dtype=mx.int32)],
        template=[("T", x.dtype), ("W", wdtype), ("C", c), ("K", k)],
        grid=(c, length, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, length, c), (1, k - 1, c)],
        output_dtypes=[x.dtype, mx.float32],
    )
    return out, nxt


# Prefill routed-expert combine straight from the expert-sorted outputs:
# ``out[r] = sum_j zs[inv[r*TOP+j]] * (w[r, j] * (gs[e[r, j]] * inv_den))``,
# the ops' rounding order (FP32 product, then sum over j in order), no
# contraction. Replaces the unsort gather, the scale gather and multiplies,
# the FP32 widening, the product, the sum and the cast.
_MOE_COMBINE_SORTED_SOURCE = """
#pragma METAL fp contract(off)
    uint n = thread_position_in_grid.x;
    uint r = thread_position_in_grid.y;
    if (n >= N) return;
    float total = 0.0f;
    for (int j = 0; j < TOP; ++j) {
        float wj = w[r * TOP + j];
        if (HAS_GS) {
            float s = gs[inds[r * TOP + j]] * inv_den[0];
            wj = wj * s;
        }
        uint src = inv[r * TOP + j];
        float p = static_cast<float>(zs[(size_t)src * N + n]) * wj;
        total = total + p;
    }
    out[(size_t)r * N + n] = static_cast<T>(total);
"""
_moe_combine_sorted_kernel = None


def moe_combine_sorted(zs, inv, weights, inds, global_scale, inv_den, out_dtype):
    """``zs`` [rows*top, N] expert-sorted outputs (row-contiguous), ``inv``
    [rows*top] the inverse permutation, ``weights``/``inds`` [rows, top]."""
    global _moe_combine_sorted_kernel
    if _moe_combine_sorted_kernel is None:
        _moe_combine_sorted_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_moe_combine_sorted",
            input_names=["zs", "inv", "w", "inds", "gs", "inv_den"],
            output_names=["out"],
            source=_MOE_COMBINE_SORTED_SOURCE,
        )
    rows, top = inds.shape
    n = zs.shape[-1]
    has_gs = global_scale is not None
    gs = global_scale if has_gs else mx.zeros((1,), mx.float32)
    (out,) = _moe_combine_sorted_kernel(
        inputs=[zs.reshape(rows * top, n), inv.astype(mx.uint32), weights, inds, gs,
                mx.array([inv_den], dtype=mx.float32)],
        template=[("T", out_dtype), ("TOP", top), ("N", n), ("HAS_GS", int(has_gs))],
        grid=(n, rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[out_dtype],
    )
    return out


# Single-row routing that also emits the expert-sorted pair order (what
# ``argsort(inds.flatten())`` and the sorted-expert gather produce for one row
# of distinct experts), so the grouped expert kernels need no argsort/gather.
_ROUTER_SORTED_SOURCE = _ROUTER_SOURCE.rstrip() + """
    if (lane == 0) {
        uint ord[K];
        for (uint k = 0; k < K; ++k) ord[k] = k;
        for (int a = 1; a < int(K); ++a) {
            uint v = ord[a];
            int b = a - 1;
            while (b >= 0 && chosen[ord[b]] > chosen[v]) { ord[b + 1] = ord[b]; --b; }
            ord[b + 1] = v;
        }
        for (uint k = 0; k < K; ++k) {
            order[row * K + k] = row * K + ord[k];
            sorted_e[row * K + k] = chosen[ord[k]];
        }
    }
"""
_router_sorted_kernel = None


def route_topk_sorted(logits, bias, top_k, route_scale):
    """``route_topk`` for one row plus (order, sorted experts), both [K] uint32."""
    global _router_sorted_kernel
    rows, experts = logits.shape
    if _router_sorted_kernel is None:
        _router_sorted_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_route_topk_sorted",
            input_names=["logits", "bias", "route_scale"],
            output_names=["inds", "weights", "order", "sorted_e"],
            source=_ROUTER_SORTED_SOURCE,
        )
    inds, weights, order, sorted_e = _router_sorted_kernel(
        inputs=[logits.astype(mx.float32), bias.astype(mx.float32),
                mx.array([route_scale], dtype=mx.float32)],
        template=[("E", experts), ("K", top_k)],
        grid=(rows * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, top_k), (rows, top_k), (rows * top_k,), (rows * top_k,)],
        output_dtypes=[mx.uint32, mx.float32, mx.uint32, mx.uint32],
    )
    return inds, weights, order, sorted_e


# Routed-expert activation in one kernel: the optional per-pair NVFP4 tensor
# scale as a BF16 multiply (``y * s.astype(bf16)``), then the softcapped SwiGLU
# of ``model._softcapped_swiglu`` on the interleaved gate/up columns, with MLX's
# own functors (precise tanh, sigmoid via precise exp) in the same order.
_SWIGLU_SOURCE = """
#pragma METAL fp contract(off)
    uint j = thread_position_in_grid.x;
    uint p = thread_position_in_grid.y;
    if (j >= I) return;
    float g = static_cast<float>(y[(size_t)p * 2 * I + 2 * j]);
    float u = static_cast<float>(y[(size_t)p * 2 * I + 2 * j + 1]);
    if (HAS_S) {
        float sv = gs[inds[p / SDIV]] * inv_den[0];
        float sb = static_cast<float>(static_cast<bfloat16_t>(sv));
        g = static_cast<float>(static_cast<T>(g * sb));
        u = static_cast<float>(static_cast<T>(u * sb));
    }
    const float gc = caps[0], uc = caps[1];
    float sg = 1.0f / (1.0f + metal::precise::exp(metal::abs(g)));
    sg = (g < 0.0f) ? sg : 1.0f - sg;
    float a = metal::precise::tanh(g / gc) * gc;
    a = a * sg;
    float b = metal::precise::tanh(u / uc) * uc;
    out[(size_t)p * I + j] = static_cast<T>(a * b);
"""
_swiglu_kernel = None


def scaled_swiglu(y, global_scale, inds, inv_den, caps, sdiv=1):
    """``y`` [P, 2I] BF16 rows (row-contiguous); the scale of pair p is
    ``global_scale[inds[p // sdiv]] * inv_den`` (``global_scale`` None: no
    scale); ``caps`` [2] FP32 (gate, up). Returns [P, I] in y's dtype."""
    global _swiglu_kernel
    if _swiglu_kernel is None:
        _swiglu_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_scaled_swiglu",
            input_names=["y", "gs", "inds", "inv_den", "caps"],
            output_names=["out"],
            source=_SWIGLU_SOURCE,
        )
    p, two_i = y.shape
    has_s = global_scale is not None
    zero = mx.zeros((1,), mx.uint32)
    (out,) = _swiglu_kernel(
        inputs=[y, global_scale if has_s else mx.zeros((1,), mx.float32),
                inds.reshape(-1).astype(mx.uint32) if has_s else zero,
                mx.array([inv_den], dtype=mx.float32), caps],
        template=[("T", y.dtype), ("I", two_i // 2), ("HAS_S", int(has_s)), ("SDIV", sdiv)],
        grid=(two_i // 2, p, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(p, two_i // 2)],
        output_dtypes=[y.dtype],
    )
    return out


# ---------------------------------------------------------------------------
# Megakernel slice 1 (OMLX_INKLING2_MK_MOE=1): the single-token MoE tail in two
# dispatches instead of six. Per element the arithmetic is the incumbent's:
#   ups   = nvfp4_gather_rows(w13) (exact variant) + scaled_swiglu;
#   downs = nvfp4_gather_rows(w2) (exact) + moe_combine + static conv (decode)
#           + ``(residual + conv).astype(T)``, and the next MLP conv state.
# Only the intermediate y / z / m round-trips through device memory go away.
# ---------------------------------------------------------------------------
_NVFP4_DOT_BODY = """
    // MLX fp_qmv traversal for NVFP4 of 4 rows (``_NVFP4_ROWS_SOURCE``, one pair).
    float result[RPS];
    for (int r = 0; r < RPS; ++r) result[r] = 0.0f;
    for (int k = 0; k < KD; k += 256) {
        if (k + (int)lane * 8 < KD) {
            float s[RPS];
            float wd[RPS][8];
            for (int r = 0; r < RPS; ++r) {
                const device ushort* w16 = (const device ushort*)(ws + r * KW * 4);
                for (int i = 0; i < 2; ++i) {
                    ushort q = w16[i];
                    wd[r][4 * i] = fp4v(q);
                    wd[r][4 * i + 1] = fp4v(q >> 4);
                    wd[r][4 * i + 2] = fp4v(q >> 8);
                    wd[r][4 * i + 3] = fp4v(q >> 12);
                }
                uchar sb = sc[r * KG];
                ushort v = sb & 127;
                ushort u = (v << 7) | (((v + 1) >> 7) << 14) | (((ushort)((sb >> 7) & 1)) << 15);
                s[r] = float(as_type<half>(u) * 256.0h);
            }
            float xt[8];
            for (int i = 0; i < 8; ++i) {
                xt[i] = float(xs[k + i]);
            }
            for (int r = 0; r < RPS; ++r) {
                float accum = 0.0f;
                for (int i = 0; i < 2; ++i) {
                    accum += (xt[4 * i] * wd[r][4 * i] + xt[4 * i + 1] * wd[r][4 * i + 1] +
                              xt[4 * i + 2] * wd[r][4 * i + 2] + xt[4 * i + 3] * wd[r][4 * i + 3]);
                }
                result[r] += s[r] * accum;
            }
        }
        ws += 128;
        sc += 16;
    }
    T z[RPS];
    for (int r = 0; r < RPS; ++r) z[r] = static_cast<T>(simd_sum(result[r]));
"""

# grid (64 * (2I / 8), TOP), threadgroup 64: simdgroup g of threadgroup x owns
# w13 rows x*8 + g*4 .. +3 = (gate, up) of intermediate columns x*4 + g*2, +1.
_MK_UPS_SOURCE = """
#pragma METAL fp contract(off)
    uint p = threadgroup_position_in_grid.y;
    uint e = inds[p];
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    constexpr int KD = H;
    constexpr int N = 2 * I;
    constexpr int KW = KD / 8;
    constexpr int KG = KD / 16;
    int out_row = threadgroup_position_in_grid.x * (2 * RPS) + simd_gid * RPS;
    const device uint8_t* ws =
        (const device uint8_t*)(w + ((size_t)e * N + out_row) * KW) + lane * 4;
    const device uint8_t* sc = scales + ((size_t)e * N + out_row) * KG + lane / 2;
    const device T* xs = x + lane * 8;
""" + _NVFP4_DOT_BODY + """
    if (lane < RPS / 2) {
        float g = static_cast<float>(z[2 * lane]);
        float u = static_cast<float>(z[2 * lane + 1]);
        float sv = gs[e] * inv_den[0];
        float sb = static_cast<float>(static_cast<bfloat16_t>(sv));
        g = static_cast<float>(static_cast<T>(g * sb));
        u = static_cast<float>(static_cast<T>(u * sb));
        const float gc = caps[0], uc = caps[1];
        float sg = 1.0f / (1.0f + metal::precise::exp(metal::abs(g)));
        sg = (g < 0.0f) ? sg : 1.0f - sg;
        float a = metal::precise::tanh(g / gc) * gc;
        a = a * sg;
        float b = metal::precise::tanh(u / uc) * uc;
        act[(size_t)p * I + out_row / 2 + lane] = static_cast<T>(a * b);
    }
"""

# grid (32 * TOP, H / 4), threadgroup 32 * TOP: simdgroup j = pick j, all four
# output rows y*4 .. +3; then 4 lanes of simdgroup 0 combine the picks in order,
# run the decode conv tail and the residual add for those 4 columns.
_MK_DOWNS_SOURCE = """
#pragma METAL fp contract(off)
    uint j = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint e = inds[j];
    constexpr int KD = I;
    constexpr int N = H;
    constexpr int KW = KD / 8;
    constexpr int KG = KD / 16;
    int out_row = threadgroup_position_in_grid.y * RPS;
    const device uint8_t* ws =
        (const device uint8_t*)(w + ((size_t)e * N + out_row) * KW) + lane * 4;
    const device uint8_t* sc = scales + ((size_t)e * N + out_row) * KG + lane / 2;
    const device T* xs = act + (size_t)j * I + lane * 8;
""" + _NVFP4_DOT_BODY + """
    threadgroup float tz[TOP][RPS];
    if (lane == 0) {
        for (int r = 0; r < RPS; ++r) tz[j][r] = static_cast<float>(z[r]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (j == 0 && lane < RPS) {
        uint c = out_row + lane;
        float total = 0.0f;
        for (int t = 0; t < TOP; ++t) {
            float wj = wts[t];
            float s = gs[inds[t]] * inv_den[0];
            wj = wj * s;
            if (COMBINE_FMA) {
                total = metal::fma(tz[t][lane], wj, total);
            } else {
                float pr = tz[t][lane] * wj;
                total = pr + total;
            }
        }
        T xm = static_cast<T>(total);
        // Decode static conv (static_conv_prefill at L = 1).
        float rr[KT];
        for (int t = 0; t < KT - 1; ++t) rr[t] = state[(size_t)t * H + c];
        rr[KT - 1] = static_cast<float>(xm);
        float o = taps[c] * rr[0];
        for (int t = 1; t < KT; ++t) {
            float tt = taps[(size_t)t * H + c] * rr[t];
            o = o + tt;
        }
        float y = (static_cast<float>(static_cast<W>(o)) + rr[KT - 1]) * cscale[0];
        T m = static_cast<T>(y);
        hout[c] = static_cast<T>(residual[c] + static_cast<float>(m));
        for (int t = 0; t < KT - 1; ++t) next_state[(size_t)t * H + c] = rr[t + 1];
    }
"""
_mk_ups_kernel = None
_mk_downs_kernel = None
# 0 = plain multiply then add (bit-equal to the incumbent moe_combine on 16 chained
# r1 steps, tools/mk_equal.py); 1 = explicit fma (differs).
MK_COMBINE_FMA = 0
# Output rows per simdgroup in ups / downs (per-row arithmetic unchanged).
MK_UPS_RPS = 2
MK_DOWNS_RPS = 2


def mk_moe_ups(x, w, scales, inds, gs, inv_den, caps):
    """x [1, H] bf16, w13 [E, 2I, H/8] u32, inds [1, TOP] -> act [TOP, I]."""
    global _mk_ups_kernel
    if _mk_ups_kernel is None:
        _mk_ups_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_mk_ups",
            input_names=["x", "w", "scales", "inds", "gs", "inv_den", "caps"],
            output_names=["act"],
            header=_NVFP4_ROWS_HEADER,
            source=_MK_UPS_SOURCE,
        )
    top = inds.size
    n2 = w.shape[1]
    h = x.shape[-1]
    (act,) = _mk_ups_kernel(
        inputs=[x, w, scales, inds.reshape(-1), gs, mx.array([inv_den], dtype=mx.float32), caps],
        template=[("T", x.dtype), ("H", h), ("I", n2 // 2), ("RPS", MK_UPS_RPS)],
        grid=(64 * (n2 // (2 * MK_UPS_RPS)), top, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(top, n2 // 2)],
        output_dtypes=[x.dtype],
    )
    return act


def mk_moe_downs(act, w, scales, inds, weights, gs, inv_den, residual, state, taps, cscale, wdtype):
    """act [TOP, I], w2 [E, H, I/8], weights [1, TOP] f32, residual [1, 1, H]
    f32, state [1, K-1, H] f32, taps [K, H] f32. Returns (h [1, 1, H] act.dtype,
    next state [1, K-1, H] f32)."""
    global _mk_downs_kernel
    if _mk_downs_kernel is None:
        _mk_downs_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_mk_downs",
            input_names=["act", "w", "scales", "inds", "wts", "gs", "inv_den",
                         "residual", "state", "taps", "cscale"],
            output_names=["hout", "next_state"],
            header=_NVFP4_ROWS_HEADER,
            source=_MK_DOWNS_SOURCE,
        )
    top, i = act.shape
    h = w.shape[1]
    kt = taps.shape[0]
    hout, nxt = _mk_downs_kernel(
        inputs=[act, w, scales, inds.reshape(-1), weights.reshape(-1), gs,
                mx.array([inv_den], dtype=mx.float32), residual.reshape(-1),
                mx.contiguous(state), taps, mx.array([cscale], dtype=mx.float32)],
        template=[("T", act.dtype), ("W", wdtype), ("H", h), ("I", i), ("TOP", top),
                  ("KT", kt), ("COMBINE_FMA", MK_COMBINE_FMA), ("RPS", MK_DOWNS_RPS)],
        grid=(32 * top, h // MK_DOWNS_RPS, 1),
        threadgroup=(32 * top, 1, 1),
        output_shapes=[(1, 1, h), (1, kt - 1, h)],
        output_dtypes=[act.dtype, mx.float32],
    )
    return hout, nxt


# ---------------------------------------------------------------------------
# Megakernel slice 2 (OMLX_INKLING2_MK_OPROJ=1): single-token attention output
# in one dispatch: x = bf16(attn * sigmoid(gate)) on load (sigmoid precomputed
# off the critical path), MLX 0.32.4 ``qmv_fast_impl`` for 8-bit affine
# (2 simdgroups x 4 rows, 8 values per lane, ``scale * accum + sum * bias``,
# simd_sum, cast), then the static conv (decode) and the residual in FP32.
# ---------------------------------------------------------------------------
_MK_OPROJ_SOURCE = """
#pragma METAL fp contract(off)
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    constexpr int VPT = 8;
    constexpr int BLOCK = VPT * 32;
    constexpr int KW = K;            // bytes per weight row (8-bit)
    constexpr int KG = K / GS;
    constexpr int SSTEP = GS / VPT;
    int out_row = threadgroup_position_in_grid.y * 8 + simd_gid * 4;
    const device uint8_t* ws = (const device uint8_t*)w + (size_t)out_row * KW + simd_lid * VPT;
    const device T* sl0 = scales + (size_t)out_row * KG + simd_lid / SSTEP;
    const device T* bl0 = biases + (size_t)out_row * KG + simd_lid / SSTEP;
    const device T* xa = attn + simd_lid * VPT;
    const device float* xg = sg + simd_lid * VPT;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < K; k += BLOCK) {
        float xt[VPT];
        float sum = 0.0f;
        for (int i = 0; i < VPT; ++i) {
            T xv = static_cast<T>(static_cast<float>(xa[k + i]) * xg[k + i]);
            sum += static_cast<float>(xv);
            xt[i] = static_cast<float>(xv);
        }
        for (int r = 0; r < 4; ++r) {
            const device uint8_t* wl = ws + r * KW;
            float s = static_cast<float>(sl0[r * KG]);
            float b = static_cast<float>(bl0[r * KG]);
            float accum = 0.0f;
            if (VAR & 2) {
                for (int i = 0; i < VPT; ++i) accum = metal::fma(xt[i], static_cast<float>(wl[i]), accum);
            } else {
                for (int i = 0; i < VPT; ++i) accum += xt[i] * wl[i];
            }
            if (VAR & 1) {
                result[r] += metal::fma(s, accum, sum * b);
            } else if (VAR & 4) {
                result[r] = metal::fma(sum, b, s * accum) + result[r];
            } else {
                result[r] += s * accum + sum * b;
            }
        }
        ws += BLOCK;
        sl0 += BLOCK / GS;
        bl0 += BLOCK / GS;
    }
    for (int r = 0; r < 4; ++r) {
        float v = simd_sum(result[r]);
        if (simd_lid == 0) {
            uint c = out_row + r;
            T a = static_cast<T>(v);
            float rr[KT];
            for (int t = 0; t < KT - 1; ++t) rr[t] = state[(size_t)t * N + c];
            rr[KT - 1] = static_cast<float>(a);
            float o = taps[c] * rr[0];
            for (int t = 1; t < KT; ++t) {
                float tt = taps[(size_t)t * N + c] * rr[t];
                o = o + tt;
            }
            float y = (static_cast<float>(static_cast<W>(o)) + rr[KT - 1]) * cscale[0];
            T m = static_cast<T>(y);
            residual[c] = BOUND ? static_cast<float>(m)
                                : static_cast<float>(h[c]) + static_cast<float>(m);
            for (int t = 0; t < KT - 1; ++t) next_state[(size_t)t * N + c] = rr[t + 1];
        }
    }
"""
_mk_oproj_kernel = None
# 1 = fma(scale, accum, sum * bias), as MLX 0.32.4's precompiled qmv rounds (tools/mk_unit.py: 20/20 equal; 0 and 4 differ)
MK_OPROJ_VAR = 1


def mk_oproj(attn, sg, w, scales, biases, group_size, state, taps, cscale, wdtype, h, boundary):
    """attn [.., K] bf16 (row-contiguous; the SDPA output of one row), sg [.., K]
    FP32 sigmoid(gate), 8-bit affine o_proj (w [N, K/4] u32, scales/biases
    [N, K/GS]), state [1, KT-1, N] FP32, taps [KT, N] FP32, h [1, 1, N].
    Returns (residual [1, 1, N] FP32, next state [1, KT-1, N] FP32)."""
    global _mk_oproj_kernel
    if _mk_oproj_kernel is None:
        _mk_oproj_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_mk_oproj",
            input_names=["attn", "sg", "w", "scales", "biases", "state", "taps", "cscale", "h"],
            output_names=["residual", "next_state"],
            source=_MK_OPROJ_SOURCE,
        )
    n = w.shape[0]
    k = attn.size
    kt = taps.shape[0]
    residual, nxt = _mk_oproj_kernel(
        inputs=[attn.reshape(-1), sg.reshape(-1), w, scales, biases, mx.contiguous(state), taps,
                mx.array([cscale], dtype=mx.float32), h.reshape(-1)],
        template=[("T", attn.dtype), ("W", wdtype), ("K", k), ("N", n), ("GS", group_size),
                  ("KT", kt), ("BOUND", int(boundary)), ("VAR", MK_OPROJ_VAR)],
        grid=(64, n // 8, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(1, 1, n), (1, kt - 1, n)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return residual, nxt


# ---------------------------------------------------------------------------
# Megakernel slice 4 (OMLX_INKLING2_MK_ROUTER=1, with MK_MOE): the MLP input
# norm, its BF16 rounding and the FP32 router GEMV in one dispatch.
#   x = bf16(w * (y * inv)) with ``rms_single_row``'s inv (``_RAW_RMS_PROLOGUE``),
#   logits = MLX ``gemv`` (BM 1, BN 8, SM 1, SN 32, TM 4, TN 4: matmul.cpp's
#   pick for a [128, 2304] row-major matrix) over float(x).
# The router rows a thread reads are loaded before the norm (they do not depend
# on it); threadgroup 0 also writes x for the experts.
# ---------------------------------------------------------------------------
_MK_ROUTER_SOURCE = """
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    constexpr int TM = 4, TN = 4, SN = 32, BN = 8;
    constexpr int BLOCKN = BN * SN * TN;
    constexpr int NIT = H / BLOCKN;
    constexpr int LEFT = H - NIT * BLOCKN;
    constexpr int NB = NIT + (LEFT > 0 ? 1 : 0);
    int out_row = threadgroup_position_in_grid.x * TM;
    const device float* mat = rw + (size_t)out_row * H;
    float wv[NB][TM][TN];
    for (int i = 0; i < NB; ++i) {
        int bn = (int)(sg * SN + lane) * TN + i * BLOCKN;
        for (int tm = 0; tm < TM; ++tm)
            for (int tn = 0; tn < TN; ++tn)
                wv[i][tm][tn] = (bn + tn < H) ? mat[(size_t)tm * H + bn + tn] : 0.0f;
    }
    const device float* p = y;
    const size_t base = 0;
""" + _RAW_RMS_PROLOGUE + """
    float result[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < NB; ++i) {
        int bn = (int)(sg * SN + lane) * TN + i * BLOCKN;
        float vc[TN];
        for (int tn = 0; tn < TN; ++tn) {
            int c = bn + tn;
            vc[tn] = (c < H) ? static_cast<float>(static_cast<bfloat16_t>(nw[c] * (y[c] * inv))) : 0.0f;
        }
        for (int tm = 0; tm < TM; ++tm)
            for (int tn = 0; tn < TN; ++tn) {
                if (FMA) result[tm] = metal::fma(wv[i][tm][tn], vc[tn], result[tm]);
                else { float pr = wv[i][tm][tn] * vc[tn]; result[tm] = result[tm] + pr; }
            }
    }
    for (int tm = 0; tm < TM; ++tm)
        for (ushort sn = SN / 2; sn >= 1; sn >>= 1)
            result[tm] += simd_shuffle_down(result[tm], sn);
    threadgroup float tg[BN][TM];
    if (lane == 0)
        for (int tm = 0; tm < TM; ++tm) tg[sg][tm] = result[tm];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        for (int tm = 0; tm < TM; ++tm) {
            float r = tg[0][tm];
            for (int s = 1; s < BN; ++s) r += tg[s][tm];
            logits[out_row + tm] = r;
        }
    }
    if (threadgroup_position_in_grid.x == 0) {
        for (uint c = tid; c < H; c += TG)
            xout[c] = static_cast<bfloat16_t>(nw[c] * (y[c] * inv));
    }
"""
_mk_router_kernel = None
# 1 = fma accumulation: bit-equal to MLX's precompiled fp32 gemv (tools/mk_unit.py 20/20; plain mul+add 0/20)
MK_ROUTER_FMA = 1


def mk_router(y, norm_w, router_w):
    """y [.., H] FP32 MLP input before its norm, norm_w [H], router_w [E, H] FP32.
    Returns (x [1, H] bf16 = rms_norm(y, w).astype(bf16), logits [1, E] FP32)."""
    global _mk_router_kernel
    if _mk_router_kernel is None:
        _mk_router_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_mk_router",
            input_names=["y", "nw", "rw"],
            output_names=["xout", "logits"],
            source=_MK_ROUTER_SOURCE,
        )
    h = y.shape[-1]
    e = router_w.shape[0]
    x, logits = _mk_router_kernel(
        inputs=[y.reshape(-1), norm_w.astype(mx.float32), router_w],
        template=[("H", h), ("TG", 256), ("FMA", MK_ROUTER_FMA)],
        grid=(256 * (e // 4), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, h), (1, e)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )
    return x, logits


# ---------------------------------------------------------------------------
# Megakernel slice 6 (OMLX_INKLING2_MK_PREP=1): a sliding visit's single-token
# projection epilogue in one dispatch (MLX-Serve's "prep"). One simdgroup per
# task (grid y): q heads (L2 clip with q_scale), k heads (static conv + L2
# clip), v heads (static conv + unit RMS + bf16), and 256-wide chunks of each
# q head's relative bias (r L2 clip, then ``r @ rel_proj`` as MLX's GEMM rounds
# it: sequential fp32 fma over the 16 taps), written distance-reversed into a
# fixed [H, 1024] buffer so the decode mask is a slice view. Arithmetic per
# element = fused_l2_clip / static_conv_prefill / rms_norm / the matmul.
# ---------------------------------------------------------------------------
_MK_PREP_SOURCE = """
#pragma METAL fp contract(off)
    uint task = threadgroup_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    const float tq = thr[0];
    const float tr = thr[1];
    if (task < HQ) {
        uint h = task;
        float v[4];
        float total = 0.0f;
        for (int j = 0; j < 4; ++j) {
            uint i = h * D + lane * 4 + j;
            v[j] = static_cast<float>(src[Q0 + i]) * static_cast<float>(qscale[i]);
            if (CLIPFMA) total = metal::fma(v[j], v[j], total);
            else { float sq = v[j] * v[j]; total = sq + total; }
        }
        total = simd_sum(total);
        float m = metal::min(tq / (metal::precise::sqrt(total) + 1e-6f), 1.0f);
        for (int j = 0; j < 4; ++j) qout[h * D + lane * 4 + j] = static_cast<T>(v[j] * m);
        return;
    }
    if (task < HQ + 2 * HKV) {
        bool isk = task < HQ + HKV;
        uint h = isk ? task - HQ : task - HQ - HKV;
        const device float* st = isk ? kst : vst;
        const device float* tp = isk ? ktaps : vtaps;
        uint off = isk ? K0 : V0;
        float cv[4];
        for (int j = 0; j < 4; ++j) {
            uint c = h * D + lane * 4 + j;
            float rr[KT];
            for (int t = 0; t < KT - 1; ++t) rr[t] = st[(size_t)t * C + c];
            rr[KT - 1] = static_cast<float>(src[off + c]);
            float o = tp[c] * rr[0];
            for (int t = 1; t < KT; ++t) {
                float tt = tp[(size_t)t * C + c] * rr[t];
                o = o + tt;
            }
            float y = (static_cast<float>(static_cast<W>(o)) + rr[KT - 1]) * cscale[0];
            cv[j] = static_cast<float>(static_cast<T>(y));
            if (isk) for (int t = 0; t < KT - 1; ++t) knext[(size_t)t * C + c] = rr[t + 1];
            else for (int t = 0; t < KT - 1; ++t) vnext[(size_t)t * C + c] = rr[t + 1];
        }
        if (isk) {
            float total = 0.0f;
            for (int j = 0; j < 4; ++j) {
                if (CLIPFMA) total = metal::fma(cv[j], cv[j], total);
                else { float sq = cv[j] * cv[j]; total = sq + total; }
            }
            total = simd_sum(total);
            float m = metal::min(tq / (metal::precise::sqrt(total) + 1e-6f), 1.0f);
            for (int j = 0; j < 4; ++j) kout[h * D + lane * 4 + j] = static_cast<T>(cv[j] * m);
        } else {
            float racc = 0.0f;
            for (int j = 0; j < 4; ++j) {
                if (RMSFMA) racc = metal::fma(cv[j], cv[j], racc);
                else { float sq = cv[j] * cv[j]; racc = racc + sq; }
            }
            racc = simd_sum(racc);
            float tot = simd_sum(lane == 0 ? racc : 0.0f);
            float inv = metal::precise::rsqrt(tot / D + 1e-6f);
            for (int j = 0; j < 4; ++j) vout[h * D + lane * 4 + j] = static_cast<T>(cv[j] * inv);
        }
        return;
    }
    uint idx = task - HQ - 2 * HKV;
    uint h = idx / RCH;
    uint chunk = idx % RCH;
    float rv = 0.0f;
    if (lane < DR) rv = static_cast<float>(src[R0 + h * DR + lane]) * static_cast<float>(rscale[h * DR + lane]);
    float sq = rv * rv;
    float total = 0.0f;
    for (int i = 0; i < DR; ++i) {
        float si = simd_shuffle(sq, (ushort)i);
        total = si + total;
    }
    float m = metal::min(tr / (metal::precise::sqrt(total) + 1e-6f), 1.0f);
    float rc = static_cast<float>(static_cast<T>(rv * m));
    for (int it = 0; it < 8; ++it) {
        uint d = chunk * 256 + it * 32 + lane;
        float acc = 0.0f;
        for (int k = 0; k < DR; ++k) {
            float rk = simd_shuffle(rc, (ushort)k);
            acc = metal::fma(rk, static_cast<float>(relp[(size_t)k * NREL + d]), acc);
        }
        if (d < NREL) relrev[h * NREL + (NREL - 1 - d)] = static_cast<T>(acc);
    }
"""
_mk_prep_kernel = None
MK_PREP_CLIPFMA = 0
MK_PREP_RMSFMA = 0


def mk_prep(src, offs, qscale, rscale, thr, ktaps, vtaps, kst, vst, cscale, wdtype, relp, hq, hkv, d, dr):
    """src: flat bf16 projection output(s); offs = (Q0, R0, K0, V0) element
    offsets of q, r, k, v; relp [DR, NREL] bf16. Returns (q [1,HQ,1,D], k, v
    [1,HKV,1,D], relrev [1,HQ,1,NREL] bf16, k/v next states [1,KT-1,C] f32)."""
    global _mk_prep_kernel
    if _mk_prep_kernel is None:
        _mk_prep_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_mk_prep",
            input_names=["src", "qscale", "rscale", "thr", "ktaps", "vtaps", "kst", "vst", "cscale", "relp"],
            output_names=["qout", "kout", "vout", "relrev", "knext", "vnext"],
            source=_MK_PREP_SOURCE,
        )
    nrel = relp.shape[1]
    rch = (nrel + 255) // 256
    c = hkv * d
    kt = ktaps.shape[0]
    q0, r0, k0, v0 = offs
    outs = _mk_prep_kernel(
        inputs=[src.reshape(-1), qscale.reshape(-1).astype(mx.float32), rscale.reshape(-1).astype(mx.float32),
                thr, ktaps, vtaps, mx.contiguous(kst), mx.contiguous(vst),
                mx.array([cscale], dtype=mx.float32), relp],
        template=[("T", src.dtype), ("W", wdtype), ("HQ", hq), ("HKV", hkv), ("D", d), ("DR", dr),
                  ("C", c), ("KT", kt), ("NREL", nrel), ("RCH", rch),
                  ("Q0", q0), ("R0", r0), ("K0", k0), ("V0", v0),
                  ("CLIPFMA", MK_PREP_CLIPFMA), ("RMSFMA", MK_PREP_RMSFMA)],
        grid=(32, hq + 2 * hkv + hq * rch, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, hq, 1, d), (1, hkv, 1, d), (1, hkv, 1, d), (1, hq, 1, nrel),
                       (1, kt - 1, c), (1, kt - 1, c)],
        output_dtypes=[src.dtype] * 4 + [mx.float32, mx.float32],
    )
    return tuple(outs)


# Megakernel slice 9 (OMLX_INKLING2_MK_PREP=1 on full-attention visits): the
# global layer's epilogue: q clip (q_scale) x tau; k/v dynamic conv (static tap
# + projected per-head tap, ``model._dynamic_conv`` order), k clip, v unit RMS;
# r clip x tau and the reversed relative bias. Separate q/k/v sources.
_MK_PREP_FULL_SOURCE = """
#pragma METAL fp contract(off)
    uint task = threadgroup_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    const float tq = thr[0];
    const float tr = thr[1];
    const float ta = tau[0];
    if (task < HQ) {
        uint h = task;
        float v[4];
        float total = 0.0f;
        for (int j = 0; j < 4; ++j) {
            uint i = h * D + lane * 4 + j;
            v[j] = static_cast<float>(qsrc[i]) * static_cast<float>(qscale[i]);
            if (CLIPFMA) total = metal::fma(v[j], v[j], total);
            else { float sq = v[j] * v[j]; total = sq + total; }
        }
        total = simd_sum(total);
        float m = metal::min(tq / (metal::precise::sqrt(total) + 1e-6f), 1.0f);
        for (int j = 0; j < 4; ++j) {
            float qc = static_cast<float>(static_cast<T>(v[j] * m));
            qout[h * D + lane * 4 + j] = static_cast<T>(qc * ta);
        }
        return;
    }
    if (task < HQ + 2 * HKV) {
        bool isk = task < HQ + HKV;
        uint h = isk ? task - HQ : task - HQ - HKV;
        const device T* src = isk ? ksrc : vsrc;
        const device float* st = isk ? kst : vst;
        const device float* tp = isk ? ktaps : vtaps;
        float dyn[KT];
        for (int lag = 0; lag < KT; ++lag) dyn[lag] = static_cast<float>(src[C + h * KT + lag]);
        float cv[4];
        for (int j = 0; j < 4; ++j) {
            uint c = h * D + lane * 4 + j;
            float rr[KT];
            for (int t = 0; t < KT - 1; ++t) rr[t] = st[(size_t)t * C + c];
            float xf = static_cast<float>(src[c]);
            rr[KT - 1] = xf;
            float out = 0.0f;
            for (int lag = 0; lag < KT; ++lag) {
                float tap = tp[(size_t)lag * C + c] + dyn[lag];
                float wv = rr[KT - 1 - lag];
                if (lag == 0) out = wv * tap;
                else if (DFMA) out = metal::fma(wv, tap, out);
                else { float term = wv * tap; out = out + term; }
            }
            float y = (out + xf) * cscale[0];
            cv[j] = static_cast<float>(static_cast<T>(y));
            if (isk) for (int t = 0; t < KT - 1; ++t) knext[(size_t)t * C + c] = rr[t + 1];
            else for (int t = 0; t < KT - 1; ++t) vnext[(size_t)t * C + c] = rr[t + 1];
        }
        if (isk) {
            float total = 0.0f;
            for (int j = 0; j < 4; ++j) {
                if (CLIPFMA) total = metal::fma(cv[j], cv[j], total);
                else { float sq = cv[j] * cv[j]; total = sq + total; }
            }
            total = simd_sum(total);
            float m = metal::min(tq / (metal::precise::sqrt(total) + 1e-6f), 1.0f);
            for (int j = 0; j < 4; ++j) kout[h * D + lane * 4 + j] = static_cast<T>(cv[j] * m);
        } else {
            float racc = 0.0f;
            for (int j = 0; j < 4; ++j) {
                if (RMSFMA) racc = metal::fma(cv[j], cv[j], racc);
                else { float sq = cv[j] * cv[j]; racc = racc + sq; }
            }
            racc = simd_sum(racc);
            float tot = simd_sum(lane == 0 ? racc : 0.0f);
            float inv = metal::precise::rsqrt(tot / D + 1e-6f);
            for (int j = 0; j < 4; ++j) vout[h * D + lane * 4 + j] = static_cast<T>(cv[j] * inv);
        }
        return;
    }
    uint idx = task - HQ - 2 * HKV;
    uint h = idx / RCH;
    uint chunk = idx % RCH;
    float rv = 0.0f;
    if (lane < DR) rv = static_cast<float>(qsrc[R0 + h * DR + lane]) * static_cast<float>(rscale[h * DR + lane]);
    float sq = rv * rv;
    float total = 0.0f;
    for (int i = 0; i < DR; ++i) {
        float si = simd_shuffle(sq, (ushort)i);
        total = si + total;
    }
    float m = metal::min(tr / (metal::precise::sqrt(total) + 1e-6f), 1.0f);
    float rc = static_cast<float>(static_cast<T>(rv * m));
    rc = static_cast<float>(static_cast<T>(rc * ta));
    for (int it = 0; it < 8; ++it) {
        uint d = chunk * 256 + it * 32 + lane;
        float acc = 0.0f;
        for (int k = 0; k < DR; ++k) {
            float rk = simd_shuffle(rc, (ushort)k);
            acc = metal::fma(rk, static_cast<float>(relp[(size_t)k * NREL + d]), acc);
        }
        if (d < NREL) relrev[h * NREL + (NREL - 1 - d)] = static_cast<T>(acc);
    }
"""
_mk_prep_full_kernel = None
MK_PREP_DFMA = 0


def mk_prep_full(qsrc, ksrc, vsrc, r0, tau, qscale, rscale, thr, ktaps, vtaps, kst, vst, cscale, relp, hq, hkv, d, dr):
    """Full-attention visit epilogue. qsrc [.., q|r|gate|sink] (r at ``r0``),
    ksrc/vsrc [.., C + HKV*KT] (values then per-head dynamic taps), tau [1] FP32.
    Returns (q, k, v, relrev, k next, v next) as ``mk_prep``."""
    global _mk_prep_full_kernel
    if _mk_prep_full_kernel is None:
        _mk_prep_full_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_mk_prep_full",
            input_names=["qsrc", "ksrc", "vsrc", "tau", "qscale", "rscale", "thr", "ktaps", "vtaps", "kst", "vst", "cscale", "relp"],
            output_names=["qout", "kout", "vout", "relrev", "knext", "vnext"],
            source=_MK_PREP_FULL_SOURCE,
        )
    nrel = relp.shape[1]
    rch = (nrel + 255) // 256
    c = hkv * d
    kt = ktaps.shape[0]
    outs = _mk_prep_full_kernel(
        inputs=[qsrc.reshape(-1), ksrc.reshape(-1), vsrc.reshape(-1), tau.reshape(-1).astype(mx.float32),
                qscale.reshape(-1).astype(mx.float32), rscale.reshape(-1).astype(mx.float32),
                thr, ktaps, vtaps, mx.contiguous(kst), mx.contiguous(vst),
                mx.array([cscale], dtype=mx.float32), relp],
        template=[("T", qsrc.dtype), ("HQ", hq), ("HKV", hkv), ("D", d), ("DR", dr),
                  ("C", c), ("KT", kt), ("NREL", nrel), ("RCH", rch), ("R0", r0),
                  ("CLIPFMA", MK_PREP_CLIPFMA), ("RMSFMA", MK_PREP_RMSFMA), ("DFMA", MK_PREP_DFMA)],
        grid=(32, hq + 2 * hkv + hq * rch, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, hq, 1, d), (1, hkv, 1, d), (1, hkv, 1, d), (1, hq, 1, nrel),
                       (1, kt - 1, c), (1, kt - 1, c)],
        output_dtypes=[qsrc.dtype] * 4 + [mx.float32, mx.float32],
    )
    return tuple(outs)


# Megakernel slice 10 (OMLX_INKLING2_MK_NORM3=1, with MK_PREP): a looped
# visit's three attention-input norms + bf16 casts + stacking in one launch
# (one threadgroup per row, ``rms_single_row`` order as ``_RAW_RMS_PROLOGUE``).
_MK_NORM_ROWS_SOURCE = """
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint row = threadgroup_position_in_grid.x;
    const device float* p = y;
    const size_t base = (size_t)row * H;
""" + _RAW_RMS_PROLOGUE + """
    for (uint c = tid; c < H; c += TG)
        xb[base + c] = static_cast<bfloat16_t>(nw[c] * (y[base + c] * inv));
"""
_mk_norm_rows_kernel = None


def mk_norm_rows(y, norm_w):
    """``rms_norm(y[i], w, 1e-6).astype(bf16)`` for every FP32 row of ``y``
    [R, H], stacked [R, H]."""
    global _mk_norm_rows_kernel
    if _mk_norm_rows_kernel is None:
        _mk_norm_rows_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_mk_norm_rows",
            input_names=["y", "nw"],
            output_names=["xb"],
            source=_MK_NORM_ROWS_SOURCE,
        )
    rows, h = y.shape
    (out,) = _mk_norm_rows_kernel(
        inputs=[y, norm_w.astype(mx.float32)],
        template=[("H", h), ("TG", 256)],
        grid=(256 * rows, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, h)],
        output_dtypes=[mx.bfloat16],
    )
    return out


# ---------------------------------------------------------------------------
# Megakernel slice 11 (OMLX_INKLING2_MK_PREP_ROWS=1): the sliding-visit prep at
# verify widths (2..16 rows): per (row, head) tasks; static conv over the
# window [state, x] (written out whole for the verify rollback), clips, v
# unit RMS, rel bias in its standard [L, H, R] layout. q/k/v sources may be
# different arrays (looped visits); each has its own row stride and offset.
# ---------------------------------------------------------------------------
_MK_PREP_ROWS_SOURCE = """
#pragma METAL fp contract(off)
    uint task = threadgroup_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    const float tq = thr[0];
    const float tr = thr[1];
    constexpr uint NQ = L * HQ, NK = L * HKV;
    if (task < NQ) {
        uint l = task / HQ, h = task % HQ;
        float v[4];
        float total = 0.0f;
        for (int j = 0; j < 4; ++j) {
            uint i = h * D + lane * 4 + j;
            v[j] = static_cast<float>(qs[Q0 + l * QST + i]) * static_cast<float>(qscale[i]);
            float sq = v[j] * v[j]; total = sq + total;
        }
        total = simd_sum(total);
        float m = metal::min(tq / (metal::precise::sqrt(total) + 1e-6f), 1.0f);
        for (int j = 0; j < 4; ++j) qout[(l * HQ + h) * D + lane * 4 + j] = static_cast<T>(v[j] * m);
        return;
    }
    if (task < NQ + 2 * NK) {
        bool isk = task < NQ + NK;
        uint t2 = isk ? task - NQ : task - NQ - NK;
        uint l = t2 / HKV, h = t2 % HKV;
        const device T* src = isk ? ks_ : vs_;
        const uint off = isk ? K0 : V0;
        const uint rst = isk ? KST : VST;
        const device float* st = isk ? kst : vst;
        const device float* tp = isk ? ktaps : vtaps;
        device float* win = isk ? kwin : vwin;
        float cv[4];
        for (int j = 0; j < 4; ++j) {
            uint c = h * D + lane * 4 + j;
            float rr[KT];
            for (int t = 0; t < KT; ++t) {
                int p = int(l) + t;   // window row
                rr[t] = p < KT - 1 ? st[(size_t)p * C + c]
                                   : static_cast<float>(src[off + (size_t)(p - (KT - 1)) * rst + c]);
            }
            float o = tp[c] * rr[0];
            for (int t = 1; t < KT; ++t) {
                if (CFMA) o = metal::fma(tp[(size_t)t * C + c], rr[t], o);
                else { float tt = tp[(size_t)t * C + c] * rr[t]; o = o + tt; }
            }
            float y = (static_cast<float>(static_cast<W>(o)) + rr[KT - 1]) * cscale[0];
            cv[j] = static_cast<float>(static_cast<T>(y));
            win[(size_t)(KT - 1 + l) * C + c] = rr[KT - 1];
            if (l == 0) for (int t = 0; t < KT - 1; ++t) win[(size_t)t * C + c] = rr[t];
        }
        if (isk) {
            float total = 0.0f;
            for (int j = 0; j < 4; ++j) { float sq = cv[j] * cv[j]; total = sq + total; }
            total = simd_sum(total);
            float m = metal::min(tq / (metal::precise::sqrt(total) + 1e-6f), 1.0f);
            for (int j = 0; j < 4; ++j) kout[(l * HKV + h) * D + lane * 4 + j] = static_cast<T>(cv[j] * m);
        } else {
            float racc = 0.0f;
            for (int j = 0; j < 4; ++j) { float sq = cv[j] * cv[j]; racc = racc + sq; }
            racc = simd_sum(racc);
            float tot = simd_sum(lane == 0 ? racc : 0.0f);
            float inv = metal::precise::rsqrt(tot / D + 1e-6f);
            for (int j = 0; j < 4; ++j) vout[(l * HKV + h) * D + lane * 4 + j] = static_cast<T>(cv[j] * inv);
        }
        return;
    }
    uint idx = task - NQ - 2 * NK;
    uint l = idx / (HQ * RCH);
    uint rem = idx % (HQ * RCH);
    uint h = rem / RCH, chunk = rem % RCH;
    float rv = 0.0f;
    if (lane < DR) rv = static_cast<float>(qs[R0 + l * QST + h * DR + lane]) * static_cast<float>(rscale[h * DR + lane]);
    float sq = rv * rv;
    float total = 0.0f;
    for (int i = 0; i < DR; ++i) { float si = simd_shuffle(sq, (ushort)i); total = si + total; }
    float m = metal::min(tr / (metal::precise::sqrt(total) + 1e-6f), 1.0f);
    float rc = static_cast<float>(static_cast<T>(rv * m));
    for (int it = 0; it < 8; ++it) {
        uint d = chunk * 256 + it * 32 + lane;
        float acc = 0.0f;
        for (int k = 0; k < DR; ++k) {
            float rk = simd_shuffle(rc, (ushort)k);
            acc = metal::fma(rk, static_cast<float>(relp[(size_t)k * NREL + d]), acc);
        }
        if (d < NREL) rel[((size_t)l * HQ + h) * NREL + d] = static_cast<T>(acc);
    }
"""
_mk_prep_rows_kernel = None
MK_PREP_ROWS_CFMA = 0


def mk_prep_rows(qs, q0, qst, r0, ks_, k0, kst_r, vs_, v0, vst_r, length, qscale, rscale, thr,
                 ktaps, vtaps, kst, vst, cscale, wdtype, relp, hq, hkv, d, dr):
    """Sliding prep for ``length`` rows. Sources are flat bf16 arrays with
    (offset, row stride). Returns q [1, L, HQ, D], k/v [1, L, HKV, D], rel
    [1, L, HQ, NREL] bf16 and the k/v windows [1, KT-1+L, C] FP32."""
    global _mk_prep_rows_kernel
    if _mk_prep_rows_kernel is None:
        _mk_prep_rows_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_mk_prep_rows",
            input_names=["qs", "ks_", "vs_", "qscale", "rscale", "thr", "ktaps", "vtaps", "kst", "vst", "cscale", "relp"],
            output_names=["qout", "kout", "vout", "rel", "kwin", "vwin"],
            source=_MK_PREP_ROWS_SOURCE,
        )
    nrel = relp.shape[1]
    rch = (nrel + 255) // 256
    c = hkv * d
    kt = ktaps.shape[0]
    outs = _mk_prep_rows_kernel(
        inputs=[qs.reshape(-1), ks_.reshape(-1), vs_.reshape(-1), qscale.reshape(-1).astype(mx.float32),
                rscale.reshape(-1).astype(mx.float32), thr, ktaps, vtaps, mx.contiguous(kst), mx.contiguous(vst),
                mx.array([cscale], dtype=mx.float32), relp],
        template=[("T", qs.dtype), ("W", wdtype), ("L", length), ("HQ", hq), ("HKV", hkv), ("D", d), ("DR", dr),
                  ("C", c), ("KT", kt), ("NREL", nrel), ("RCH", rch),
                  ("Q0", q0), ("QST", qst), ("R0", r0), ("K0", k0), ("KST", kst_r), ("V0", v0), ("VST", vst_r),
                  ("CFMA", MK_PREP_ROWS_CFMA)],
        grid=(32, length * (hq + 2 * hkv + hq * rch), 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(1, length, hq, d), (1, length, hkv, d), (1, length, hkv, d), (1, length, hq, nrel),
                       (1, kt - 1 + length, c), (1, kt - 1 + length, c)],
        output_dtypes=[qs.dtype] * 4 + [mx.float32, mx.float32],
    )
    return tuple(outs)


# Megakernel slice 12 (OMLX_INKLING2_MK_MOE_ROWS=1): verify-width MoE tail in
# one kernel, one thread per hidden column over the L rows: the top-k combine
# (``moe_combine``: products then in-order sum, cast), the MLP static conv over
# [state, m] (window written whole for the rollback), the residual add.
_MK_TAIL_ROWS_SOURCE = """
#pragma METAL fp contract(off)
    uint c = thread_position_in_grid.x;
    if (c >= H) return;
    float win[KT - 1 + L];
    for (int t = 0; t < KT - 1; ++t) win[t] = state[(size_t)t * H + c];
    for (int l = 0; l < L; ++l) {
        float total = 0.0f;
        for (int j = 0; j < TOP; ++j) {
            float wj = w[l * TOP + j];
            float s = gs[inds[l * TOP + j]] * inv_den[0];
            wj = wj * s;
            float pr = static_cast<float>(z[((size_t)l * TOP + j) * H + c]) * wj;
            total = pr + total;
        }
        win[KT - 1 + l] = static_cast<float>(static_cast<T>(total));
    }
    for (int t = 0; t < KT - 1 + L; ++t) window[(size_t)t * H + c] = win[t];
    for (int l = 0; l < L; ++l) {
        float o = taps[c] * win[l];
        for (int t = 1; t < KT; ++t) {
            if (CFMA) o = metal::fma(taps[(size_t)t * H + c], win[l + t], o);
            else { float tt = taps[(size_t)t * H + c] * win[l + t]; o = o + tt; }
        }
        float xf = win[l + KT - 1];
        float y = (static_cast<float>(static_cast<W>(o)) + xf) * cscale[0];
        T m = static_cast<T>(y);
        hout[(size_t)l * H + c] = static_cast<T>(residual[(size_t)l * H + c] + static_cast<float>(m));
    }
"""
_mk_tail_rows_kernel = None
MK_TAIL_ROWS_CFMA = 0


def mk_moe_tail_rows(z, weights, inds, gs, inv_den, residual, state, taps, cscale, wdtype):
    """z [L, TOP, H] expert outputs (no tensor scale), weights/inds [L, TOP],
    residual [1, L, H] FP32, state [1, KT-1, H] FP32, taps [KT, H] FP32.
    Returns (h [1, L, H] z.dtype, window [1, KT-1+L, H] FP32)."""
    global _mk_tail_rows_kernel
    if _mk_tail_rows_kernel is None:
        _mk_tail_rows_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_mk_tail_rows",
            input_names=["z", "w", "inds", "gs", "inv_den", "residual", "state", "taps", "cscale"],
            output_names=["hout", "window"],
            source=_MK_TAIL_ROWS_SOURCE,
        )
    length, top, h = z.shape
    kt = taps.shape[0]
    hout, win = _mk_tail_rows_kernel(
        inputs=[z, weights, inds, gs, mx.array([inv_den], dtype=mx.float32), residual.reshape(-1),
                mx.contiguous(state), taps, mx.array([cscale], dtype=mx.float32)],
        template=[("T", z.dtype), ("W", wdtype), ("H", h), ("L", length), ("TOP", top), ("KT", kt),
                  ("CFMA", MK_TAIL_ROWS_CFMA)],
        grid=(h, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, length, h), (1, kt - 1 + length, h)],
        output_dtypes=[z.dtype, mx.float32],
    )
    return hout, win
