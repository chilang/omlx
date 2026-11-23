# Faster Inkling 2 Mini decode on Apple Silicon (fork branch)

> **Status (2026-11-23):** this branch is based on oMLX 0.8.2. Upstream has since released 0.8.3 and merged its own
> versions of several of these optimizations, plus fixes this branch lacks (for example an HTTP 500 on M1 from oversized
> threadgroups and an FP16 NVFP4 overflow fix). It has only been measured on an M4 Max.
>
> Measured against upstream main (`04e6e8e0`) on the same M4 Max: this branch's forward step is 17% faster at one token
> and 23-28% faster at 3-4 verify rows (partly because upstream's verify is row-exact, which this branch's is not); one-stream decode on a code prompt is 1.20x upstream with MTP depth 2 and 1.07x with
> MTP off. Upstream is 7% faster on time to first token at 16K tokens. Upstream's greedy output (MTP off) differs from this
> branch on 3 of 4 test prompts, because upstream's numerics changed after 0.8.2. Work is under way to bring the expert
> kernels to upstream. Until then: on an M4 Max, this branch is faster for decode; elsewhere, or if you need the newer
> fixes, use upstream oMLX.

This branch (`prod/fused-20261123`) is oMLX 0.8.2 plus a series of Inkling 2
kernel changes tuned on an M4 Max (40-core GPU, 128 GB). It is not upstreamed. All changes are in
`omlx/patches/inkling2/{kernels,model}.py` and `omlx/cache/paged_ssd_cache.py`; the Inkling kernels are built at run time
through `mx.fast.metal_kernel`, so they need no Xcode step.

## Install

```bash
git clone -b prod/fused-20261123 https://github.com/chilang/omlx.git
cd omlx
make install            # or: make install-no-kernels (the Inkling 2 path does not use the native kernels)
```

Model: [`mlx-community/Inkling-2-Mini-oQ4e-mtp`](https://huggingface.co/mlx-community/Inkling-2-Mini-oQ4e-mtp)
(tested at revision `3eea042`; includes the MTP draft head).

```bash
hf download mlx-community/Inkling-2-Mini-oQ4e-mtp --revision 3eea042 --local-dir ~/models/Inkling-2-Mini-oQ4e-mtp
```

## Settings

Speculative decoding with draft depth 2 (3 rows verified per step) was fastest here. Put this in
`<base-path>/model_settings.json` (default base path `~/.omlx`), or set the same values in the web UI's model settings:

```json
{"version": 2, "models": {"Inkling-2-Mini-oQ4e-mtp": {"mtp_enabled": true, "mtp_fixed_depth": 2}}}
```

Then:

```bash
omlx serve --model-dir ~/models --port 8000
```

The server log prints `Inkling 2 megakernel flags active: ...` once the model loads.

## Kernel flags

On this branch the bit-exact fused kernels are on by default. Set any of them to `0` to turn it off:
`OMLX_INKLING2_MK_{MOE,OPROJ,PREP,NORM3,PREP_ROWS,MOE_ROWS,OPROJ_ROWS,ROUTE_ROWS,SINGLE,AFFINE,ROUTER}`.
With all of them at `0` the code runs the previous build's path.

Two experimental flags change greedy output and stay off: `OMLX_INKLING2_MK_LOOP_FAST` and
`OMLX_INKLING2_MK_ROUTER_PERROW`.

## Measured on an M4 Max (40-core GPU, 128 GB)

| | Stock oMLX 0.8.2 (MTP depth 3) | This branch (MTP depth 2) |
|---|---:|---:|
| Decode, code prompt, one stream | 51 tok/s | 91-92 tok/s on an idle machine |
| Decode, same prompt, MTP off | - | 74-76 tok/s |
| Prefill, 512 / 4K / 16K-token prompt | ~640 tok/s | ~757 / 678 / 686 tok/s |

Single-token forward step: 15.5 ms with the fused kernels, 17.8 ms without. Absolute numbers move by up to 30% with other GPU
load on the machine; compare builds in alternating runs.

Correctness checks used for every kept change: identical logits and greedy output with MTP off, against the previous build.
With all fused kernels on, MTP-off output matched the previous build on 200/200 prompts, and HumanEval, GSM8K and a
tool-call set scored identically item by item. One known difference from stock 0.8.2 predates the fused kernels: the
2-8 row expert kernel used in MTP verification uses fused multiply-add, so about 2 in a million values differ from stock
by one bf16 step and MTP-on text can differ slightly. `OMLX_INKLING2_EXACT_VERIFY=1` makes that kernel bit-equal to stock
at about 2% of verify time.

Other Macs: the kernels are tuned for M4 Max threadgroup sizes. They should run on other Apple Silicon GPUs, but this
branch has only been measured on M4 Max.
