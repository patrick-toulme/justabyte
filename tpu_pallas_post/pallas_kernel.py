import os
import time

# IR dump directories
DUMP_ROOT = "compiler_dump/"
HLO_PATH = os.path.join(DUMP_ROOT, "kernel_hlo")
LLO_PATH = os.path.join(DUMP_ROOT, "kernel_llo")
MOSAIC_PATH = os.path.join(DUMP_ROOT, "kernel_mosaic")

os.makedirs(HLO_PATH, exist_ok=True)
os.makedirs(LLO_PATH, exist_ok=True)
os.makedirs(MOSAIC_PATH, exist_ok=True)

os.environ["XLA_FLAGS"] = (
    f"--xla_dump_hlo_as_text "
    f"--xla_dump_to={HLO_PATH} "
    f"--xla_dump_hlo_pass_re=.* "
)
os.environ["LIBTPU_INIT_ARGS"] = (
    f"--xla_jf_dump_to={LLO_PATH} "
    f"--xla_jf_dump_hlo_text=true "
    f"--xla_jf_dump_llo_text=true "
    f"--xla_jf_dump_llo_html=false "
    f"--xla_jf_dump_llo_static_gaps=true "
    f"--xla_jf_emit_annotations=true "
    f"--xla_jf_debug_level=2 "
    f"--xla_mosaic_dump_to={MOSAIC_PATH} "
    f"--xla_mosaic_enable_dump_debug_info=true "
    f"--xla_mosaic_enable_llo_source_annotations=true"
)

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu import splash_attention
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as kernel_mod
import hashlib

# Monkey-patch get_kernel_name to produce shorter filenames for Mosaic dumps
_original_get_kernel_name = kernel_mod.get_kernel_name

def _short_kernel_name(block_metadata, is_mqa, save_residuals, is_segmented, phase):
    """Shorter kernel name to avoid 255 char filename limit."""
    attention_type = "mqa" if is_mqa else "mha"
    residuals = "_res" if save_residuals else ""
    segments = "_seg" if is_segmented else ""
    # Hash the block metadata instead of spelling it all out
    meta_str = str(sorted(block_metadata.items()))
    meta_hash = hashlib.md5(meta_str.encode()).hexdigest()[:8]
    return f"splash_{attention_type}_{phase}{segments}{residuals}_{meta_hash}"

kernel_mod.get_kernel_name = _short_kernel_name


def benchmark_kernel(q, k, v, mask, block_sizes, name="splash_kernel", warmup=3, iters=10):
    """Benchmark the Pallas splash attention kernel."""
    splash_attn = splash_attention.make_splash_mha_single_device(
        mask=mask,
        block_sizes=block_sizes,
    )

    @jax.named_call
    def splash_attention_kernel(q, k, v):
        with jax.named_scope(name):
            return splash_attn(q, k, v)

    jitted = jax.jit(splash_attention_kernel)

    # Warmup
    for _ in range(warmup):
        result = jitted(q, k, v)
        result.block_until_ready()

    # Timed runs
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        result = jitted(q, k, v)
        result.block_until_ready()
        times.append(time.perf_counter() - start)

    return result, times


def main():
    print("=" * 70)
    print("TPU Splash Attention KERNEL Benchmark")
    print("=" * 70)
    print(f"HLO dump: {HLO_PATH}")
    print(f"LLO dump: {LLO_PATH}")
    print(f"MOSAIC dump: {MOSAIC_PATH}")

    # Benchmark configurations: (num_heads, seq_len, head_dim)
    # Note: head_dim must be multiple of 128 for TPU
    # Using BF16 for TPU v6e (918 TFLOP/s peak)
    sizes = [
        #(8, 1024, 128),    # Small
        (8, 2048, 128),    # Medium
       # (8, 4096, 128),    # Large
        #(32, 4096, 128),   # Production-like
        #(32, 8192, 128),   # Very Large
    ]

    # Block size configurations to try
    # Note: block_kv_compute must be multiple of 128
    # b2048 is best for large sequences, but needs seq >= 2048
    block_configs = [
        ("b512", splash_attention.BlockSizes(
            block_q=512, block_kv=512, block_kv_compute=512,
            block_q_dkv=512, block_kv_dkv=512, block_kv_dkv_compute=512,
            block_q_dq=512, block_kv_dq=512,
        )),
        ("b1024", splash_attention.BlockSizes(
            block_q=1024, block_kv=1024, block_kv_compute=1024,
            block_q_dkv=1024, block_kv_dkv=1024, block_kv_dkv_compute=1024,
            block_q_dq=1024, block_kv_dq=1024,
        )),
        ("b2048", splash_attention.BlockSizes(
            block_q=2048, block_kv=2048, block_kv_compute=2048,
            block_q_dkv=2048, block_kv_dkv=2048, block_kv_dkv_compute=2048,
            block_q_dq=2048, block_kv_dq=2048,
        )),
    ]

    key = jax.random.PRNGKey(42)

    print("\n" + "-" * 70)
    print("Running benchmarks...")
    print("-" * 70)

    results = []

    for num_heads, seq_len, head_dim in sizes:
        print(f"\n[Size] heads={num_heads}, seq={seq_len}, head_dim={head_dim}")

        # Create inputs (BF16 for TPU v6e)
        k1, k2, k3, key = jax.random.split(key, 4)
        q = jax.random.normal(k1, (num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        k = jax.random.normal(k2, (num_heads, seq_len, head_dim), dtype=jnp.bfloat16)
        v = jax.random.normal(k3, (num_heads, seq_len, head_dim), dtype=jnp.bfloat16)

        # Causal mask - needs MultiHeadMask for MHA
        single_mask = splash_attention.CausalMask(shape=(seq_len, seq_len))
        mask = splash_attention.MultiHeadMask(masks=[single_mask] * num_heads)

        for block_name, block_sizes in block_configs:
            # Skip if block size > seq_len
            if block_sizes.block_q > seq_len or block_sizes.block_kv > seq_len:
                print(f"  [{block_name:8s}] SKIPPED (block > seq_len)")
                continue

            try:
                kernel_name = f"splash_kernel_{block_name}_h{num_heads}_s{seq_len}"
                result, times = benchmark_kernel(q, k, v, mask, block_sizes, name=kernel_name)
                avg_ms = sum(times) / len(times) * 1000
                min_ms = min(times) * 1000
                max_ms = max(times) * 1000

                # Calculate FLOPS (approximate for attention)
                # 2 * num_heads * seq^2 * head_dim (for QK^T and softmax@V)
                flops = 2 * 2 * num_heads * seq_len * seq_len * head_dim
                tflops = flops / (avg_ms / 1000) / 1e12

                print(f"  [{block_name:8s}] avg={avg_ms:7.2f}ms  min={min_ms:7.2f}ms  max={max_ms:7.2f}ms  ~{tflops:.2f} TFLOP/s")

                results.append({
                    "heads": num_heads,
                    "seq": seq_len,
                    "head_dim": head_dim,
                    "block": block_name,
                    "avg_ms": avg_ms,
                    "tflops": tflops,
                })
            except Exception as e:
                print(f"  [{block_name:8s}] ERROR: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary - Best block size per configuration:")
    print("=" * 70)

    from itertools import groupby
    for (h, s, d), group in groupby(results, key=lambda x: (x["heads"], x["seq"], x["head_dim"])):
        group_list = list(group)
        best = min(group_list, key=lambda x: x["avg_ms"])
        print(f"  heads={h:2d}, seq={s:4d}, dim={d:3d} -> best: {best['block']:8s} ({best['avg_ms']:.2f}ms, {best['tflops']:.2f} TFLOP/s)")

    print(f"\nDone! IR dumps in: {DUMP_ROOT}")


if __name__ == "__main__":
    main()
