import os
import time

# IR dump directories
DUMP_ROOT = "compiler_dump/"
HLO_PATH = os.path.join(DUMP_ROOT, "ref_hlo")
LLO_PATH = os.path.join(DUMP_ROOT, "ref_llo")

os.makedirs(HLO_PATH, exist_ok=True)
os.makedirs(LLO_PATH, exist_ok=True)

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
    f"--xla_jf_debug_level=2"
)

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu import splash_attention


def benchmark_reference(q, k, v, mask, name="reference_attention", warmup=3, iters=10):
    """Benchmark the reference attention implementation."""
    ref_attn = splash_attention.make_masked_mha_reference(mask=mask)

    @jax.named_call
    def reference_attention(q, k, v):
        with jax.named_scope(name):
            return ref_attn(q, k, v)

    jitted = jax.jit(reference_attention)

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
    print("TPU Splash Attention REFERENCE Benchmark")
    print("=" * 70)
    print(f"HLO dump: {HLO_PATH}")
    print(f"LLO dump: {LLO_PATH}")

    # Benchmark configurations: (num_heads, seq_len, head_dim)
    # Note: Reference implementation is O(n^2) memory, so limit sizes
    # Using BF16 for TPU v6e
    sizes = [
        #(8, 1024, 128),    # Small
        (8, 2048, 128),    # Medium
        #(8, 4096, 128),    # Large
        #(32, 4096, 128),   # Production-like
        #(32, 8192, 128),   OOMs on Reference
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

        try:
            ref_name = f"reference_attention_h{num_heads}_s{seq_len}"
            result, times = benchmark_reference(q, k, v, mask, name=ref_name)
            avg_ms = sum(times) / len(times) * 1000
            min_ms = min(times) * 1000
            max_ms = max(times) * 1000

            # Calculate FLOPS (approximate for attention)
            flops = 2 * 2 * num_heads * seq_len * seq_len * head_dim
            tflops = flops / (avg_ms / 1000) / 1e12

            print(f"  avg={avg_ms:7.2f}ms  min={min_ms:7.2f}ms  max={max_ms:7.2f}ms  ~{tflops:.2f} TFLOP/s")

            results.append({
                "heads": num_heads,
                "seq": seq_len,
                "head_dim": head_dim,
                "avg_ms": avg_ms,
                "tflops": tflops,
            })
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    for r in results:
        print(f"  heads={r['heads']:2d}, seq={r['seq']:4d}, dim={r['head_dim']:3d} -> {r['avg_ms']:7.2f}ms ({r['tflops']:.2f} TFLOP/s)")

    print(f"\nDone! IR dumps in: {DUMP_ROOT}")


if __name__ == "__main__":
    main()
