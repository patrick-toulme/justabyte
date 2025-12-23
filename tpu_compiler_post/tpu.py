import os

# Create dump directories
DUMP_ROOT = "compiler_dump/"
HLO_DUMP_PATH = os.path.join(DUMP_ROOT, "hlo")
LLO_DUMP_PATH = os.path.join(DUMP_ROOT, "llo")

os.makedirs(HLO_DUMP_PATH, exist_ok=True)
os.makedirs(LLO_DUMP_PATH, exist_ok=True)

os.environ["XLA_FLAGS"] = (
    f"--xla_dump_hlo_as_text "
    f"--xla_dump_to={HLO_DUMP_PATH} "
    f"--xla_dump_hlo_pass_re=.* "
)

os.environ["LIBTPU_INIT_ARGS"] = (
    f"--xla_jf_dump_to={LLO_DUMP_PATH} "
    f"--xla_jf_dump_hlo_text=true "
    f"--xla_jf_dump_llo_text=true "
    f"--xla_jf_dump_llo_html=false "
    f"--xla_jf_dump_llo_static_gaps=true "
    f"--xla_jf_emit_annotations=true "
    f"--xla_jf_debug_level=2"
)

# Import JAX after setting env vars
import jax
import jax.numpy as jnp


@jax.named_call
def matmul_1(x, w1):
    """Stage 1: Linear projection (like Q @ K^T)"""
    return x @ w1


@jax.named_call
def rms_norm(h):
    """Stage 2: RMS Normalization"""
    rms = jnp.sqrt(jnp.mean(h ** 2, axis=-1, keepdims=True) + 1e-6)
    return h / rms


@jax.named_call
def softmax(h):
    """Stage 3: Softmax (row-wise, numerically stable)"""
    h_max = jnp.max(h, axis=-1, keepdims=True)
    exp_h = jnp.exp(h - h_max)
    return exp_h / jnp.sum(exp_h, axis=-1, keepdims=True)


@jax.named_call
def matmul_2(h, w2):
    """Stage 4: Output projection (like attention @ V)"""
    return h @ w2


def mini_attention(x, w1, w2):
    """
    A minimal attention-like block:
    matmul → rms_norm → softmax → matmul
    
    """
    h = matmul_1(x, w1)
    h = rms_norm(h)
    h = softmax(h)
    out = matmul_2(h, w2)
    return out


def main():
    # Small shapes to keep IR readable
    batch, d_in, d_mid, d_out = 16, 64, 64, 32
    
    # Create inputs
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    
    x = jax.random.normal(k1, (batch, d_in))
    w1 = jax.random.normal(k2, (d_in, d_mid)) * 0.02
    w2 = jax.random.normal(k3, (d_mid, d_out)) * 0.02
    
    # JIT compile and run
    jitted_fn = jax.jit(mini_attention)
    
    # First call triggers compilation (and IR dump)
    result = jitted_fn(x, w1, w2)
    
    # Block until computation is done
    result.block_until_ready()
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Output sample: {result[0, :5]}")
    print(f"\nDumps written to:")
    print(f"  HLO: {HLO_DUMP_PATH}")
    print(f"  LLO: {LLO_DUMP_PATH}")


if __name__ == "__main__":
    main()