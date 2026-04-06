#!/bin/bash
# train_gpt_oss_3b.sh - GPT-OSS 3.5B MoE pretraining on TPU v6e-4
# Demonstrates FSDP-2 + Expert Parallelism-2 with MegaBlox kernels
# 16 experts, 2 active per token, ~3.5B total params, ~0.7B active per token

set -euo pipefail

# Activate virtualenv
source /home/ptoulme/maxtext_env/bin/activate

# Directory setup
export MAXTEXT_DIR="/home/ptoulme/maxtext"
export OUTPUT_DIR="/home/ptoulme/maxtext_blog/output_3b"
export HLO_DUMP_DIR="/home/ptoulme/maxtext_blog/ir_dumps_3b"

rm -rf "${OUTPUT_DIR}" "${HLO_DUMP_DIR}"
mkdir -p "${OUTPUT_DIR}" "${HLO_DUMP_DIR}"

# Decouple from GCS so local-only runs don't fail on upload
export DECOUPLE_GCLOUD=TRUE

# XLA flags for IR dumping
export XLA_FLAGS="--xla_dump_to=${HLO_DUMP_DIR} \
  --xla_dump_hlo_module_re=jit_train_step \
  --xla_dump_hlo_as_text \
  --xla_dump_hlo_as_proto"

# v6e optimization flags
export LIBTPU_INIT_ARGS="--xla_tpu_enable_async_collective_fusion=true \
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true \
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true \
  --xla_enable_async_all_gather=true \
  --xla_tpu_overlap_compute_collective_tc=true \
  --xla_tpu_scoped_vmem_limit_kib=98304 \
  --xla_tpu_enable_data_parallel_all_reduce_opt=true \
  --xla_tpu_data_parallel_opt_different_sized_ops=true \
  --xla_tpu_use_minor_sharding_for_major_trivial_input=true \
  --xla_tpu_relayout_group_size_threshold_for_reduce_scatter=1"

echo "=== Starting GPT-OSS 3.5B MoE Pretraining ==="
echo "Model: GPT-OSS 3.5B (16 experts, 2 active, ~0.7B active/token)"
echo "Parallelism: FSDP-2 x Expert Parallelism-2 (4 chips total)"
echo "Hardware: 4x TPU v6e (32GB HBM each, 134GB total)"
echo "Steps: 200"
echo ""

cd "${MAXTEXT_DIR}"

python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
  model_name="gpt-oss-20b" \
  override_model_config=true \
  run_name="gpt_oss_3b_fsdp2_ep2" \
  base_output_directory="${OUTPUT_DIR}" \
  dataset_type=synthetic \
  steps=200 \
  enable_checkpointing=false \
  reuse_example_batch=1 \
  remat_policy=full \
  attention='flash' \
  sa_block_q=512 \
  dtype=bfloat16 \
  per_device_batch_size=1 \
  max_target_length=1024 \
  base_emb_dim=2048 \
  base_num_query_heads=32 \
  base_num_kv_heads=8 \
  head_dim=64 \
  base_num_decoder_layers=16 \
  base_mlp_dim=2048 \
  base_moe_mlp_dim=2048 \
  num_experts=16 \
  num_experts_per_tok=2 \
  vocab_size=32768 \
  megablox=true \
  sparse_matmul=true \
  capacity_factor=-1.0 \
  ici_fsdp_parallelism=2 \
  ici_expert_parallelism=2 \
  gcs_metrics=false \
  dump_hlo=true \
  dump_hlo_local_dir="${HLO_DUMP_DIR}" \
  dump_jaxpr=true \
  dump_jaxpr_local_dir="${HLO_DUMP_DIR}/jaxpr" \
  2>&1 | tee "${OUTPUT_DIR}/training.log"

echo ""
echo "=== Training Complete ==="
echo "Output: ${OUTPUT_DIR}"
echo "IR dumps: ${HLO_DUMP_DIR}"
