#!/usr/bin/env bash
# 可通过环境变量覆盖测试规模与拓扑参数
# NUM_TOKENS_PER_RANK / EXPERT_NUM / HIDDEN_SIZE / TOPK / BLOCKS_PER_KERNEL
# NNODES / RANKS_PER_NODE / NP / CHUNK_TOKENS
NUM_TOKENS_PER_RANK=${NUM_TOKENS_PER_RANK:-1024}
EXPERT_NUM=${EXPERT_NUM:-128}
HIDDEN_SIZE=${HIDDEN_SIZE:-8192}
TOPK=${TOPK:-8}
BLOCKS_PER_KERNEL=${BLOCKS_PER_KERNEL:-16}
CHUNK_TOKENS=${CHUNK_TOKENS:-32}
# CHUNK_TOKENS 为每个 chunk 的 token 数，影响 flag 数量与流水化重叠程度
NNODES=${NNODES:-2}
RANKS_PER_NODE=${RANKS_PER_NODE:-4}
NP=${NP:-8}
SSH_PORT=${SSH_PORT:-2345}

#bench
BENCH_ONLY=${BENCH_ONLY:-0}
BENCH_ITERS=${BENCH_ITERS:-10}
BENCH_WARMUP=${BENCH_WARMUP:-5}

#数据偏斜
ZIPF_ALPHA=${ZIPF_ALPHA:-0.5}

export NVSHMEM_BOOTSTRAP_LIBRARY=/workspace/nvshmem_install/lib/nvshmem_bootstrap_mpi.so

/usr/local/openmpi/bin/mpirun \
  --allow-run-as-root \
  -np "$NP" \
  -x NUM_TOKENS_PER_RANK="$NUM_TOKENS_PER_RANK" \
  -x ZIPF_ALPHA="$ZIPF_ALPHA" \
  -x EXPERT_NUM="$EXPERT_NUM" \
  -x HIDDEN_SIZE="$HIDDEN_SIZE" \
  -x TOPK="$TOPK" \
  -x BLOCKS_PER_KERNEL="$BLOCKS_PER_KERNEL" \
  -x CHUNK_TOKENS="$CHUNK_TOKENS" \
  -x NVSHMEM_BOOTSTRAP="MPI" \
  -x NVSHMEM_BOOTSTRAP_LIBRARY="/workspace/nvshmem_install/lib/nvshmem_bootstrap_mpi.so" \
  -x BENCH_ONLY="$BENCH_ONLY" \
  -x BENCH_ITERS="$BENCH_ITERS" \
  -x BENCH_WARMUP="$BENCH_WARMUP" \
  /workspace/balance_ep/build/test_dispatch \
  --nnodes "$NNODES" \
  --ranks_per_node "$RANKS_PER_NODE"
