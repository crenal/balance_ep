#!/usr/bin/env bash
# 可通过环境变量覆盖测试规模与拓扑参数
# NUM_TOKENS_PER_RANK / EXPERT_NUM / HIDDEN_SIZE / TOPK / BLOCKS_PER_KERNEL
# NNODES / RANKS_PER_NODE / NP / CHUNK_TOKENS
export NUM_TOKENS_PER_RANK=${NUM_TOKENS_PER_RANK:-4096}
export EXPERT_NUM=${EXPERT_NUM:-64}
export HIDDEN_SIZE=${HIDDEN_SIZE:-8192}
export TOPK=${TOPK:-8}
export BLOCKS_PER_KERNEL=${BLOCKS_PER_KERNEL:-32}
export CHUNK_TOKENS=${CHUNK_TOKENS:-16}
# CHUNK_TOKENS 为每个 chunk 的 token 数，影响 flag 数量与流水化重叠程度
export NNODES=${NNODES:-2}
export RANKS_PER_NODE=${RANKS_PER_NODE:-8}
export NP=${NP:-16}
export SSH_PORT=${SSH_PORT:-2345}
export DISPATCH_MODE=${DISPATCH_MODE:-2}
#bench
export BENCH_ONLY=${BENCH_ONLY:-1}
export BENCH_ITERS=${BENCH_ITERS:-10}
export BENCH_WARMUP=${BENCH_WARMUP:-5}


export NVSHMEM_BOOTSTRAP_LIBRARY=/workspace/nvshmem_install/lib/nvshmem_bootstrap_mpi.so

/usr/local/openmpi/bin/mpirun \
  --allow-run-as-root \
  --mca plm_rsh_agent "ssh -p ${SSH_PORT}" \
  --mca oob_tcp_if_include eth0 \
  --mca btl_tcp_if_include eth0 \
  --hostfile hostfile \
  -np "$NP" \
  -x NUM_TOKENS_PER_RANK="$NUM_TOKENS_PER_RANK" \
  -x ZIPF_ALPHA=0.99\
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
  -x NVSHMEM_SYMMETRIC_SIZE=2G \
  -x DISPATCH_MODE="$DISPATCH_MODE" \
  -x RANDOM_MAP=0 \
  -x NVSHMEM_DISABLE_GDRCOPY=0 \
  /workspace/balance_ep/build/test_dispatch \
  --nnodes "$NNODES" \
  --ranks_per_node "$RANKS_PER_NODE"
