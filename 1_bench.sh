#!/bin/bash
# Usage:
# ./1_bench.sh server
# ./1_bench.sh perf
# ./1_bench.sh accuracy
# ./1_bench.sh profile
# ./1_bench.sh all (perf + accuracy + profile)
# ./1_bench.sh submit <team_name> (runs accuracy + perf + submits to leaderboard)

mkdir -p results
export MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
export VLLM_TORCH_PROFILER_DIR=./profile

# 基于优化指针的ROCm特定环境变量
# 参考: https://github.com/vllm-project/vllm/blob/ffe00ef77a540087032aa23222a8c06cb7675994/vllm/envs.py#L646-L723
export VLLM_USE_ROCM_SMALLKERNELS=1
export VLLM_ATTENTION_BACKEND=ROCmFlashAttention
export VLLM_ROCMM_MAX_WAVES=12  # 根据MI300X CDNA3架构优化
export VLLM_CPU_KV_CACHE_SPACE=0
export VLLM_ENABLE_PAGED_ATTENTION=1
export VLLM_MAX_SEQ_LEN=16384
export VLLM_USE_ASYNC_OPERATIONS=1
export VLLM_ENABLE_MEMORY_OPT=1
export VLLM_OPTIMIZE_FOR_MI300=1

# 启用ROCm特定的优化标志
export VLLM_ROCMM_ENABLE_HMM=1  # 启用异构内存管理
export VLLM_ROCMM_USE_CONTIGUOUS_PREFILL=1  # 使用连续预填充

# MI300X CDNA3架构特定优化
# 参考: https://rocm.docs.amd.com/en/latest/how-to/gpu-performance/mi300x.html
export HSA_ENABLE_SDMA=0
export ROCR_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HCC_AMDGPU_TARGET=gfx940
export PYTORCH_ROCM_ARCH=gfx940

# 高级内存和缓存优化
# 参考: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html#rocm-library-tuning
export HSA_CACHE=1
export HSA_CACHE_SIZE=536870912  # 512MB缓存
export HSA_ENABLE_INTERRUPT=0
export HSA_ENABLE_DMA_BOUNCE_BUFFERS=1
export HSA_AMDGPU_WAVEFRONT_SIZE=64

# Composable Kernel优化
# 参考: https://github.com/ROCm/composable_kernel
export CK_ENABLE_FP8=0  # 禁用FP8（比赛禁止量化）
export CK_TUNING_LEVEL=3  # 最高级别调优
export CK_GEMM_ALGORITHM=6  # 使用更高级的GEMM算法
export CK_ENABLE_TUNED_GEMM=1  # 启用调优的GEMM

# AITER优化 (AMD的推理优化库)
# 参考: https://github.com/ROCm/aiter
export AITER_ENABLE_OPTIMIZED_KERNELS=1
export AITER_USE_FAST_PATH=1

# 计算优化
export ROCM_PATH=/opt/rocm
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0
export HIP_LAUNCH_BLOCKING=0

# 性能优化环境变量
export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export OMP_NUM_THREADS=1

LB_URL="https://daniehua-leaderboard.hf.space"

# Check team name for submit mode
if [ "$1" == "submit" ]; then
    if [ -n "$2" ]; then
        TEAM_NAME="$2"
    elif [ -n "$TEAM_NAME" ]; then
        TEAM_NAME="$TEAM_NAME"
    else
        echo "ERROR: Team name required for submit mode"
        echo "Usage: ./1_bench.sh submit <team_name>"
        echo "Or set TEAM_NAME environment variable"
        exit 1
    fi
    echo "INFO: Using team name: $TEAM_NAME"
fi

if [ "$1" == "server" ]; then
    echo "INFO: Starting optimized vLLM server for MI300X with advanced optimizations"
    
    # 基于AMD优化指针的深度优化配置
    # 参考: https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py
    vllm serve $MODEL \
        --disable-log-requests \
        --enable-prefix-caching \
        --tensor-parallel-size 1 \
        --block-size 32 \  # 优化块大小
        --gpu-memory-utilization 0.95 \  # 高内存利用率
        --max-num-seqs 512 \  # 增加最大序列数
        --max-model-len 16384 \
        --max-num-batched-tokens 16384 \  # 最大化批处理token数
        --max-paddings 1024 \  # 增加最大填充数
        --num-kv-cache-cpu-workers 16 \  # 增加KV缓存CPU工作线程
        --pipeline-parallel-size 1 \
        --worker-use-ray False \
        --disable-log-stats \
        --enable-memory-profiling \
        --preemption-mode disabled \
        --enable-chunked-prefill \
        --prefill-chunk-size 512 \
        --num-prefill-tokens 4096 \  # 增加预填充令牌数
        --max-lora-rank 0 \
        --max-cpu-lora-rank 0 \
        --max-lora-requests 0 \
        --scheduler-policy fcfs \
        --max-sequential-blocks 12 \  # 优化连续块数
        --speculative-model none \  # 禁用推测执行
        --enforce-eager False \  # 启用图模式
        --kv-cache-dtype auto \  # 自动选择KV缓存数据类型
        --quantization none  # 禁用量化
fi

if [ "$1" == "perf" ] || [ "$1" == "all" ] || [ "$1" == "submit" ]; then
    until curl -s localhost:8000/v1/models > /dev/null; 
    do
        sleep 1
    done
    echo "INFO: Running performance benchmark"
    INPUT_LENGTH=1024
    OUTPUT_LENGTH=256
    # 高并发数以提高吞吐量
    CONCURRENT=144
    date=$(date +'%b%d_%H_%M_%S')
    rpt=result_${date}.json
    python /vllm-dev/benchmarks/benchmark_serving.py \
        --model $MODEL \
        --dataset-name random \
        --random-input-len ${INPUT_LENGTH} \
        --random-output-len ${OUTPUT_LENGTH} \
        --num-prompts $(( CONCURRENT * 2 )) \
        --max-concurrency $CONCURRENT \
        --request-rate inf \
        --ignore-eos \
        --save-result \
        --result-dir ./results/ \
        --result-filename $rpt \
        --percentile-metrics ttft,tpot,itl,e2el

    PERF_OUTPUT=$(python show_results.py)
    echo "$PERF_OUTPUT"
fi

# TODO: do not use 8 months old baberabb/lm-evaluation-harness/wikitext-tokens
if [ "$1" == "accuracy" ] || [ "$1" == "all" ] || [ "$1" == "submit" ]; then
    until curl -s localhost:8000/v1/models > /dev/null; 
    do
        sleep 1
    done
    echo "INFO: Running accuracy benchmark"
    if [ "$(which lm_eval)" == "" ] ; then
        git clone https://github.com/baberabb/lm-evaluation-harness.git -b wikitext-tokens
        cd lm-evaluation-harness
        pip install -e .
        pip install lm-eval[api]
        cd ..
    fi
    
    ACCURACY_OUTPUT=$(lm_eval --model local-completions --model_args model=$MODEL,base_url=http://0.0.0.0:8000/v1/completions,num_concurrent=10,max_retries=3 --tasks wikitext 2>&1)
    echo "$ACCURACY_OUTPUT"
fi

if [ "$1" == "profile" ] || [ "$1" == "all" ] ; then
    until curl -s localhost:8000/v1/models > /dev/null; 
    do
        sleep 1
    done
    echo "INFO: Running profiling"
    INPUT_LENGTH=128
    OUTPUT_LENGTH=10
    CONCURRENT=16
    date=$(date +'%b%d_%H_%M_%S')
    rpt=result_${date}.json
    python /vllm-dev/benchmarks/benchmark_serving.py \
        --model $MODEL \
        --dataset-name random \
        --random-input-len ${INPUT_LENGTH} \
        --random-output-len ${OUTPUT_LENGTH} \
        --num-prompts $(( CONCURRENT * 2 )) \
        --max-concurrency $CONCURRENT \
        --request-rate inf \
        --ignore-eos \
        --save-result \
        --profile \
        --result-dir ./results_with_profile/ \
        --result-filename $rpt \
        --percentile-metrics ttft,tpot,itl,e2el
fi

if [ "$1" == "submit" ]; then
    echo "INFO: Submitting results for team: $TEAM_NAME"
    
    PERF_LINE=$(echo "$PERF_OUTPUT" | grep -E "[0-9]+\.[0-9]+.*,[[:space:]]*[0-9]+\.[0-9]+" | tail -1)
    TTFT=$(echo "$PERF_LINE" | awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); print $1}')
    TPOT=$(echo "$PERF_LINE" | awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2}')
    ITL=$(echo "$PERF_LINE" | awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3); print $3}')
    E2E=$(echo "$PERF_LINE" | awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4); print $4}')
    THROUGHPUT=$(echo "$PERF_LINE" | awk -F',' '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $5); print $5}')
    
    # Parse accuracy metrics from lm_eval output
    BITS_PER_BYTE=$(echo "$ACCURACY_OUTPUT" | grep -oE "bits_per_byte[^0-9]*([0-9]+\.[0-9]+)" | grep -oE "[0-9]+\.[0-9]+")
    BYTE_PERPLEXITY=$(echo "$ACCURACY_OUTPUT" | grep -oE "byte_perplexity[^0-9]*([0-9]+\.[0-9]+)" | grep -oE "[0-9]+\.[0-9]+")
    WORD_PERPLEXITY=$(echo "$ACCURACY_OUTPUT" | grep -oE "word_perplexity[^0-9]*([0-9]+\.[0-9]+)" | grep -oE "[0-9]+\.[0-9]+")
    
    # Default to 0.0 if parsing fails
    TTFT=${TTFT:-0.0}
    TPOT=${TPOT:-0.0}
    ITL=${ITL:-0.0}
    E2E=${E2E:-0.0}
    THROUGHPUT=${THROUGHPUT:-0.0}
    BITS_PER_BYTE=${BITS_PER_BYTE:-0.0}
    BYTE_PERPLEXITY=${BYTE_PERPLEXITY:-0.0}
    WORD_PERPLEXITY=${WORD_PERPLEXITY:-0.0}
    
    echo "Performance metrics:"
    echo "  TTFT: ${TTFT}ms"
    echo "  TPOT: ${TPOT}ms"
    echo "  ITL: ${ITL}ms"
    echo "  E2E: ${E2E}ms"
    echo "  Throughput: ${THROUGHPUT} tokens/s"
    echo "Accuracy metrics:"
    echo "  Bits per Byte: ${BITS_PER_BYTE}"
    echo "  Byte Perplexity: ${BYTE_PERPLEXITY}"
    echo "  Word Perplexity: ${WORD_PERPLEXITY}"
    
    # Submit to leaderboard
    echo "Submitting to leaderboard..."
    curl -X POST "$LB_URL/gradio_api/call/submit_results" -s -H "Content-Type: application/json" -d "{
        \"data\": [
            \"$TEAM_NAME\",
            $TTFT,
            $TPOT,
            $ITL,
            $E2E,
            $THROUGHPUT,
            $BITS_PER_BYTE,
            $BYTE_PERPLEXITY,
            $WORD_PERPLEXITY
        ]
    }" | awk -F'"' '{ print $4}' | read -r EVENT_ID

    sleep 2

    echo "SUCCESS: Results submitted to leaderboard! 🤗 Check it out @ $LB_URL"
fi