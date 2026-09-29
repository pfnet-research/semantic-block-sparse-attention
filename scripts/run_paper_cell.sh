#!/usr/bin/env bash
# One measurement of the paper's main configuration (Qwen3.5-4B), on one GPU.
#
#   bash scripts/run_paper_cell.sh ruler   <ours|dense> <length> <task> <seed> <outdir> [gpu]
#   bash scripts/run_paper_cell.sh latency <ours|dense> <length> <window> <outdir> [gpu]
#
#   length  RULER: 131072 | 262144 | 524288 | 1048576
#           latency: 131072 | 262144 | 524288 | 1047552 (the 1M latency prompt)
#   task    mk3 | mk2 | mv | cwe | hp   (RULER-5: multi-key 3 and 2, multi-value, common words, HotpotQA)
#   seed    the clustering seed; the paper averages 0, 1 and 2 (dense ignores it)
#   window  0 or 1, the window of the latency prompt
#   gpu     the device to use (default 0)
#
# The paper's latency runs use both windows for ours through 512K and dense at 512K, and window 0 otherwise.
# Each window is measured three times. The paper takes the median for each window and averages the results
# when two windows are used.
#
# The settings are the paper's: mean block size u = 2,048, budget K = L/16, selection temperature 0.1, size penalty
# 0.25, whole-vector normalised pre-RoPE keys, KV-outer prefill, at least 2^7 clusters, no initial block.  RULER
# prefills the prompt in 32K slices (sliced prefill) at every length and decodes with the one-launch decode under
# PIECEWISE CUDA graphs; latency uses whole-prompt prefill up to 512K, sliced prefill at 1M, the five-launch decode
# and the whole decode step as one CUDA graph (both decodes select the same blocks).  Beyond the native 256K window
# the RoPE is factor-4 YaRN interpolation with its attention scaling cancelled.  Dense RULER uses the stock
# checkpoint; dense latency uses chunked prefill in 16K chunks.
#
# Environment (defaults in brackets):
#   PAPER_RULER_CACHE   the RULER prompt cache of MK-3, MK-2, MV and the 128K CWE        [data/ruler_cache/qwen3_5]
#   PAPER_RULER_ROOT    the cache root of the filled CWE and HotpotQA, and their inputs    [data/ruler_cache]
#   PAPER_WORDLIST      the word list of the filled CWE                        [data/wordlists/usr_share_dict_words.txt]
#   PAPER_CHECKPOINT    the prepared checkpoint for latency (scripts/prepare_qwen3_5_checkpoint.py)
#                                                               [checkpoints/qwen3_5_4b/raw-keys]
#   PAPER_DENSE_MODEL   the stock Qwen3.5-4B snapshot for dense latency       [the Hugging Face cache's snapshot]
#   PAPER_LATENCY_IDS   the latency prompt's token ids (scripts/build_latency_prompt.py)
#                                               [assets/latency_prompt/dolma1m_qwen_ids.npy, in the repository]
#   PAPER_RANGE         START:STOP, a range of the 100 RULER items (to split a measurement over several GPUs)
# The RULER model presets (configs/models.yaml) name the prepared checkpoint's directory themselves.
# A measurement whose result file exists is not run again.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PAPER_RULER_CACHE="${PAPER_RULER_CACHE:-$ROOT/data/ruler_cache/qwen3_5}"
PAPER_RULER_ROOT="${PAPER_RULER_ROOT:-$ROOT/data/ruler_cache}"
PAPER_WORDLIST="${PAPER_WORDLIST:-$ROOT/data/wordlists/usr_share_dict_words.txt}"
PAPER_CHECKPOINT="${PAPER_CHECKPOINT:-$ROOT/checkpoints/qwen3_5_4b/raw-keys}"
PAPER_LATENCY_IDS="${PAPER_LATENCY_IDS:-$ROOT/assets/latency_prompt/dolma1m_qwen_ids.npy}"
if [ -z "${PAPER_DENSE_MODEL:-}" ]; then
  for PAPER_DENSE_MODEL in "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--Qwen--Qwen3.5-4B/snapshots/*/; do break; done
fi

# The model presets of configs/models.yaml.
PAPER_OURS=qwen3_5_4b_semblo_512k
PAPER_OURS_1M=qwen3_5_4b_semblo_1m

# The selection: temperature 0.1 and size penalty 0.25, and the fixed sizes of the selection kernels.
PAPER_RULE='"semblo_selection_temperature":0.1,"semblo_bucket_size_penalty":0.25'
PAPER_BASE='"semblo_max_probe_buckets":16384,"semblo_num_centroids":128,"semblo_top_centroids_per_factor":128,"semblo_fused_select_block_queries":32,"semblo_vllm_query_chunk_size":8192'
# Beyond the native 256K: factor-4 YaRN interpolation with its attention scaling cancelled (attn_factor = 1 / (0.1 ln 4 + 1)).
PAPER_YARN='{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144,"attn_factor":0.8782488562869419}'
PAPER_CC_QUALITY='{"cudagraph_mode":"PIECEWISE","inductor_compile_config":{"combo_kernels":false,"benchmark_combo_kernel":false}}'
PAPER_CC_LATENCY='{"inductor_compile_config":{"combo_kernels":false,"benchmark_combo_kernel":false}}'
# The latency tool takes the selection settings as flags.
PAPER_OPS=(--hf-text-override semblo_bucket_size_penalty=0.25
           --hf-text-override semblo_selection_temperature=0.1
           --hf-text-override semblo_num_centroids=128
           --hf-text-override semblo_top_centroids_per_factor=128
           --hf-text-override semblo_max_probe_buckets=16384
           --hf-text-override semblo_fused_select_block_queries=32
           --hf-text-override semblo_vllm_query_chunk_size=8192)

paper_env () {
  export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  export VLLM_PLUGINS=semblo
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  export SEMBLO_FUSED_UMAX=L SEMBLO_FUSED_DECODE=1
  export SEMBLO_FUSED_STRICT_MASK=0 SEMBLO_FUSED_ROWMASK=0
  export SEMBLO_FA3_BODY=16 SEMBLO_FA3_ALIGN=16 SEMBLO_KV_OUTER=1
  export SEMBLO_FLAT_MIN_CLUSTERS_LOG2=7
  export TORCHINDUCTOR_DETERMINISTIC=1
  export VLLM_DISABLE_COMPILE_CACHE=1
  export TOKENIZERS_PARALLELISM=false
  export SEMBLO_FLAT_UNIT_TOKENS=2048
  export SEMBLO_ROUTING_NORMALIZE=whole
  export SEMBLO_HOTPOT_JSON="$PAPER_RULER_ROOT/hotpot_dev_distractor_v1.json"
  export SEMBLO_WORDLIST="$PAPER_WORDLIST"
}

paper_task () {   # tag -> lm_eval task
  case "$1" in
    mk3) echo semblo_ruler_niah_multikey_3 ;;
    mk2) echo semblo_ruler_niah_multikey_2 ;;
    mv)  echo semblo_ruler_niah_multivalue ;;
    cwe) echo semblo_ruler_cwe ;;
    hp)  echo semblo_ruler_qa_hotpot ;;
    *) echo "unknown task '$1' (mk3, mk2, mv, cwe, hp)" >&2; return 1 ;;
  esac
}

paper_suite () {   # length tag -> suite file
  local n
  case "$1" in 131072) n=128k ;; 262144) n=256k ;; 524288) n=512k ;; 1048576) n=1m ;;
    *) echo "RULER lengths are 131072, 262144, 524288, 1048576 (got $1)" >&2; return 1 ;;
  esac
  case "$2" in
    mk3|mk2|mv) echo "configs/lm_eval/ruler_${n}_focus_vllm_cached.yaml" ;;
    cwe) if [ "$1" = 131072 ]; then echo "configs/lm_eval/ruler_${n}_focus_vllm_cached.yaml"
         else echo "configs/lm_eval/ruler_${n}_focus_vllm_cached_fill.yaml"; fi ;;
    hp) echo "configs/lm_eval/ruler_hotpot_${n}_vllm_cached.yaml" ;;
  esac
}

paper_cache () {   # length tag -> the prompt cache (MK-3, MK-2, MV and the 128K CWE: the paper's items; else the root)
  case "$2" in
    mk3|mk2|mv) echo "$PAPER_RULER_CACHE" ;;
    cwe) if [ "$1" = 131072 ]; then echo "$PAPER_RULER_CACHE"; else echo "$PAPER_RULER_ROOT"; fi ;;
    hp) echo "$PAPER_RULER_ROOT" ;;
  esac
}

report_ruler () {   # outdir label
  if [ -f "$1/results_python_api.json" ]; then
    python3 -c "
import json, sys; d = json.load(open(sys.argv[1])); v = next(iter(d['results'].values()))
k = next(k for k in v if k.endswith(',none') and not k.startswith(('alias', '4096')) and 'stderr' not in k)
print('   %-40s %s = %.4f  n=%d' % (sys.argv[2], k, v[k], v['sample_len']))" "$1/results_python_api.json" "$2"
  else
    echo "   $2 FAILED (see $1/run.log)"; return 1
  fi
}

paper_ruler () {   # arm gpu length tag seed outdir
  local arm="$1" gpu="$2" plen="$3" tag="$4" seed="$5" R="$6"
  local task suite cache; task=$(paper_task "$tag") || return 1; suite=$(paper_suite "$plen" "$tag") || return 1
  cache=$(paper_cache "$plen" "$tag")
  local mml=$((plen + 1024)) rope=""
  [ "$plen" -gt 262144 ] && rope=",\"rope_parameters\":$PAPER_YARN"
  local -a range=(); [ -n "${PAPER_RANGE:-}" ] && range=(--samples-range "$PAPER_RANGE")
  mkdir -p "$R"
  if [ -f "$R/results_python_api.json" ]; then report_ruler "$R" "$plen $arm $tag seed $seed (cached)"; return; fi
  if [ "$arm" = ours ]; then
    local model=$PAPER_OURS K=$((plen / 16)); local -a tp=()
    [ "$plen" = 1048576 ] && model=$PAPER_OURS_1M && tp=(--vllm-arg tensor_parallel_size=1)
    [ "$plen" = 524288 ] && [ ${#range[@]} = 0 ] && range=(--samples-range 0:100)
    env SEMBLO_RULER_CACHE_DIR="$cache" SEMBLO_SINK_TOKENS=0 SEMBLO_PREFILL_CHUNK=32768 SEMBLO_TARGET_TOKENS=$K \
        SEMBLO_ENGAGE_ABOVE_TOKENS=0 SEMBLO_EMPTY_CACHE_ABOVE_GIB=16 SEMBLO_DECODE_GRAPH=0 CUDA_VISIBLE_DEVICES=$gpu \
      python -m semblo.lm_eval_runner \
      --suite "$suite" --models-config configs/models.yaml \
      --model "$model" --task "$task" --limit 100 "${range[@]}" \
      --vllm-arg max_model_len=$mml --vllm-arg max_num_batched_tokens=$mml "${tp[@]}" \
      --vllm-arg kv_cache_memory_bytes=2000000000 --vllm-arg gpu_memory_utilization=0.90 \
      --vllm-json-arg "hf_overrides={\"text_config\":{\"max_position_embeddings\":$mml,\"semblo_vllm_token_capacity\":$plen,\"semblo_vllm_own_prompt_cache\":true,$PAPER_BASE,$PAPER_RULE,\"semblo_vllm_codebook_seed\":$seed$rope}}" \
      --vllm-json-arg "compilation_config=$PAPER_CC_QUALITY" \
      --output-path "$R" > "$R/run.log" 2>&1
  elif [ "$arm" = dense ]; then
    local -a a=()
    case "$plen" in
      131072)  a=(--model qwen3_5_4b_dense_512k --vllm-arg max_model_len=132096
                  --vllm-json-arg "hf_overrides={\"text_config\":{\"max_position_embeddings\":132096}}"
                  --vllm-json-arg "compilation_config=$PAPER_CC_QUALITY") ;;
      262144)  a=(--model qwen3_5_4b_dense_256k --vllm-arg max_model_len=263168
                  --vllm-json-arg "compilation_config=$PAPER_CC_QUALITY") ;;
      524288)  a=(--model qwen3_5_4b_dense_512k --vllm-arg max_model_len=525312
                  --vllm-json-arg "hf_overrides={\"text_config\":{\"max_position_embeddings\":525312$rope}}"
                  --vllm-json-arg "compilation_config=$PAPER_CC_QUALITY") ;;
      1048576) a=(--model qwen3_5_4b_dense_1m
                  --vllm-json-arg "hf_overrides={\"text_config\":{\"max_position_embeddings\":1049600$rope}}") ;;
    esac
    env -u TORCHINDUCTOR_DETERMINISTIC SEMBLO_RULER_CACHE_DIR="$cache" VLLM_PLUGINS="" CUDA_VISIBLE_DEVICES=$gpu \
      python -m semblo.lm_eval_runner \
      --suite "$suite" --models-config configs/models.yaml --task "$task" --limit 100 "${range[@]}" \
      "${a[@]}" --output-path "$R" > "$R/run.log" 2>&1
  else
    echo "the arm is ours or dense (got $arm)" >&2; return 1
  fi
  report_ruler "$R" "$plen $arm $tag seed $seed"
}

paper_latency () {   # arm gpu length window outdir
  local arm="$1" gpu="$2" plen="$3" off="$4" O="$5" mode=FULL_DECODE_ONLY
  local name="${arm}_${mode}_${plen}_w${off}"
  mkdir -p "$O"
  if [ ! -f "$O/$name.json" ]; then
    local mml=$((plen + 1024)) K=$((plen / 16)) ntok=64 util=0.60
    local -a extra=() envs=() require=(--require-cudagraph-mode "$mode")
    local model="$PAPER_CHECKPOINT" plugins=semblo
    if [ "$arm" = dense ]; then
      model="$PAPER_DENSE_MODEL"; plugins=""; extra=(--enable-chunked-prefill --max-num-batched-tokens 16384); require=()
    elif [ "$arm" = ours ]; then
      extra=("${PAPER_OPS[@]}" --hf-text-override semblo_vllm_token_capacity=$plen
             --hf-text-override "semblo_vllm_own_prompt_cache=true" --kv-cache-memory-bytes 2000000000)
      envs+=(SEMBLO_FULL_GRAPH_DECODE=1 SEMBLO_ENGAGE_ABOVE_TOKENS=0 SEMBLO_FUSED_DECODE=3)
      # The plugin's defaults (sliced prefill, releasing cached allocator memory) are for RULER. Latency runs without them.
      envs+=(SEMBLO_EMPTY_CACHE_ABOVE_GIB=0)
      [ "$plen" -le 1000000 ] && envs+=(SEMBLO_PREFILL_CHUNK=0)
    else
      echo "the arm is ours or dense (got $arm)" >&2; return 1
    fi
    [ "$plen" -gt 262144 ] && extra+=(--hf-text-override "rope_parameters=$PAPER_YARN")
    if [ "$plen" -gt 1000000 ]; then
      ntok=32; mml=1048192
      if [ "$arm" = dense ]; then util=0.85
      else
        util=0.90
        envs+=(SEMBLO_PREFILL_CHUNK=32768 SEMBLO_STREAM_KERNELS=compiled SEMBLO_CHECK_POSITIONS=1)
        extra+=(--hf-text-override "semblo_target_tokens=$K"
                --hf-text-override semblo_flat_unit_tokens=2048)
      fi
    fi
    env "${envs[@]}" VLLM_PLUGINS="$plugins" SEMBLO_TARGET_TOKENS=$K SEMBLO_FLAT_UNIT_TOKENS=2048 CUDA_VISIBLE_DEVICES=$gpu \
      python -m semblo.vllm_plugin.benchmark \
      --model "$model" --tokenizer Qwen/Qwen3.5-4B \
      --prompt-ids-npy "$PAPER_LATENCY_IDS" --prompt-tokens "$plen" --prompt-offset "$off" \
      --output-tokens $ntok --max-model-len $mml \
      --warmup 1 --repeats 3 --gpu-memory-utilization $util \
      --cuda-graphs --cudagraph-mode "$mode" "${require[@]}" --allow-token-mismatch \
      --limit-mm-per-prompt '{"image":0,"video":0}' \
      --compilation-config "$PAPER_CC_LATENCY" \
      --output-json "$O/$name.json" \
      --hf-text-override max_position_embeddings=$mml "${extra[@]}" \
      > "$O/$name.log" 2>&1
  fi
  if [ -f "$O/$name.json" ]; then
    python3 -c "
import json, sys; d = json.load(open(sys.argv[1]))
print('   %-36s prefill %7.2f s   decode %6.2f ms/token' % (sys.argv[2], d['prefill_median_ms'] / 1000, d['tpot_median_ms']))" "$O/$name.json" "$name"
  else
    echo "   $name FAILED (see $O/$name.log)"; return 1
  fi
}

paper_env
kind="${1:-}"; shift || true
case "$kind" in
  ruler)
    [ $# -ge 5 ] || { sed -n '4,5p' "$0"; exit 2; }
    paper_ruler "$1" "${6:-0}" "$2" "$3" "$4" "$5" ;;
  latency)
    [ $# -ge 4 ] || { sed -n '4,6p' "$0"; exit 2; }
    paper_latency "$1" "${5:-0}" "$2" "$3" "$4" ;;
  *)
    sed -n '2,32p' "$0"; exit 2 ;;
esac
