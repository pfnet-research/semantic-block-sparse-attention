# Reproducing the paper

## Data

The prompt for the latency measurements is included in this repository.
It is `assets/latency_prompt/dolma1m_qwen_ids.npy`, which holds 1,050,624 Qwen3.5 token ids of text from Dolma 3 Longmino.
You need to get or build the rest yourself.

| Data | Location | How to get it |
|---|---|---|
| RULER prompts | `data/ruler_cache/` | Built on first use, or in advance with `scripts/build_ruler_caches.py` |
| HotpotQA for RULER QA | `data/ruler_cache/hotpot_dev_distractor_v1.json` | Download the HotpotQA distractor dev set |
| Word list for RULER CWE | `data/wordlists/usr_share_dict_words.txt` | Convert [RULER's English word list](https://github.com/NVIDIA/RULER/blob/main/scripts/data/synthetic/json/english_words.json) with the command below |
| LongBench v2 token counts | `data/longbench_v2/prompt_token_counts_qwen35_official.json` | Run `scripts/count_longbench_v2_tokens.py` |

The CWE word list needs one word per line.
To download and convert it:

```bash
mkdir -p data/wordlists
curl -fL https://media.githubusercontent.com/media/NVIDIA/RULER/main/scripts/data/synthetic/json/english_words.json \
  | python -c 'import json, sys; print("\n".join(json.load(sys.stdin).values()))' \
  > data/wordlists/usr_share_dict_words.txt
```

The environment variables `SEMBLO_RULER_CACHE_DIR`, `SEMBLO_HOTPOT_JSON` and `SEMBLO_WORDLIST` change the first three locations when running the tools directly.
For `scripts/run_paper_cell.sh`, use the `PAPER_*` overrides listed at the top of the script (`PAPER_WORDLIST` for the word list).

A RULER cache is built with a fixed random seed, so rebuilding it with the same inputs gives the same prompts.
These prompts are not the ones used in the paper, which were built without a fixed seed.
Your scores may therefore differ slightly from the paper.
The cache records hashes of the CWE word list and local HotpotQA file, so changing either input builds a separate cache.

## Running the experiments

`scripts/run_paper_cell.sh` runs one measurement on one GPU with the settings of the paper.
Sparse runs use the [prepared checkpoints](#prepared-checkpoints).

```bash
# RULER: method (ours or dense), length, task (mk3, mk2, mv, cwe or hp), seed, output directory, GPU
bash scripts/run_paper_cell.sh ruler ours  524288 mk3 0 runs/ours_512k_mk3_seed0 0
bash scripts/run_paper_cell.sh ruler dense 524288 mk3 0 runs/dense_512k_mk3 1
# Latency: method, prompt length, prompt window (0 or 1), output directory, GPU
bash scripts/run_paper_cell.sh latency ours 131072 0 runs/latency 0
```

RULER runs at 131072, 262144, 524288 and 1048576 tokens.
Latency runs at 131072, 262144, 524288 and 1047552 tokens.
The paper averages RULER scores over seeds 0, 1 and 2.
For latency, it takes the median of three measurements per prompt window, then averages the results if two windows are used.

A RULER run has 100 items.
To split a run across GPUs, set `PAPER_RANGE=START:STOP` to run a subset of them.
Other RULER suites use `semblo-lm-eval` with a suite from `configs/lm_eval/`.

For MRCR (8 needles), use `scripts/eval_openai_mrcr.py`.
The script uses the current dataset, and `--repeats` changes only the sampling seed.
The paper used a fixed dataset version and varied both sampling and clustering seeds.

LongBench v2 uses `semblo-run` with a config such as `configs/benchmarks/longbench_v2_qwen_512k.yaml`.
These configs are evaluation examples.
Their item selection and decoding settings differ from the paper's main results.

Start each `semblo-run` experiment in a new output directory.
Use `--resume` only when the model and experiment settings are unchanged.
The runner rejects existing output unless `--resume` is set, and checks the saved settings before resuming.

## Settings

The defaults are the settings the paper uses for RULER.

- The mean block size is 2,048 tokens.
- The budget is L/16 tokens. Here L is `max_model_len` rounded down to a multiple of 16,384.
- The selection temperature is 0.1, and the size penalty is 0.25.
- Prefill processes the prompt in slices of 32K tokens.

Nemotron 3.5 Lightning uses the paper's settings for Nemotron instead.
The budget is L/8, the keys are centred for each prompt, and k-means runs longer.

You can change the settings with these environment variables.

| Variable | Default | Effect |
|---|---|---|
| `SEMBLO_TARGET_TOKENS` | L/16 | Budget in tokens |
| `SEMBLO_FLAT_UNIT_TOKENS` | 2048 | Mean block size |
| `SEMBLO_PREFILL_CHUNK` | 32768 | Prefill slice size. With 0, prefill runs on the whole prompt at once, which is faster but needs more memory. |
| `SEMBLO_FUSED_DECODE`, `SEMBLO_FULL_GRAPH_DECODE` | 1, unset | Decode kernels. The paper's latency runs use 3 and 1, with vLLM's `cudagraph_mode` set to `FULL_DECODE_ONLY` (see `scripts/run_paper_cell.sh`). |
| `SEMBLO_ENGAGE_ABOVE_TOKENS` | 0 | Prompts up to this length use dense attention; this requires `semblo_vllm_own_prompt_cache=false` so vLLM retains their KV cache |
| `SEMBLO_SINK_TOKENS` | 0 | Always read the first N tokens. The paper's initial block uses 1024. |
| `SEMBLO_FIT_PREFIX_TOKENS` | unset | Build the blocks from the first N tokens only (the prefix index) |
| `SEMBLO_PREFILL_ORDER=token` | unset | Build the blocks from about the first 64K tokens, then run the prompt through all layers 8K tokens at a time, like ordinary chunked prefill. Set `SEMBLO_PREFILL_CHUNK=8192` with it. |
| `SEMBLO_SUFFIX_FLUSH_EVERY`, `SEMBLO_SUFFIX_KEEP_EXACT` | unset, 1024 | Every N generated tokens, move older generated tokens into the blocks, keeping the last M as they are |
| `SEMBLO_BLOCK_TOKENS` | unset | Use blocks of N consecutive tokens instead (the positional baseline of the paper) |
| `SEMBLO_ROUTING_CENTER` | 0 (1 for Nemotron) | Centre the keys of each prompt before building the blocks |

The memory-saving prefill mode (`SEMBLO_PREFILL_ORDER=token SEMBLO_PREFILL_CHUNK=8192`) is available for Qwen3.5.
For Qwen3.5-4B, the paper reports a drop in peak memory from 41 to 36 GiB at 512K and from 70 to 60 GiB at 1M, with about 15% longer prefill and similar accuracy.

## Full-graph decode

The fixed-length latency runs in `scripts/run_paper_cell.sh` use `SEMBLO_FULL_GRAPH_DECODE=1` and `cudagraph_mode=FULL_DECODE_ONLY`.
With the plugin's own KV cache and no suffix flushing, the prompt length must equal `semblo_vllm_token_capacity`.
Use `PIECEWISE` for variable-length prompts.

## Comparisons in the appendix

These settings run the training-free comparisons described in the appendix.

| Comparison | How to run it |
|---|---|
| Post-RoPE routing keys | `SEMBLO_ROUTING_KEYS=post` |
| Raw keys with Euclidean k-means | `SEMBLO_ROUTING_NORMALIZE=none SEMBLO_FLAT_METRIC=euclidean`, with the temperature in `semblo_selection_temperature` |
| Non-rotary dimensions, PCA, random projections | A checkpoint made by `scripts/make_routing_map.py` (`--arm nope-dims`, `pca`, `random`) |
| Min/max block descriptor | `scripts/make_routing_map.py --arm raw-keys-doubled`, then `SEMBLO_BLOCK_DESCRIPTOR=minmax` with positional blocks |
| Centroids fitted on a corpus | `scripts/dump_corpus_keys.py`, then `scripts/fit_corpus_codebook.py`, then `SEMBLO_FIXED_CODEBOOK=<file>` |
| Clusters ranked by dense attention | `SEMBLO_ORACLE_SELECT=clusters` |
| Seeding of k-means | `SEMBLO_FLAT_INIT=kmeanspp` or `farthest` |
| Whitened keys (Nemotron) | `SEMBLO_ROUTING_CENTER=whiten` |
| Dense-prefix cutoff | `SEMBLO_FA3_DENSE_PREFIX_X` (default 4) |
| Positional blocks with shared selection | `SEMBLO_BLOCK_TOKENS=B SEMBLO_ROUTING_NORMALIZE=none SEMBLO_TORCH_SELECT=1 SEMBLO_KV_OUTER=0 SEMBLO_FUSED_CONSENSUS=1 SEMBLO_CONSENSUS_SCORE=sumexp`, with `semblo_fused_select_block_queries` = g |

PCA needs pre-RoPE keys from a corpus: run `scripts/dump_corpus_keys.py` with `SEMBLO_DUMP_PROJECTED_KEYS=<dir>` and pass the directory to `--keys-dir`.

## YaRN in vLLM 0.19.1

Beyond 256K tokens, the paper extends Qwen3.5 with YaRN (factor 4).
The serving example in README.md sets `attn_factor` to 1 / (0.1 ln 4 + 1), cancelling YaRN's attention scaling so that only frequency interpolation is applied.
A different `factor` needs its own `attn_factor`.

vLLM 0.19.1 computes one part of YaRN differently from Transformers for this model (vllm-project/vllm#58879).
The first time vLLM builds a YaRN RoPE, the plugin checks it against Transformers.
If it differs in this known way, the plugin fixes it.
If it differs in any other way, the plugin stops with an error.

## Prepared checkpoints

For the paper, the model directories were prepared with `scripts/prepare_qwen3_5_checkpoint.py`, `scripts/prepare_qwen3_5_moe_checkpoint.py` and `scripts/prepare_nemotron_checkpoint.py`.
The presets in `configs/models.yaml` and `scripts/run_paper_cell.sh` still use these directories.
You do not need them to run the plugin.
A stock Hugging Face model uses identity routing built in memory instead of loading routing tensors from a prepared directory.
