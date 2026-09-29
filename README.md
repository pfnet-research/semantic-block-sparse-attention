# Semantic Block Sparse Attention

This is a vLLM plugin for long-context inference with sparse attention, based on the paper:

> **Coarse Semantic Blocks for Efficient Training-Free Sparse Attention**
>
> Sosuke Kobayashi, Shoichiro Yamaguchi, Seiya Tokui

The plugin groups prompt keys into semantic blocks in each full-attention layer.
Each query attends to the blocks whose centroids match it best, reducing the cost of both prefill and decode.
No training or changes to the model weights are required.

The code was rewritten for public release with the help of a coding agent.

## Quick start

Use an NVIDIA Hopper GPU (H100/H200) with Python 3.12 and vLLM 0.19.1.
The plugin uses vLLM's internal APIs, so other vLLM versions may need changes.

```bash
git clone https://github.com/pfnet-research/semantic-block-sparse-attention.git
cd semantic-block-sparse-attention
pip install -r requirements/h200-vllm.txt
pip install --no-deps -e .
```

Serve Qwen3.5-4B with a 512K context:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
vllm serve Qwen/Qwen3.5-4B \
  --hf-overrides '{"architectures": ["SembloQwen3_5ForConditionalGeneration"],
                   "text_config": {"max_position_embeddings": 525312,
                                   "rope_parameters": {"rope_type": "yarn", "factor": 4.0,
                                       "original_max_position_embeddings": 262144, "attn_factor": 0.8782488562869419,
                                       "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                                       "mrope_section": [11, 11, 10], "mrope_interleaved": true}}}' \
  --max-model-len 525312 --max-num-batched-tokens 525312 \
  --max-num-seqs 1 --no-enable-chunked-prefill --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
  --kv-cache-memory-bytes 2000000000 --gpu-memory-utilization 0.9 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}'
```

The custom architecture enables the plugin, and YaRN with factor 4 extends the model's native 256K context.
The plugin uses the paper's settings by default.
To use `vllm.LLM`, pass the same options as keyword arguments.

Other supported models and their architecture names:

- Qwen3.5-35B-A3B: `SembloQwen3_5MoeForConditionalGeneration`
- Nemotron 3.5 Lightning: `SembloNemotronHForCausalLM`

## Saving memory

For Qwen3.5, the following mode reduces activation memory by processing the prompt through all layers 8K tokens at a time.
It builds blocks from roughly the first 64K tokens; later keys join those blocks.

```bash
export SEMBLO_PREFILL_ORDER=token SEMBLO_PREFILL_CHUNK=8192
```

The paper reports about 15% longer prefill with similar accuracy in this mode.
It is not supported for Nemotron.

## Limitations

- The plugin handles one request at a time (`--max-num-seqs 1`).
- Prompts must contain at least 128 tokens after chat formatting.
- It does its own chunked prefill, so vLLM's chunked prefill must be off and `--max-num-batched-tokens` must be at least `--max-model-len`.
- Prefix caching, speculative decoding and pipeline parallelism are not supported.
- CUDA graphs must be `PIECEWISE` or off.
  Full-graph decode is intended for [fixed-length latency runs](REPRODUCE.md#full-graph-decode).

## Reproducing the paper

[REPRODUCE.md](REPRODUCE.md) covers the evaluations (RULER, MRCR, LongBench v2), latency measurements, settings and differences from the paper.

## License

BSD 3-Clause. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists the code adapted from other projects.
