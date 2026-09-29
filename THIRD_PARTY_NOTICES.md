# Third-party notices

This repository is distributed under the BSD 3-Clause License (see `LICENSE`).
The parts below are copied from or adapted from other projects, whose copyright and licenses remain in force for that material.

## vLLM (Apache License 2.0)

Copyright contributors to the vLLM project.  <https://github.com/vllm-project/vllm>

The attention forwards of the models this plugin replaces follow vLLM 0.19.1's model code, so that the dense path of each model is reproduced exactly where the sparse path is not taken:

- `semblo/vllm_plugin/qwen3_5_model.py` (`_project_qkv`, `_official_dense_attention`): adapted from `Qwen3NextAttention.forward` in `vllm/model_executor/models/qwen3_next.py`.
- `semblo/vllm_plugin/nemotron_h_model.py` (the attention decoder layer's `forward`): adapted from `NemotronHAttentionDecoderLayer.forward` and `NemotronHAttention.forward` in `vllm/model_executor/models/nemotron_h.py`.

`semblo/mrope_yarn_fix.py` and `semblo/vllm_plugin/gdn_conv_patch.py` wrap vLLM functions at run time and copy no vLLM code.
FlashAttention-3 is used through vLLM (`vllm.vllm_flash_attn`) and is not included here.

## NVIDIA RULER (Apache License 2.0) and lm-evaluation-harness (MIT License)

RULER: Copyright (c) 2024, NVIDIA CORPORATION. <https://github.com/NVIDIA/RULER>

lm-evaluation-harness: Copyright (c) 2020 EleutherAI. <https://github.com/EleutherAI/lm-evaluation-harness>

- `lm_eval_tasks/ruler_cached/semblo_ruler_*.yaml`: the RULER task definitions of lm-evaluation-harness (`lm_eval/tasks/ruler/*.yaml`), renamed, pointed at the cached prompt builders, with one added length.
- `lm_eval_tasks/ruler_cached/cached_utils.py` (`qa_generate_input_output_filled`, `qa_generate_samples_filled`, `cwe_sys_word_pair_random_filled`): adapted from lm-evaluation-harness's `lm_eval/tasks/ruler/qa_utils.py` (`generate_input_output`, `generate_samples`) and `cwe_utils.py` (`sys_word_pair_random`), which carry NVIDIA's RULER license header.
  The rest of the module calls lm-evaluation-harness's RULER functions and copies none.

## LongBench v2 (MIT License) and EvalScope (Apache License 2.0)

LongBench: Copyright (c) 2023 THU-KEG & Zhipu AI. <https://github.com/THUDM/LongBench>

EvalScope: Copyright 2022-2023 Alibaba ModelScope. <https://github.com/modelscope/evalscope>

- `semblo/tasks/longbench_v2.py`: the zero-shot and chain-of-thought prompt templates of LongBench v2 (`prompts/0shot*.txt`) and EvalScope's LongBench-v2 prompt, verbatim.
- `semblo/scoring.py`: the answer extraction of LongBench v2's `extract_answer`, re-implemented.

## OpenAI MRCR

`scripts/eval_openai_mrcr.py` re-implements the grading of OpenAI MRCR (the `SequenceMatcher` ratio after the required prefix, as the dataset card describes) and uses the sampling settings of the Qwen3.5 model card.

## Dolma 3 Longmino (Open Data Commons Attribution License v1.0)

`assets/latency_prompt/dolma1m_qwen_ids.npy`, the prompt of the latency measurements, holds the Qwen3.5 token ids of text from Dolma 3 Longmino (`allenai/dolma3_longmino_pool`, Allen Institute for AI, <https://huggingface.co/datasets/allenai/dolma3_longmino_pool>), which is licensed under ODC-By 1.0.

## Data used at run time (not distributed here)

The evaluations read datasets and files that this repository does not contain and that are subject to their own terms: LongBench v2 (`THUDM/LongBench-v2`), OpenAI MRCR (`openai/mrcr`), HotpotQA (`hotpot_dev_distractor_v1.json`, CC BY-SA 4.0), and the English word list used for common-word extraction.
The model weights (Qwen3.5, Nemotron 3.5 Lightning) are loaded from their Hugging Face repositories under their own licenses.
