# exllamav3-rdna

**[exllamav3](https://github.com/turboderp-org/exllamav3) for AMD RDNA3 — Radeon RX 7900 XTX (gfx1100),
and the gfx11 family generally (RX 7800 XT gfx1101, Strix Halo gfx1151 build; benchmarked on the XTX).**

Upstream exllamav3 has had its own ROCm backend since v1.6.0, and it is the base of this branch. Before
that, [phoenixhaxor/exllamav3-rocm](https://github.com/phoenixhaxor/exllamav3-rocm) ported exllamav3 to
RDNA3 with hand-written kernels; that port has not moved since 2026-09-26 and cannot be rebased (two
independent ROCm backends over the same files). Measured on the same card, upstream decoded 10-19 %
slower than that port while using 2 GB less VRAM. Branch `rdna` is upstream plus the port's kernels that
still pay off on top of current upstream, each behind a switch and each kept only on a measurement.
Default branch `rdna`; `master` mirrors upstream.

## What branch `rdna` carries

| # | Change | Switch (default on) | Measured on a 7900 XTX |
|---|---|---|---|
| 1 | Split-k small-m EXL3 matmul (`rocm/quant/exl3_rdna3*`): input transform once per matmul, one wave per 16x16 tile column, register prefetch ring on raw buffer loads, RDNA3-tuned `mul1` decode (24-bit multiplies, `v_sad_u8` byte sums, `v_dot2_f32_f16`), direct tile-word loads for 5/6-bit, residency-wave grid sizing. Dispatched ahead of the multirow GEMV for every EXL3 linear up to the reconstruct threshold | `EXL3_RDNA3_GEMM` | DFlash2 round 39.6 -> 37.8 ms (-4.5 %) |
| 2 | Shared-input and **sliced** projection bundles (MLP gate/up, attention q/k/v/g, Gated DeltaNet qkv/z) as one input-transform launch plus one matmul launch, `blockIdx.z` per entry; sliced Q/K/V bundles re-enabled on gfx11 (`EXL3_QKV_SLICE`) | `EXL3_RDNA3_GEMM` | 37.8 -> 36.4 ms (-3.7 %) |
| 3 | HIP flash-decoding split kernels for the graphed decode / verify path (`rocm/rdna3_attn.cu`): q_len 1 with `v_dot2` scores and online softmax; q_len 2-8 on an 8-bit cache with `v_wmma_f32_16x16x16_f16` for QK^T and PV. Same arguments and partial layout as the Triton split kernel (adapted to upstream's dense row packing), so the Triton combine kernel reduces it | `EXL3_RDNA3_ATTN`, `EXL3_RDNA3_ATTN_SPLIT_MULT` (16) | round at 8k context 39.3 -> 36.9 ms (-6 %), at 32k 47.8 -> 40.1 ms (-16 %) |
| 4 | fp32-output GEMMs that the WMMA GEMM's gfx1100 table declines run as an fp16 product plus a widening copy (rocBLAS has no matrix-core solution for fp16 x fp16 -> fp32 on gfx11). Not under graph capture | `EXL3_RDNA3_HGEMM_WIDEN` | 300-token prefill 644 -> 343 ms; 2000 tokens unchanged |
| 5 | Fewer launches per decode step: `silu(gate) * up`, the attention output gate and the Gated DeltaNet gated RMSNorm folded into the next matmul's input transform; the RMSNorm in front of a projection bundle writes that bundle's input transform in its tail; the gate/up matmul's epilogue writes the down projection's | `EXL3_FUSE_ACT`, `EXL3_FUSE_GNORM`, `EXL3_FUSE_NORM_HAD`, `EXL3_ACT_EPI` | kernels per generation 66.8k -> 51.1k; round 36.1 -> 35.5 ms (8k: 36.85 -> 36.3) |
| 7 | A block's final residual add (`x += mlp(x)`) is handed to the next block's input RMSNorm, which adds it in the same kernel; byte-identical output | `EXL3_RESID_DEFER` | round 34.75 -> 34.6 ms (8k: 35.55 -> 35.4) |
| 6 | Gated DeltaNet b/a projection: all token rows from one pass over the weights, 16-byte loads, `v_dot2_f32_f16` (upstream's kernel re-reads the 1 MB weight per token with 4-byte loads: 23 us -> 4.5 us per call). New in this branch | `EXL3_RDNA_BA_GEMV` | round 35.5 -> 34.8 ms (8k: 36.3 -> 35.6) |

All of them are runtime-gated to gfx11 devices; anything else keeps upstream's paths.

Numerics: 1-3 change the accumulation order of decode-path matmuls / attention. Decode logits against
upstream's own kernels: KL 3.8e-5 (matmul) / 8.4e-5 (with bundles), top-1 agreement 100 %, and both
builds sit at the same distance from a full-forward reference. Greedy text is unchanged on most prompts.
4 rounds residual-stream projections to fp16 during prefill (logit KL 2.6e-4, top-1 100 %). 5 is
byte-identical for the norm hand-off; 6 moves decode logits by KL 7.9e-6.

### Measured

Qwen3.8-27B EXL3 3.5 bpw + DFlash2 5.0 bpw draft, Q8 cache (draft Q4), batch 1, ROCm 7.2.4,
torch 2.13.0+rocm7.2. In-process, greedy, median ms per speculative round on the same prompts:

| | short prompts | 8k context | 32k context | VRAM (served, 196k cache) |
|---|---|---|---|---|
| upstream `a3bfbfd` | 39.6 | 42.6 | 47.8 | 22.3 G |
| this branch | 34.6 | 35.4 | — | 22.6 G |
| exllamav3-rocm `360c902` | 34.6 | 36.3 | — | 24.4 G |

Served through TabbyAPI (decode t/s; prompt processing t/s):

| | prose | json | refactor | reasoning | 8k: prefill / decode | 30k prefill | prose prompt |
|---|---|---|---|---|---|---|---|
| upstream `a3bfbfd` | 59.1 | 163.4 | 197.5 | 135.4 | 1110 / 66.1 | 1138 | 325 |
| this branch | 75.3 | 175.6 | **225.1** | 160.2 | 1299 / **83.9** | 1232 | 596 |
| exllamav3-rocm `360c902` | 67.5 | 184.7 | 219.9 | 158.7 | 1217 / 81.3 | 1147 | 563 |

Served decode depends on the greedy text, which differs between builds; refactor and the 8k row produce
byte-identical text on both, so those two are like for like. VRAM served: 22.6 vs 24.4 GB.

## What is still left

Per generation this branch launches ~51k kernels against exllamav3-rocm's ~37k for the same GPU work
(1594 vs 1638 ms busy): the remaining Gated DeltaNet decode kernels (the port runs b/a, conv and the
recurrence as one megakernel, written for a recurrent-state layout and rollback scheme upstream no longer
has), residual adds (the port defers them into the next norm) and per-round sampling launches. Together
rounds now take the same time on short prompts and less at depth. Next candidates, each under 1 %:
batched verification sampling, a cheaper draft lm_head (the 6-bit 248k-vocabulary head is ~1.1 ms of
every round). Known gap: in MTP draft mode a 30k-token prefill runs at ~1020 t/s against ~1170 on
exllamav3-rocm (DFlash2 mode is unaffected); not investigated yet.

## Tried and dropped

| Idea | Why not |
|---|---|
| The port's Gated DeltaNet register-resident recurrence | upstream now carries the state slice in registers itself (`0f685aa9`) and re-advances from one initial state instead of writing per-token history (`3a8e2fb8`), which was the port's own bottleneck |
| The port's sliding-window skip in the decode split kernel | upstream has it |
| The port's fused-megakernel Gated DeltaNet step | written against the old recurrent-state format and per-token history; 6 alone brings the DeltaNet decode time per round to the port's (~2.2 vs ~2.3 ms) |
| `EXL3_GRAPHS=1` on top of upstream | no difference in speed or VRAM; the port's `EXL3_NOGRAPH=mlp,gdn,attn` saves ~0.2 GB there, so graphs are not where its extra 2 GB goes |

### Quality

EvalScope (ifeval, gsm8k, humaneval; 250 / 250 / 164 items, temperature 0, thinking on, `qwen-sharp`
template), paired against exllamav3-rocm on the same items: 77.2 / 98.0 / 96.3 % vs 76.4 / 97.6 / 97.6 %;
discordant items 7 vs 6 over 664, McNemar p = 1.0. No measurable difference.

Functional checks on the served build: tool calls, image input, four concurrent requests at batch 1 and
batch 2 (no aborted jobs; batch 2 at 21.9 GB), retrieval of a needle from ~100k tokens of context.

## Build

ROCm 7.2.x and a ROCm build of PyTorch, then `PYTORCH_ROCM_ARCH=gfx1100 python setup.py build_ext
--inplace` (or `pip install .`); `gfx1100;gfx1101;gfx1151` builds as well. TabbyAPI runs it unchanged.

## Credits

- exllamav3 by turboderp and contributors, MIT.
- The RDNA3 kernels in 1-4 come from [phoenixhaxor/exllamav3-rocm](https://github.com/phoenixhaxor/exllamav3-rocm)
  (MIT); their design notes and measurements are in that repository's README.

---


<p align="center">
  <img src="doc/logo.png" width="640" alt="Llama 3.1 8B Instruct quantization benchmark across bits per weight">
</p>

[Installation](#installation) · [Supported models](#architecture-support) · [Examples](#examples) · [Quantization](#exl3-quantization) · [Community](#community)

ExLlamaV3 is an inference library for running local LLMs on modern consumer GPUs, with flexible quantization and parallel inference.

- **Quantization** - [EXL3](#exl3-quantization), based on QTIP, plus 2–8 bit cache quantization.
- **Parallel inference** - Flexible tensor-parallel and expert-parallel inference for consumer hardware setups.
- **CPU offloading** - Allows large MoE models to run with limited GPU resources. AVX2 and AVX512 support.  
- **Generation** - Continuous, dynamic batching, speculative decoding, multimodal support.
- **Integrations** - Broad [HF model support](#architecture-support), a [Transformers plugin](examples/transformers_integration.py), and an OpenAI-compatible API via [TabbyAPI](https://github.com/theroyallab/tabbyAPI/).

> [!TIP]
> **Looking for a server?** [TabbyAPI](https://github.com/theroyallab/tabbyAPI/) is the official and recommended backend server. It provides an OpenAI-compatible API for local or remote inference, HF model downloading, embedding model support, and HF Jinja2 chat templates. Its startup script manages and installs prerequisites to help you get started.

<p align="center">
  <img src="doc/qb_kld.png" width="640" alt="Llama 3.1 8B Instruct quantization benchmark across bits per weight">
</p>

## Installation

Start by making sure you have the appropriate version of [PyTorch](https://pytorch.org/get-started/locally/) installed (CUDA 12.4 or later) since the Torch dependency is not automatically handled by `pip`. Then pick a method below:

### Prebuilt wheel · recommended

Pick a wheel from the [releases page](https://github.com/turboderp-org/exllamav3/releases), then e.g.:

```sh
pip install https://github.com/turboderp-org/exllamav3/releases/download/v0.0.6/exllamav3-0.0.6+cu128.torch2.8.0-cp313-cp313-linux_x86_64.whl
```

### Install from PyPI

```sh
pip install exllamav3
```
Note that the PyPI package does not contain a prebuilt extension and requires the CUDA toolkit and build prerequisites (i.e. VS Build Tools on Windows, gcc on Linux, `python-dev` headers etc.).

### Build from source

<details>
<summary>Source installation with uv or pip</summary>


`exllamav3` declares a minimum `torch` version (>= 2.6.0) and CUDA version (>= 12.4), but beyond that the user is free to select a version of `torch` that is compatible with their environment.

`torch` can be installed in three ways (from least to most effort):
1. **with `uv`, setting only `--extra cuXXX`** installs `torch` automatically with the specified CUDA version, `torch` version is selected by `uv` from compatible versions in the specific index associated with the chosen CUDA version (options 1 and 2)
2. **with `uv`, creating a thin project that depends on `exllamav3[cuXXX]` and pins a specific `torch` version** — like (1) but `torch` is pinned in the thin project's `pyproject.toml`, see [pinning a specific PyTorch version (optional)](#pinning-a-specific-pytorch-version-optional) for details
3. Manually with `uv pip` or `pip` (options 3 and 4)

The flavor extras (`--extra`) are `cu124`, `cu126`, `cu128`, `cu129`, `cu130`, and `cu132` — pick the one matching your installed CUDA build. Both `uv sync` and `pip install .` build the package in an isolated environment where your `torch` is not visible, so they install the extension sources and compile them at first import (JIT, a few minutes once per torch version). For a precompiled install run `pip install --no-build-isolation .` in an environment that already has `torch`, or use the release wheels. Selecting a flavor installs the matching CUDA build of `torch`.

**Option 1 — Working in the cloned repo directly (`uv sync`):**

```sh
git clone https://github.com/turboderp-org/exllamav3
cd exllamav3
# (Optional) switch to dev branch for latest in-progress features
git checkout dev

uv venv
uv sync --extra cu130
# add --extra examples and/or --extra eval for those extra dependencies
```

**Option 2 — Using `exllamav3` as a dependency from another project (`uv add`):**

```sh
# `uv add` works inside an existing project (a directory with a pyproject.toml).
# `uv init` creates one if you're starting a new project, if integrating into
# an existing project skip `uv init`.
uv init my-project
cd my-project

# local checkout
uv add 'path/to/exllamav3[cu130]'               # non-editable
uv add 'path/to/exllamav3[cu130]' --editable    # editable

# straight from GitHub
uv add 'git+https://github.com/turboderp-org/exllamav3.git[cu130]'                 # default branch
uv add 'git+https://github.com/turboderp-org/exllamav3.git[cu130]' --branch dev    # specific branch
```

**Option 3 — Bring your own `torch` and let `uv` pick the backend automatically:**

```sh
uv venv            # or: uv venv --python-preference only-managed
source .venv/bin/activate
uv pip install torch --torch-backend=auto
uv pip install .
```

`--torch-backend=auto` inspects your system and installs the matching PyTorch CUDA build; see [Automatic backend selection](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection).

**Option 4 — With `pip`:**

On Windows, you also need the `triton-windows` package (declared as a dependency in `pyproject.toml`); the attention, cache and recurrent kernels are Triton and ExLlamaV3 does not import without it.

```sh
# install a CUDA-enabled torch first so it matches your setup, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install .
```

</details>

<details>
<summary>Pinning a specific PyTorch version (optional)</summary>

#### Pinning a specific PyTorch version (optional)

The flavor extra picks the *index*, but by default torch resolves to the latest version on that
index that satisfies `>=2.6.0`. To pin a specific torch version while developing on `exllamav3`,
create a **"thin" project** that consumes your local checkout as an editable install and declares
the exact `torch` version itself. This keeps the pin out of the `exllamav3` pyproject, so
you can change the torch version freely without touching the repo.

```
my-exllamav3-dev/          # thin project (uv init)
├── pyproject.toml
└── src/                  # package sources (auto-generated)
```

In `pyproject.toml`:

```toml
[project]
name = "my-exllamav3-dev"
version = "0.1.0"
description = "Dev environment for exllamav3"
requires-python = ">=3.10.11"
dependencies = [
    "exllamav3[cu130]",   # select correct CUDA version
    "torch==2.13.0",      # pin the exact torch version you need
]

[tool.uv.sources]
exllamav3 = { path = "../exllamav3", editable = true }
```

Adjust `../exllamav3` to point at your local checkout, then a plain `uv sync` sets up an
environment with the correct PyTorch index (routed via the `cuXXX` extra),
the pinned version of `torch` from that index (as long as it exists), and an editable install of `exllamav3` so code
changes apply immediately. Switch CUDA flavors by changing the extra (`exllamav3[cu124]`,
`exllamav3[cu128]`, …) and/or the torch pin in the thin project.

Or, if you're installing torch manually with `uv pip install torch` (e.g. as in Option 3 above),
specify the version directly, e.g. `uv pip install "torch==2.11.0" --torch-backend=auto`.

</details>

After installing with one of the options above, you should be able to run the conversion, eval and 
example scripts from the main repo directory, e.g., `uv run python convert.py -i ...` or, for manual
installations once the venv is active, `python convert.py -i ...`

**Build environment variables**

- `MAX_JOBS`: by default ninja may launch too many processes and run out of system memory for 
compilation. Set this to a reasonable value like 4 in that case.
- `EXLLAMA_NOCOMPILE`: set to install the library without compiling the C++/CUDA extension. Torch
will build/load it at runtime instead.
- `EXLLAMA_EXT_LINEINFO`, `EXLLAMA_EXT_COMPRESS`: see [doc/env_vars.md](doc/env_vars.md).

## Examples

A number of example scripts are provided to showcase the features of the backend and generator. 
For instance, a versatile CLI chatbot:

<p align="center">
  <img src="doc/chatpy.png" width="640" alt="Llama 3.1 8B Instruct quantization benchmark across bits per weight">
</p>

```sh
python examples/chat.py -m <input_dir> -mode <prompt_mode>

# Wealth of options
python examples/chat.py -h
```

## Architecture support

| Model family                                     | HF architecture | Multimodal | Notes |
|--------------------------------------------------| --- | :---: | --- |
| **AFM**                                          | `ArceeForCausalLM` |  |  |
| **AfMoE**                                        | `AfmoeForCausalLM` |  |  |
| **Apertus**                                      | `ApertursForCausalLM` |  |  |
| **Command-R** etc.                               | `CohereForCausalLM` |  |  |
| **Command-A**, **Command-R+** etc.               | `Cohere2ForCausalLM` |  |  |
| **DeciLM**, **Nemotron**                         | `DeciLMForCausalLM` |  |  |
| **Deepseek V3**                                  | `DeepseekV3ForCausalLM` |  |  |
| **Deepseek V4**                                  | `DeepseekV4ForCausalLM` | ✓ |  |
| **dots.llm1**                                    | `Dots1ForCausalLM` |  | |
| **ERNIE 4.5**                                    | `Ernie4_5_ForCausalLM`<br>`Ernie4_5_MoeForCausalLM` |  |  |
| **EXAONE 4.0**                                   | `Exaone4ForCausalLM` |  |  |
| **Gemma 2**                                      | `Gemma2ForCausalLM` |  |  |
| **Gemma 3**                                      | `Gemma3ForCausalLM`<br>`Gemma3ForConditionalGeneration` | ✓ |  |
| **Gemma 4**                                      | `Gemma4ForConditionalGeneration`<br>`Gemma4UnifiedForConditionalGeneration` | ✓ | E2B/E4B unsupported |
| **GLM 4**, **GLM 4.6**, etc.                     | `Glm4ForCausalLM`<br>`Glm4MoeForCausalLM` |  |  |
| **GLM 4.1V**, **GLM 4.5V**                       | `Glm4vForConditionalGeneration`<br>`Glm4vMoeForConditionalGeneration` | ✓ |  |
| **GLM 4.7 Flash**                                | `Glm4MoeLiteForCausalLM` |  |  |
| **GLM 5.2**                                      | `GlmMoeDsaForCausalLM` |  |  |
| **GLM 5.3-Flash**                                | `Glm5NextForConditionalGeneration` | ✓ |  |
| **GPT-OSS**                                      | `GptOssForCausalLM` |  |  |
| **HyperCLOVAX**                                  | `HyperCLOVAXForCausalLM`<br>`HCXVisionV2ForCausalLM` | ✓ |  |
| **Hy3**                                          | `HYV3ForCausalLM` |  |  |
| **IQuest-Coder**                                 | `IQuestCoderForCausalLM` |  |  |
| **Kimi Linear**                                  | `KimiLinearForCausalLM` |  |  |
| **Ling 3.x**                                     | `BailingMoeV3ForCausalLM` |  |  |
| **Laguna 2.1**                                   | `LagunaForCausalLM` |  |  |
| **LFM 2.5**                                      | `Lfm2ForCausalLM`<br>`Lfm2MoeForCausalLM` |  |  |
| **Llama 1/2/3**,**3.1-Nemotron** etc.            | `LlamaForCausalLM` |  |  |
| **MiMo-RL**                                      | `MiMoForCausalLM` |  |  |
| **MiMo-V2.6-Flash**                              | `MiMoV2ForCausalLM` | ✓ | no audio |
| **MiniMax-M2**                                   | `MiniMaxM2ForCausalLM` |  |  |
| **Mistral**, **Ministral 3**, **Mistral-4** etc. | `MistralForCausalLM`<br>`Mistral3ForConditionalGeneration` | ✓ |  |
| **Mixtral**                                      | `MixtralForCausalLM` |  |  |
| **NemotronH, Nemotron-3 Nano/Super**              | `NemotronHForCausalLM` |  |  |
| **Olmo 3.1**                                     | `Olmo3ForCausalLM` |  |  |
| **Olmo-Hybrid**                                  | `OlmoHybridForCausalLM` |  |  |
| **Phi3**, **Phi4**                               | `Phi3ForCausalLM` |  |  |
| **Qwen 2**, **Qwen 2.5**, **Qwen 2.5 VL**        | `Qwen2ForCausalLM`<br>`Qwen2_5_VLForConditionalGeneration` | ✓ |  |
| **Qwen 3**                                       | `Qwen3ForCausalLM`<br>`Qwen3MoeForCausalLM` |  |  |
| **Qwen 3-Next**                                  | `Qwen3NextForCausalLM` |  |  |
| **Qwen 3-VL**                                    | `Qwen3VLForConditionalGeneration` | ✓ |  |
| **Qwen 3-VL MoE**                                | `Qwen3VLMoeForConditionalGeneration` | ✓ |  |
| **Qwen 3.5**                                     | `Qwen3_5ForConditionalGeneration` | ✓ |  |
| **Qwen 3.5 MoE**                                 | `Qwen3_5MoeForConditionalGeneration` | ✓ |  |
| **Qwen 3.8-Flash-Next**                          | `Qwen4ExpForConditionalGeneration` | ✓ |  |
| **Seed-OSS**                                     | `SeedOssForCausalLM` |  |  |
| **SmolLM**                                       | `SmolLM3ForCausalLM` |  |  |
| **SolarOpen**                                    | `SolarOpenForCausalLM` |  |  |
| **Step 3.5 Flash**                               | `Step3p5ForCausalLM` |  |  |
| **Step 3.7 Flash**                               | `Step3p7ForConditionalGeneration` | ✓ |  |

Always adding more, stay tuned.

## Conversion

To convert a model to EXL3 format, use:

```sh
# Convert model
python convert.py -i <input_dir> -o <output_dir> -w <working_dir> -b <bitrate>

# Resume an interrupted quant job
python convert.py -w <working_dir> -r

# More options
python convert.py -h
```

The working directory is temporary storage for state checkpoints and for storing quantized tensors 
until the converted model can be compiled. It should have enough free space to store an entire copy 
of the output model.

See the [conversion guide](doc/convert.md) for more information, or the 
[self-calibration guide](doc/optimize.md). 

## EXL3 quantization

EXL3 quantization is a streamlined variant of [**QTIP**](https://github.com/Cornell-RelaxML/qtip) from Cornell RelaxML. It aims to make
SOTA quantization available to users on consumer hardware. The conversion process is designed to be
simple and efficient and requires only an input model (in HF format) and a target bitrate. By
computing Hessians on the fly and thanks to a fused Viterbi kernel, the quantizer can convert a 
model in a single step, taking a couple of minutes for smaller models, up to a few hours for larger
ones (70B+) on a single high-end consumer GPU (see the [conversion guide](doc/convert.md)).

For more information, see the [**QTIP**](https://arxiv.org/abs/2406.11235) and [**QuIP#**](https://arxiv.org/abs/2402.04396) papers, as well as this 
[excellent writeup](https://www.together.ai/blog/even-better-even-faster-quantized-llms-with-qtip) on **QTIP** from together.ai.


## Community

You are always welcome to join the [ExLlama discord server](https://discord.gg/NSFwVuCjRq) ←🎮


### 🤗 Models on Hugging Face

Browse the [EXL3 model collection](https://huggingface.co/collections/turboderp/exl3-models-67f2dfe530f05cb9f596d21a) for quantized models. Also shout out to the following lovely
people:

- [ArtusDev](https://huggingface.co/ArtusDev)
- [MikeRoz](https://huggingface.co/MikeRoz)
- [MetaphoricalCode](https://huggingface.co/MetaphoricalCode)
- [Ready.Art](https://huggingface.co/ReadyArt)
- [isogen](https://huggingface.co/isogen/models)


## Acknowledgements

This project owes its existence to a wonderful community of FOSS developers and some very generous
supporters (🐈❤️!) The following projects in particular deserve a special mention:

- [TabbyAPI](https://github.com/theroyallab/tabbyAPI/)
- [PyTorch](https://github.com/pytorch/pytorch)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [QTIP](https://github.com/Cornell-RelaxML/qtip)
- [Transformers](https://github.com/huggingface/transformers)
- [Marlin](https://github.com/IST-DASLab/marlin)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) (chunked linear-attention prefill kernels, vendored under `exllamav3/vendor/fla`)

<p align="center">
  <img src="doc/cat.png" width="40" alt="">
</p>


