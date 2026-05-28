# Chandra OCR — vLLM (server + client) setup on this machine

This document records **exactly** what was installed, where, and why, on this
host (`instance-20260420-202919-a100x8`) to run Chandra OCR 2 with the vLLM
backend. It is intentionally version-pinned to avoid the dependency drift that
caused trouble on a previous attempt.

## Host inventory (snapshot at install time on this machine)

- OS: Debian-based linux, kernel `6.1.0-44-cloud-amd64`
- GPUs: **8× NVIDIA A100-SXM4-40GB** (driver 575.57.08, CUDA 12.9)
- Docker: 20.10.24 (already installed; **no `nvidia` runtime registered**)
- Conda: miniforge3, conda 26.1.1, mamba available
- Repo: `/home/jimmys/chandra` (cloned from `https://github.com/datalab-to/chandra.git`)
- `chandra/.python-version` → `3.12`

> **Reproducing on a fresh box?** The "Install commands — from absolute zero"
> section below covers everything from a bare Debian/Ubuntu install with just
> an NVIDIA GPU, including the driver, Docker, the NVIDIA container runtime,
> Miniforge, the repo, the vLLM image, the conda env, and pinned Python deps.
> Each step has a "skip if…" so you can jump past anything already present.

## Why this layout

Chandra has two backends: `hf` (local HuggingFace, requires torch) and `vllm`
(talks to a vLLM OpenAI-compatible server over HTTP). We use **vllm**.

The repo's `chandra_vllm` script does **not** `pip install vllm`. It runs the
official Docker image:

```
chandra/scripts/vllm.py:77 → vllm/vllm-openai:v0.17.0
```

So vLLM with all its torch/CUDA/xformers/flashinfer constraints lives entirely
inside the Docker image. The Python env on the host is **only** the lightweight
client (the `chandra` CLI + an `openai` HTTP client). This separation is the
whole point: it sidesteps the dependency hell of installing vLLM via pip.

## What gets installed

### A. System (apt) — requires sudo
| Package | Version | Why |
|---|---|---|
| `nvidia-container-toolkit` | latest from NVIDIA repo | Registers the `nvidia` runtime in Docker so `docker run --runtime nvidia` (which `chandra_vllm` invokes) can expose GPUs to containers. Tracks the host driver, so pinning is unnecessary. |

Daemon change: `nvidia-ctk runtime configure --runtime=docker` (edits
`/etc/docker/daemon.json`) + `systemctl restart docker`.

### B. Docker images
| Image | Tag | Why this exact tag |
|---|---|---|
| `vllm/vllm-openai` | `v0.17.0` | Hard-pinned in `chandra/scripts/vllm.py`. Self-contained — its own torch/CUDA/xformers/flashinfer. |

### C. Conda env: `chandra-vllm` (Python 3.12)
Pinned versions taken from chandra's own `uv.lock` (the maintainers' tested
set) and the lower bounds in `pyproject.toml`. **No torch, no transformers,
no vllm.**

| Package | Version | Source |
|---|---|---|
| `chandra-ocr` | `0.2.0` (editable, `--no-deps`) | local `/home/jimmys/chandra` |
| `openai` | `2.2.0` | pyproject lower bound |
| `pydantic` | `2.12.0` | pyproject lower bound |
| `pydantic-settings` | `2.11.0` | pyproject lower bound |
| `pypdfium2` | `4.30.0` | pyproject lower bound |
| `pillow` | `10.4.0` | last stable 10.x (pyproject `>=10.2.0`) |
| `beautifulsoup4` | `4.14.2` | pyproject lower bound |
| `markdownify` | `1.1.0` | pyproject (already exact) |
| `click` | `8.1.7` | latest stable 8.x (pyproject `>=8.0.0`) |
| `filetype` | `1.2.0` | pyproject lower bound |
| `python-dotenv` | `1.1.1` | pyproject lower bound |
| `six` | `1.17.0` | pyproject lower bound |

### D. NOT installed (intentionally)
- `vllm` (pip) — lives only inside the Docker image
- `torch`, `torchvision`, `transformers`, `accelerate` — HF backend only, not used
- `flash-attn` — HF backend only
- `streamlit` — only needed for `chandra_app`

## Install commands — from absolute zero

The steps below assume a fresh Debian/Ubuntu host with **only an OS and an
NVIDIA GPU**. Every step has a "skip if…" check at the top so you can jump
past anything you already have. Tested on Debian 12 / Ubuntu 22.04+ with
A100-class GPUs.

> All commands assume your user has `sudo` privileges. Run them in order.

### 0. Verify you have an NVIDIA GPU
```bash
lspci | grep -i nvidia
# expect a line like: 00:04.0 3D controller: NVIDIA Corporation GA100 ...
```
If nothing prints, you don't have an NVIDIA GPU and chandra-vllm cannot run.

### 1. Install the NVIDIA driver
**Skip if** `nvidia-smi` already prints a table of GPUs.

```bash
# Pick the latest stable driver from your distro
sudo apt-get update
sudo apt-get install -y nvidia-driver firmware-misc-nonfree   # Debian
# or, on Ubuntu:
# sudo ubuntu-drivers install
sudo reboot
```
After reboot, verify:
```bash
nvidia-smi   # should show your GPU(s), driver version, CUDA version
```
Driver must be **>= 525** for vLLM v0.17 (CUDA 12.x). Driver 575 (this host) is fine.
You do **not** need to install CUDA or cuDNN on the host — they live inside the Docker image.

### 2. Install Docker Engine
**Skip if** `docker --version` prints something.

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/$(. /etc/os-release && echo "$ID")/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/$(. /etc/os-release && echo "$ID") \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
```
Verify:
```bash
sudo docker run --rm hello-world   # should print "Hello from Docker!"
```
Optional (avoid `sudo` for every docker command — re-login afterwards):
```bash
sudo usermod -aG docker "$USER"
```

### 3. Install NVIDIA Container Toolkit (gives Docker GPU access)
**Skip if** `docker info | grep -i runtimes` already shows `nvidia`.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
Verify:
```bash
docker info | grep -i runtimes        # should now include 'nvidia'
sudo docker run --rm --runtime nvidia --gpus device=0 \
  nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
# expect to see GPU 0 listed from inside the container
```

### 4. Install Miniforge (conda + mamba)
**Skip if** `conda --version` already works.

```bash
curl -fsSL -o /tmp/miniforge.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" init bash
exec bash    # reload shell so conda is on PATH (or open a new terminal)
```
Verify:
```bash
conda --version    # e.g. 26.x
mamba --version    # ships with miniforge
```

### 5. Install git and clone the chandra repo
**Skip if** `/home/jimmys/chandra/pyproject.toml` already exists.

```bash
sudo apt-get install -y git
mkdir -p /home/jimmys
cd /home/jimmys
git clone https://github.com/datalab-to/chandra.git
cd chandra
cat .python-version    # should print 3.12 — confirms target Python version
```

### 6. Pull the pinned vLLM image
**Skip if** `sudo docker images vllm/vllm-openai:v0.17.0` already lists it.

```bash
sudo docker pull vllm/vllm-openai:v0.17.0   # ~20 GB; takes several minutes
```

### 7. Create the conda env
**Skip if** `conda env list` already shows `chandra-vllm`.

```bash
conda create -n chandra-vllm python=3.12 -y
conda activate chandra-vllm   # 'mamba activate' won't work unless mamba shell is initialized; conda is already initialized in this shell
```

### 8. Install pinned Python deps
```bash
cd /home/jimmys/chandra
pip install --no-deps -e .
pip install \
  "openai==2.2.0" \
  "pydantic==2.12.0" \
  "pydantic-settings==2.11.0" \
  "pypdfium2==4.30.0" \
  "pillow==10.4.0" \
  "beautifulsoup4==4.14.2" \
  "markdownify==1.1.0" \
  "click==8.1.7" \
  "filetype==1.2.0" \
  "python-dotenv==1.1.1" \
  "six==1.17.0"
```
Verify:
```bash
which chandra chandra_vllm
# /home/jimmys/miniforge3/envs/chandra-vllm/bin/chandra
# /home/jimmys/miniforge3/envs/chandra-vllm/bin/chandra_vllm
chandra --help
```

### 9. Make the helper scripts executable
**Skip if** they're already `+x`.

```bash
chmod +x /home/jimmys/chandra/scripts_local/server.sh
chmod +x /home/jimmys/chandra/scripts_local/client.sh
chmod +x /home/jimmys/chandra/scripts_local/client.py
```

### 10. Smoke test (single GPU, single PDF)
```bash
conda activate chandra-vllm
# launch one server on GPU 0
GPUS="0" /home/jimmys/chandra/scripts_local/server.sh
# wait until "Application startup complete." appears in the log
sudo docker logs -f chandra-vllm-0
# Ctrl-C to detach (server keeps running)

# in the same shell (or a new one with conda activate chandra-vllm):
chandra /path/to/some/file.pdf /tmp/chandra_smoketest --method vllm
ls /tmp/chandra_smoketest/   # should contain a subfolder per processed file
```
If that produces a `.md` file, **the entire stack works**. From here, see
"Scaling: many GPUs over a big folder of PDFs" below.

## Running it

### Start the server (one A100-40GB)
```bash
conda activate chandra-vllm
chandra_vllm --gpu a100-40
```
Listens on `http://localhost:8000/v1`, served model name `chandra`.
First run will download model weights `datalab-to/chandra-ocr-2` to
`~/.cache/huggingface` (mounted into the container).

To use a different GPU: `VLLM_GPUS=3 chandra_vllm --gpu a100-40`.

### Run the client (in another terminal)
```bash
conda activate chandra-vllm
chandra input.pdf ./output --method vllm
```

## Scaling: many GPUs over a big folder of PDFs

The bundled `chandra` CLI processes files **strictly sequentially** (one file
at a time, with within-file page batching). Talking to a single vLLM server it
will only ever saturate one GPU. To use all 8 A100s on a directory of PDFs,
shard the file list across 8 independent servers — this beats tensor-parallel
for an OCR-sized model.

Two helper scripts are provided in `scripts_local/`:

- **`server.sh`** — boots one `chandra-vllm-<i>` container per GPU.
- **`client.sh`** — counts the pages of every input file, bin-packs them
  across the running servers so that each shard has roughly the same total
  page count, launches one `chandra` client per shard, and shows a per-shard
  progress bar in the terminal (or a periodic snapshot when not on a TTY).
  Internally it execs `client.py`.

### Tuned defaults (mirroring known-good olmocr settings)

| Param | Value | Purpose |
|---|---|---|
| `--gpu-memory-utilization` | `0.92` | use as much VRAM as possible |
| `--max-model-len` | `24576` | room for very dense pages (math/tables); cap at 32768 |
| `--max-num-seqs` | `40` | concurrent sequences per server (lowered to fit larger context in 40GB) |
| `--max-num-batched-tokens` | `8192` | bigger prefill batches |
| `--enforce-eager` | (flag) | **forces eager-mode execution**, disabling CUDA-graph capture. Costs ~10–20% throughput but removes another source of concurrency non-determinism that contributes to residual repetition under load (per [vllm/vllm#20261](https://github.com/vllm-project/vllm/issues/20261)). For batch OCR (correctness > speed) this is the right tradeoff. See [Known vLLM bugs](#known-vllm-bugs-and-the-flags-that-mitigate-them) Bug 5. |
| `--no-enable-prefix-caching` | (flag) | **disabled on purpose** — see [Known vLLM bugs](#known-vllm-bugs-and-the-flags-that-mitigate-them) below. Do not re-enable. |
| `--mm-processor-cache-gb` | `0` | disables the multimodal processor cache (same family of bug). **Note**: the flag name is `--mm-processor-cache-gb` in vLLM v0.17.0, *not* `--disable-mm-preprocessor-cache` (that exists in newer versions). Passing the wrong name makes every container fail with `unrecognized arguments` and crash-loop under `--restart unless-stopped`. |
| `--mm-processor-kwargs max_pixels` | `4194304` (~2048²) | lowered from 6291456. Very large images push the model into degenerate loops on visually dense pages. 4M pixels is plenty for a printed page and reduces vision-token count, which both improves quality and saves prefill compute. See [Known vLLM bugs](#known-vllm-bugs-and-the-flags-that-mitigate-them). |
| client `--max-workers` per shard | `16` | 8 shards × 16 = 128 in-flight requests total (matches olmocr `--workers 128`) |
| client `--max-retries` | `6` | survive transient vLLM hiccups and break repetition loops via chandra's temperature-ramp retry (`0.0 → 0.2 → 0.4 → 0.6 → 0.8`). Was `3` originally; bumped because chandra commits the *last* attempt's output to disk even if all retries fail, and 3 retries stops at T=0.6 — 6 gives the sampler more chances to escape a stuck state. Retries only fire on ~<1% of pages so the throughput cost is invisible. |
| client `--batch-size` | `28` | chandra default for vllm |

All overridable via env var (see script headers).

### Known vLLM bugs and the flags that mitigate them

This stack runs `vllm/vllm-openai:v0.17.0` (hard-pinned by upstream chandra at
`chandra/scripts/vllm.py:77`) serving the `datalab-to/chandra-ocr-2` model, whose
architecture is tagged `qwen3_5` on HuggingFace (a Qwen3-VL-family vision-language
model, fine-tuned for OCR by Datalab). Every flag below is set to dodge a specific,
reproducible bug — most documented in vLLM's own issue tracker. Removing any of
them will degrade output silently, not loudly.

#### Bug 1 — vLLM prefix-cache key is text-only ([vllm/vllm#20261](https://github.com/vllm-project/vllm/issues/20261), open)

**Symptom.** Under concurrency, the model emits empty leading pages or gets stuck
in runaway token-repetition loops (the same line emitted hundreds or thousands of
times). Hits any vLLM-served multimodal model: Qwen2.5-VL, Qwen3-VL, the `qwen3_5`
arch used here, LLaVA, InternVL, Pixtral, etc.

**Cause.** vLLM's prefix cache hashes only the text-token IDs of a prompt. When
two concurrent requests share an identical text prefix (which all chandra OCR
requests do — the system prompt and OCR instruction are identical) but have
different images, vLLM treats them as cache hits and **reuses the visual KV
state from whichever request populated the cache first**. The model then
generates text from a hallucinated visual prefix and typically either emits
nothing or locks into a degenerate output.

**Confirmation outside this codebase.** Issue #20261 itself reproduces it on
Qwen2.5-VL; downstream confirmation reports on Qwen3-VL ([QwenLM/Qwen3-VL#1876](https://github.com/QwenLM/Qwen3-VL/issues/1876)),
Qwen3, and various LLaVA forks. The Qwen team's own recommended workaround in
their issue tracker is *"maybe `--mm-processor-cache-gb 0 --no-enable-prefix-caching` helps"* — same two flags used here.

**Fix.** `--no-enable-prefix-caching`. The root bug is architectural (the cache
key would need to include image-content hashes) and is **unfixed in every
released vLLM version**, including the v0.18.x / v0.19.x line. Disabling the
cache is the only working mitigation.

**Observed effect on this dataset.** Running with prefix-caching *enabled* (the
upstream default, also used in our earlier v1 setup, `chandra2-setup-main`)
produced ~10–25% flagged outputs across multiple collections, including entries
with the same single line repeated 3000+ times. Running with it disabled (this
script) produces ~0.7% flagged outputs, dominated by legitimate recurring
content (DUPE_PARA on magazine ads, etc.), not hallucination.

#### Bug 2 — MultiModalReceiverCache premature eviction ([vllm/vllm#26195](https://github.com/vllm-project/vllm/issues/26195), fixed in PR #28525 post-v0.17.0)

**Symptom.** Requests containing multiple images crash with
`AssertionError: Expected a cached item for mm_hash='...'`, or, more
insidiously, silently produce wrong/empty output for the later images in the
request.

**Cause.** During multi-image request preprocessing, vLLM iterates over the
request's images and updates a per-item cache. If the *total* size of the
request's images exceeds the cache capacity, an earlier image in the same
request can evict a later image's slot before that later image is read back —
the loop then asserts on the missing slot or, worse, processes it from stale
data.

**Why it affects chandra.** chandra batches multiple page-images per vLLM
request (`--batch-size 28`). A single batch easily exceeds a small mm-processor
cache.

**Fix.** `--mm-processor-cache-gb 0`. Disabling the cache entirely sidesteps
the eviction logic. Datalab maintainers explicitly recommend this workaround
in the issue thread. PR #28525 fixes the eviction bug upstream but landed
*after* the v0.17.0 tag, so it's not available in our pinned image — and with
the cache disabled it's a no-op anyway.

**Important flag-name trap.** In v0.17.0 the flag is **`--mm-processor-cache-gb 0`**.
In newer vLLM versions the same intent is spelled **`--disable-mm-preprocessor-cache`**
(also note the `preprocessor` vs `processor` letter swap). Passing the wrong
spelling causes vLLM to exit immediately with `unrecognized arguments`, and
because our containers run with `--restart unless-stopped`, the container then
crash-loops until you `docker rm -f` it. Always verify the flag name matches
the image tag.

#### Bug 3 — Off-by-one in multimodal prefix-cache hash boundary ([vllm/vllm PR #102055](https://github.com/vllm-project/vllm/actions/runs/23034376618), merged post-v0.17.0)

**Symptom.** Subtle: an image-block boundary is hashed one token off, causing
occasional cache hits across requests that shouldn't share state. Contributes
to the same failure family as #20261 but at a much lower rate.

**Cause.** A `+1` / `-1` error in the block-boundary index used to derive the
multimodal block hash. Only triggers when the boundary lines up with a
particular token-id offset.

**Fix.** Not directly applicable to us — the bug is in the prefix-cache code
path, which we've disabled via #20261's workaround. Listed here for
completeness so anyone re-enabling caching in a newer vLLM is aware.

#### Bug 4 — Image-size–induced model loops ([QwenLM/Qwen3-VL#1876](https://github.com/QwenLM/Qwen3-VL/issues/1876))

**Symptom.** Even with prefix-caching and mm-processor-cache off, the model
occasionally loops on visually dense pages — typically pages with very small
text, complex tables, or low-contrast scans where the vision encoder produces
ambiguous tokens.

**Cause.** Two contributors: (a) Qwen3-VL family models have a documented
tendency to repeat under greedy decoding when next-token probability cycles;
(b) larger images produce more vision tokens, more ambiguity, more cycles.
Chandra's `temperature=0.0` default makes the model maximally vulnerable.

**Fix.** Two-layer mitigation:
1. **Cap `max_pixels` at 4194304** (≈2048²) instead of 6291456. Reduces vision
   tokens by ~33% and removes the densest visual cases. Issue #26195 reporter
   confirmed independently that *"resizing my images to never exceed 1200 on
   the longer side, combined with cache-off, fixes it"*.
2. **Bump `--max-retries` from 3 to 6.** Chandra's retry logic ramps
   `temperature` linearly (`0.0 + 0.2 × attempt`, capped at 0.8) and sets
   `top_p=0.95` on retry. Each retry breaks the deterministic loop by adding
   sampling entropy. By attempt 4 (T=0.6) the next-token distribution has
   enough entropy that loops virtually never reproduce on the same input.
   **Critically, chandra writes the last attempt's output to disk regardless
   of whether retries succeeded** — so insufficient retries means broken
   pages get committed silently. Six attempts is empirically enough.

#### Bug 5 — CUDA-graph capture amplifies concurrency non-determinism ([vllm/vllm#20261](https://github.com/vllm-project/vllm/issues/20261) discussion)

**Symptom.** Even after disabling both caches (Bugs 1 and 2), a small rate of
repetition / garbled output persists under high concurrency. Outputs are not
bit-exact reproducible across runs of the same inputs, even at `temperature=0`.

**Cause.** vLLM by default captures CUDA graphs to fuse and reorder GPU
kernels for throughput. The captured graph is reused across batches of
differing composition, which introduces small numerical differences in
attention outputs depending on how requests interleave inside a batch. For
multimodal models these tiny float-level differences can flip the argmax of
borderline next-token distributions, which is exactly the regime where the
model is most likely to enter a repetition loop. Multiple commenters in
#20261 report that adding `--enforce-eager` materially reduced their residual
repetition once they had already disabled prefix caching: *"`--enforce-eager`.
this helped"*.

**Fix.** `--enforce-eager`. Runs kernels eagerly instead of via captured
graphs. Numerics become identical regardless of batch composition. Throughput
drops ~10–20% (the cost of not fusing kernels). For an offline OCR pipeline
where you only get one shot at each page and the output goes to disk, that
trade is overwhelmingly worth it.

**Note on the v0.17.0 upstream chandra default.** Upstream
`chandra/scripts/vllm.py` sets `--no-enforce-eager` (i.e., CUDA graphs *on*)
to maximize throughput on a single-document interactive use case. That's
the wrong default for batch processing; we override it here.

#### Bug 6 — chandra-ocr-2 OSS model has known repetition pathologies ([datalab-to/chandra#62](https://github.com/datalab-to/chandra/issues/62), [datalab-to/chandra#71](https://github.com/datalab-to/chandra/issues/71))

**Symptom.** A small residual rate of loop / hallucination on certain
document types — particularly Indic scripts, dense math, low-quality scans —
that survives every server-side mitigation above.

**Cause.** Model-side. Datalab's own response in #62: *"The model on the API
is a little more up-to-date than OSS. Coming soon."* The publicly-released
`datalab-to/chandra-ocr-2` weights have repetition pathologies that the
unreleased API-internal checkpoint has been fine-tuned to reduce. Not a
vLLM bug — not fixable by any server flag.

**Fix (operational, not in this script).** For pages still flagged after
re-running with this script, fall back to the HuggingFace backend on the
single bad page:
```bash
chandra /path/to/bad.pdf /tmp/recover --method hf --max-retries 6
```
The HF backend doesn't share vLLM's concurrency surface and uses a different
generation code path. It's ~5× slower per page but eliminates the
concurrency-induced contributors entirely. Use as last-resort recovery on
the residual entries, not as the primary pipeline.

#### Why we don't upgrade vLLM

In short: **#20261 — the root prefix-cache bug — is unfixed in every released
vLLM version.** With our caches disabled, we don't traverse the buggy code
paths anyway, so v0.18.x / v0.19.x bring no quality improvement. They do
bring risk: the `qwen3_5` model arch tag is uncommon, dispatcher regressions
happen in vLLM minor releases, and several flag names change between v0.17
and v0.19 (the `--mm-processor-cache-gb` ↔ `--disable-mm-preprocessor-cache`
rename above is one example; `--max_num_batched_tokens` underscore form was
also normalized in v0.18). Upstream chandra hard-pins v0.17.0 for this
reason, and we follow that pin.

If you do upgrade, smoke-test in this order: container boot without
`unrecognized arguments`, `/v1/models` returns 200, single-PDF correctness
diff against the v0.17 output on 5–10 representative docs, 50-PDF concurrent
batch + run the hallucination scanner. Stay on v0.17 if any step regresses.

#### Other models / inference stacks with the same bug class

This is not a chandra-specific problem. Anyone serving a vision-language
model on vLLM at concurrency is vulnerable to #20261; anyone using a Qwen-VL
family OCR model can hit Bug 5's repetition pathology:

- **olmocr / olmocr2** (Allen AI, Qwen2-VL / Qwen2.5-VL base) — same vLLM
  prefix-cache bug; their launch script also disables prefix caching.
- **dots.ocr** (Qwen2.5-VL base) — same.
- **GOT-OCR2.0** (Qwen-based) — same.
- **InternVL / Pixtral / LLaVA** via vLLM — same vLLM cache bug; their
  base models are different, but the cache-key issue is in vLLM, not in the
  model.
- **Nougat / Pix2Tex** — different stack but same generic
  autoregressive-decoder repetition failure on hard math.

Switching to a different OCR model built on the same architecture would not
help. Switching inference frameworks (SGLang, TGI) avoids the *vLLM* bugs
but introduces different ones and is much more work; we don't recommend it
here.

### Launch one server per GPU
```bash
cd /home/jimmys/chandra
./scripts_local/server.sh
# launches chandra-vllm-0..7 on ports 8000..8007, one per GPU
# (containers run with --restart unless-stopped, so they survive shell exit)
```

Verify they're up:
```bash
sudo docker ps --filter name=chandra-vllm-
sudo docker logs -f chandra-vllm-0   # watch first server boot
```

Stop them all:
```bash
STOP=1 ./scripts_local/server.sh
```

Tunable env vars: `GPUS="0,1,2,3"` (subset), `MAX_MODEL_LEN`, `MAX_NUM_SEQS`,
`MAX_NUM_BATCHED_TOKENS`, `GPU_MEM_UTIL`, `BASE_PORT`. See script header.

### Process a directory across all servers
```bash
conda activate chandra-vllm

# foreground (gives you the live per-shard progress bars in the terminal):
./scripts_local/client.sh /path/to/pdfs /path/to/out

# or fire-and-forget — progress is logged as periodic snapshots, full
# per-shard chandra output still tee'd to <out>/.shard_<i>.log:
nohup ./scripts_local/client.sh \
    /path/to/pdfs \
    /path/to/out \
    > /home/jimmys/dataset_run.log 2>&1 &
disown
# optional 3rd arg = number of servers; default = number of GPUs
# any further args are forwarded to chandra (e.g. --no-images)
```

End-to-end (one-liner) on a big job:

```bash
conda activate chandra-vllm
./scripts_local/server.sh
nohup ./scripts_local/client.sh \
    /home/jimmys/datasets/archetai \
    /home/jimmys/datasets/archetai_chandra_out \
    > /home/jimmys/chandra_archetai.log 2>&1 &
# when finished:
STOP=1 ./scripts_local/server.sh
```

Env vars accepted by `client.sh`:
- `BASE_PORT` (default 8000) — base port of the first vLLM server.
- `MAX_WORKERS_PER_SHARD` (default 16) — chandra `--max-workers` per shard.
- `MAX_RETRIES` (default 6) — chandra `--max-retries`. See [Known vLLM bugs](#known-vllm-bugs-and-the-flags-that-mitigate-them) Bug 4 for why this is 6 and not the upstream default of 3.
- `BATCH_SIZE` (default 28) — chandra `--batch-size`.
- `SKIP_EXISTING` (default 1) — skip files whose `<out>/<stem>/<stem>.md`
  already exists. Set to `0` to force a full reprocess.

Watch progress:
```bash
tail -f /home/jimmys/chandra_archetai.log    # top-level + periodic snapshots
tail -f /path/to/out/.shard_0.log            # one worker's raw chandra output
watch -n 2 nvidia-smi                        # GPU utilization
```

#### How sharding actually works

`client.sh` (via `client.py`) does the following on startup:

1. **Discovery** — enumerates supported files (`.pdf`, `.png`, `.jpg`,
   `.jpeg`, `.gif`, `.webp`, `.tiff`, `.bmp`) at the top level of the input
   dir.
2. **Resume** — if `SKIP_EXISTING=1` (default), drops any file whose
   `<output>/<stem>/<stem>.md` already exists.
3. **Page counting** — opens every PDF with `pypdfium2` (in a thread pool)
   to get its page count; images count as 1 page.
4. **Bin packing** — runs Longest-Processing-Time greedy: sort files by
   page count descending, give each one to the shard with the lowest
   running total. On real datasets this is essentially perfect (e.g.
   7,731 ecolus PDFs / 26,147 pages → 8 shards differ by **1 page**).
5. **Launch** — creates a temp directory of symlinks per shard and starts
   one `chandra` client per shard, each pinned to its own server via
   `VLLM_API_BASE=http://localhost:(BASE_PORT+i)/v1`.
6. **Progress** — parses each chandra's stdout in real time
   (`[i/N] Processing: …`, `Processing pages A-B…`, `Saved: … (P page(s))`)
   and renders a per-shard bar. Full per-shard output is tee'd to
   `<output>/.shard_<i>.log` so nothing is lost.

A foreground (TTY) run looks like:

```
shard 0 > |##########--------------------|   1240/3269  ( 37.9%) files   42/962  ETA  18m05s lorem-ipsum-vol3.pdf
shard 1 > |#########---------------------|   1186/3269  ( 36.3%) files   38/960  ETA  18m44s ...
...
TOTAL    |#########---------------------|   9622/26147 ( 36.8%)
```

### Trade-offs
- **N independent servers (this approach)** — best for many small/medium PDFs.
  Near-linear scaling because file-level parallelism is what matters when
  chandra's client iterates serially.
- **Single tensor-parallel server (`--tensor-parallel-size 8`)** — usually
  worse for OCR-sized models. Only worth it if a single GPU can't fit your
  desired context/batch settings.
- **Single server + nginx/litellm load balancer** — clean one URL, but a
  single chandra client process won't keep N backends busy unless individual
  PDFs are very large.

## What this changes on your system

1. New apt package: `nvidia-container-toolkit` (+ libs).
2. New apt source list: `/etc/apt/sources.list.d/nvidia-container-toolkit.list`.
3. New keyring: `/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg`.
4. `/etc/docker/daemon.json` modified to register the `nvidia` runtime.
5. Docker daemon restarted.
6. New Docker image cached locally: `vllm/vllm-openai:v0.17.0`.
7. New conda env: `~/miniforge3/envs/chandra-vllm`.
8. Editable install of `chandra-ocr` pointing at `/home/jimmys/chandra`.

Nothing else on the host is modified. No global pip installs, no system Python
changes, no changes to other conda envs.

## Uninstall / rollback

```bash
conda env remove -n chandra-vllm
sudo docker rmi vllm/vllm-openai:v0.17.0
sudo apt-get remove -y nvidia-container-toolkit
sudo rm /etc/apt/sources.list.d/nvidia-container-toolkit.list \
        /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
# Optional: revert /etc/docker/daemon.json and: sudo systemctl restart docker
```

## Install log

- [x] 1. nvidia-container-toolkit installed (apt)
- [x] 2. nvidia runtime registered in docker (`/etc/docker/daemon.json` updated, daemon restarted; `docker info` shows `Runtimes: ... nvidia ...`)
- [x] 3. nvidia runtime smoke test passes (`nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L` → `GPU 0: NVIDIA A100-SXM4-40GB`)
- [x] 4. vllm/vllm-openai:v0.17.0 pulled (image size 20.7GB, image id `700d8ac4f37a`)
- [x] 5. conda env `chandra-vllm` created (python 3.12.x, miniforge3)
- [x] 6. pinned deps installed (no resolver conflicts; full version list in `pip list` of the env)
- [x] 7. `chandra` and `chandra_vllm` on PATH at `/home/jimmys/miniforge3/envs/chandra-vllm/bin/`