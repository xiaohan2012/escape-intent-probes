# Setup Log

How long it takes to bring a fresh machine to a working state. Recorded so a box
can be terminated mid-project without fear: anything here is the cost of getting
it back.

Timings are wall-clock on the machines actually used; network speed dominates,
so treat them as a floor rather than a guarantee.

**To rebuild a box, run the script rather than these steps by hand:**

```bash
curl -fsSL https://raw.githubusercontent.com/xiaohan2012/escape-intent-probes/main/scripts/setup_machine.sh | bash
```

It refuses to start on a non-x86_64 host or without a usable Docker daemon —
the two ways to waste an entire instance — fetches the model and the images
concurrently, runs both test suites, and writes its own timings to
`/tmp/setup_timings.txt`.

---

## Current box — Lambda Labs, H100 PCIe

**Why Lambda:** it hands out real VMs, so Docker works. This project needs the
model and the SWE-bench containers on one machine, and container-based GPU
clouds cannot provide the second half (see the vast.ai section below).

**Instance:** `ubuntu@209.20.158.60`, x86_64, 26 cores, 221 GB RAM, 993 GB disk,
1× NVIDIA H100 PCIe (81559 MiB). Docker preinstalled but the `ubuntu` user is
not in the `docker` group — `sudo usermod -aG docker ubuntu`, then reconnect.

| Step | Time |
|---|---|
| Install uv | **2 s** |
| Clone repo + `uv sync` | **4 s** |
| Download Qwen3-Coder-30B-A3B-Instruct (57 GB) | **65 s** |
| Pull 3 SWE-bench images | **70 s** |
| Install torch + transformers (`model` group) | **~60 s** |
| Unit test suite | **0.5 s** |
| Docker integration suite | **45 s** |
| Load Qwen3-Coder-30B-A3B onto the GPU | **~20 s** (531 shards) |

Model and images were fetched concurrently, so wall-clock from a bare instance
to a verified environment is **about two minutes**. Terminate the box when idle
without hesitation.

Two things that cost time the first round and should not cost it again:

* **Sync with `--frozen`.** A plain `uv sync --group ...` rewrites `uv.lock` on
  the box, and the dirty lock then blocks the next `git pull` with "commit your
  changes before you merge".
* **`swebench` is pinned to 2.x/3.x**, following the dataset schema rather than
  recency — see the note in `tasks.py`. Installing latest gives an API that
  cannot read these instances at all.

Lambda instances **cannot be paused** — stopping one destroys its data — so
anything worth keeping (trajectories above all) must be copied off, not left
on disk.

---

## Previous box — vast.ai, H200 (abandoned: no Docker)

**Instance:** `Intel Xeon Platinum 8480+`, 224 cores, 3 TB RAM, 500 GB disk,
1× NVIDIA H200 (143771 MiB). `HF_HOME=/workspace/.hf_home`.

| Step | Command | Time |
|---|---|---|
| Install uv | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | preinstalled (0 s) |
| Clone repo + `uv sync` | — | **3 s** |
| Download Qwen3-Coder-30B-A3B-Instruct | `uv run --with huggingface-hub hf download …` | **54 s** (57 GB) |

Total from fresh instance to model on disk: **about one minute.** Rebuilding
this box is cheap; terminate it without hesitation when idle.

### Blocker found: no Docker

The instance is an **unprivileged Docker container, not a VM**, so Docker itself
is unavailable. From `/etc/vast-agents-guide.md`:

> This is an **unprivileged Docker container**, not a VM. … you **cannot** load
> kernel modules, **run another container engine (no Docker-in-Docker)**, use
> kernel profilers, mount block devices, or change sysctls/cgroups.

Confirmed empirically: no `/var/run/docker.sock`, and `unshare -Ur` returns
`Operation not permitted`, so rootless Podman is out too.

`cap_sys_chroot` *is* granted, which would allow a workaround — pull image
layers over the registry HTTP API, unpack them, and `chroot` in. Rejected: that
loses network isolation (no netns, no `CAP_NET_ADMIN`), and the design requires
the sandbox to have no network so the planted exit is the only way out. Running
an agent we are deliberately nudging toward a boundary, with arbitrary `bash` and
live internet, is not an acceptable trade.

**Consequence:** the sandbox has to live on a second, Docker-capable machine.
`DockerSandbox` already routes every `docker` invocation through one place, so
pointing it at a remote host over SSH is a small change.

## Sandbox box — Docker-capable x86_64

Requirements: x86_64 (SWE-bench publishes x86_64 images only), Docker, ≥8 cores,
≥16 GB RAM, 100 GB disk, on-demand billing. No GPU.

A vast.ai **VM instance** would in principle work, but checking the full
on-demand listing found **no VM-capable H200, H100 or A100 host** — the only
VM-capable machine on offer was a single RTX A4000 16GB. Vast's large GPUs are
containers essentially without exception, so this is a platform property rather
than bad luck. Same for RunPod, whose pods are containers too. Among GPU clouds
only Lambda Labs hands out real VMs, at a price and without the ability to
pause.

So the sandbox goes on a plain VPS (Hetzner, DigitalOcean, Linode) with no GPU,
and `DockerSandbox(host=...)` routes docker over SSH from the GPU box. ARENA's
own provider comparison, for reference, is about cost and preinstalled
libraries and does not cover this distinction:

| Feature | VastAI | RunPod | Lambda Labs |
|---|---|---|---|
| Cost | Cheapest ($2/day when active) | Mid-range | Most expensive (~$0.50–0.80/hour) |
| Setup | Minimal pre-installed libraries | Some pre-installed | Most user-friendly, PyTorch pre-installed |
| Pausing | Easy pause, low idle costs | Can pause | Cannot pause without data loss |
| **Runs Docker** | **No — container** | **No — container** | **Yes — VM** |

| Step | Time |
|---|---|
| Install uv, clone repo, `uv sync` | **~5 s** |
| `docker pull` one SWE-bench instance image | *to measure* |
| Run the docker-marked test suite | **~26 s** for 16 tests |

Recorded from the first (now terminated) box: the image pull for
`django__django-12419` completed within a 110 s timeout; exact figure to be
measured on the replacement.

---

## 2026-09-12: 4×RTX A6000, a full bootstrap timed

Second box, this time four A6000s rather than an H100 (see D21 for why that
turned out to be the better card for this project, and what it retires).

```
arch                         x86_64
gpu                          4x NVIDIA RTX A6000, 49140 MiB each
cores                        56
RAM                          393 GB
disk free                    968G
uv                           1s
clone + sync (model + tasks) 17s
model download               40s
image pulls (3)              88s
```

About two and a half minutes to a verified environment, the model download and
the image pulls running concurrently because they come from different hosts.

Three things the script did not do and now does, each of which cost a manual
round trip:

* **Docker needed sudo.** The script detects this, adds the user to the
  `docker` group and exits, but group membership only applies to a new login —
  and an SSH `ControlMaster` connection will happily reuse the old session's
  credentials. `ssh -O exit <host>` before reconnecting.
* **vLLM gets its own venv.** vllm 0.29 pins torch 2.13+cu130 where the
  `model` group wants 2.14. Nothing needs vllm and transformers in one
  interpreter, so resolving them together is a fight with no prize.
* **`ninja` must be on PATH, not merely importable.** vLLM JIT-compiles kernels
  at engine start; without it the engine core dies with a bare
  `RuntimeError: Worker failed with error '[Errno 2] No such file or directory:
  'ninja''`, which says nothing about what is missing or why.

Engine startup, once installed: **58 s** for Qwen3-Coder-30B-A3B at
`tensor_parallel_size=2`, about 44 GB on each of two cards. Throughput numbers
are in `scripts/bench_batch.py`.

**A rollout batch, measured:** a round of 18 trajectories takes 631–713 s, so
**35–40 s per trajectory** at 25 steps, against 216 s on the HuggingFace path.
Of that round, the first ~2.5 minutes used to be serial container setup with
the GPU at zero — `--setup-workers` now threads it.
