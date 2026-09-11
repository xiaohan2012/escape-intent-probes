# Setup Log

How long it takes to bring a fresh machine to a working state. Recorded so a box
can be terminated mid-project without fear: anything here is the cost of getting
it back.

Timings are wall-clock on the machines actually used; network speed dominates,
so treat them as a floor rather than a guarantee.

---

## GPU box — vast.ai, H200

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

On vast.ai this means a **VM instance**, not the default container instance;
a plain VPS (Hetzner, DigitalOcean, Linode) is simpler and cheaper for this.

| Step | Time |
|---|---|
| Install uv, clone repo, `uv sync` | **~5 s** |
| `docker pull` one SWE-bench instance image | *to measure* |
| Run the docker-marked test suite | **~26 s** for 16 tests |

Recorded from the first (now terminated) box: the image pull for
`django__django-12419` completed within a 110 s timeout; exact figure to be
measured on the replacement.
