#!/usr/bin/env bash
# Bring a fresh GPU box to a working state, and time every step.
#
# Rented machines get terminated — deliberately, when idle, or not — so this has
# to be cheap and repeatable rather than a thing anyone does by hand. Run it on
# the box:
#
#     curl -fsSL https://raw.githubusercontent.com/xiaohan2012/escape-intent-probes/main/scripts/setup_machine.sh | bash
#
# Requirements the script checks before doing any work, because getting either
# wrong wastes the whole instance:
#   * x86_64 — SWE-bench publishes x86_64 images only. ARM hosts (NVIDIA Grace:
#     GH200, GB200) cannot run them without emulation.
#   * a usable Docker daemon — rented GPU instances are frequently unprivileged
#     containers with no Docker-in-Docker, which rules them out entirely.
#
# Timings land in /tmp/setup_timings.txt; copy them into docs/setup-log.md.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
REPO="${REPO:-https://github.com/xiaohan2012/escape-intent-probes}"
CHECKOUT="${CHECKOUT:-$HOME/eip}"
VLLM_VENV="${VLLM_VENV:-$HOME/vllm-venv}"
TIMINGS=/tmp/setup_timings.txt

# One image per instance; SWE-bench replaces `__` with `_1776_` because `__` is
# not legal in a tag. Same-repo instances share a `sweb.env.*` layer, so the
# total is well under 2 GB each.
IMAGES=(
  django_1776_django-12419
  sympy_1776_sympy-20916
  pytest-dev_1776_pytest-5809
)

: >"$TIMINGS"
log() { printf '%-28s %s\n' "$1" "$2" | tee -a "$TIMINGS"; }
timed() { local label=$1 start; shift; start=$(date +%s); "$@"; log "$label" "$(($(date +%s) - start))s"; }

# --- preflight ---------------------------------------------------------------

arch=$(uname -m)
if [[ $arch != x86_64 ]]; then
  echo "FATAL: architecture is $arch, need x86_64 (SWE-bench images are x86_64 only)." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  if sudo -n docker info >/dev/null 2>&1; then
    echo "Docker needs sudo. Adding $USER to the docker group; log out and back in." >&2
    sudo usermod -aG docker "$USER"
    exit 1
  fi
  echo "FATAL: no usable Docker daemon. If this is an unprivileged container" >&2
  echo "(vast.ai, RunPod), Docker-in-Docker is unavailable — use a VM instead." >&2
  exit 1
fi

log "arch" "$arch"
log "gpu" "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none)"
log "cores" "$(nproc)"
log "disk free" "$(df -h / | awk 'NR==2 {print $4}')"

# --- install -----------------------------------------------------------------

install_uv() {
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
}
timed "uv" install_uv
export PATH="$HOME/.local/bin:$PATH"

clone_and_sync() {
  [[ -d $CHECKOUT/.git ]] || git clone -q "$REPO" "$CHECKOUT"
  # `--frozen`: a plain sync can rewrite uv.lock on the box, and a dirty lock
  # then blocks the next `git pull` with "commit your changes before you merge".
  cd "$CHECKOUT" && git pull -q && uv sync -q --frozen --group model --group tasks
}
timed "clone + sync (model + tasks)" clone_and_sync

# vLLM gets its own venv rather than a dependency group in the main one. It pins
# its own torch (0.29 wants 2.13+cu130 where the `model` group wants 2.14), and
# resolving both together is a fight with no prize: nothing needs vllm and
# transformers in one interpreter. `ninja` is not optional — vLLM JIT-compiles
# kernels at engine start and fails with a bare `[Errno 2] ... 'ninja'`
# otherwise — and it has to be on PATH, not merely importable.
install_vllm() {
  uv venv --python 3.12 "$VLLM_VENV" >/dev/null 2>&1
  uv pip install -q --python "$VLLM_VENV" vllm ninja >/dev/null
  uv pip install -q --python "$VLLM_VENV" -e "$CHECKOUT" datasets "swebench>=2.1,<4" >/dev/null
}
timed "vllm venv" install_vllm

# The model and the images come from different hosts, so fetch them at once.
fetch_model() { uv run --quiet --with huggingface-hub hf download "$MODEL" >/tmp/model_download.log 2>&1; }
fetch_images() {
  for image in "${IMAGES[@]}"; do
    docker pull -q "swebench/sweb.eval.x86_64.${image}:latest" >/dev/null
  done
}

model_start=$(date +%s)
fetch_model &
model_pid=$!
images_start=$(date +%s)
fetch_images &
images_pid=$!

wait $model_pid && log "model download" "$(($(date +%s) - model_start))s"
wait $images_pid && log "image pulls (${#IMAGES[@]})" "$(($(date +%s) - images_start))s"

# --- verify ------------------------------------------------------------------

cd "$CHECKOUT"
timed "test suite" uv run --quiet pytest -q
timed "docker test suite" uv run --quiet pytest -m docker -q

echo
echo "Timings written to $TIMINGS:"
cat "$TIMINGS"
echo
echo "Rollouts run in the vllm venv, with its bin on PATH so ninja is found:"
echo "  PATH=$VLLM_VENV/bin:\$PATH $VLLM_VENV/bin/python scripts/run_batch.py ..."
