#!/usr/bin/env bash
# Run a batch on a rented card, from a clean GPU.
#
#     scripts/run_on_box.sh configs/probe-01-qwen3.8-27b.yaml 24
#
# Two things it does that running `run_batch.py` directly does not, both of
# which cost this project a restart each:
#
# * **Takes a lock.** Only one of these may run at a time, because the kill
#   below is indiscriminate — see the flock at the top.
# * **Frees the card first.** vLLM's engine core does not always exit with the
#   batch, and it renames itself to `VLLM::EngineCore` — so a pattern built from
#   the interpreter path misses exactly the process that matters. Asking the
#   driver who holds the card is the only reliable form. A leftover engine and a
#   card too small for the model produce the same startup error ("free memory is
#   less than desired GPU memory utilization"), so this is cheaper than reading
#   it.
# * **Puts the vLLM venv on PATH.** vLLM JIT-compiles kernels at engine start and
#   dies with a bare `FileNotFoundError: 'ninja'` if ninja is merely installed
#   rather than on PATH.
set -euo pipefail

# One at a time. The kill below frees the card for *this* run, so a second
# instance starting while the first is generating will kill the first's engine —
# which surfaces as `EngineDeadError` in the first run's log and looks like a
# vLLM fault rather than a second launch. That happened.
exec 9>/tmp/run_on_box.lock
flock -n 9 || { echo "another run holds /tmp/run_on_box.lock" >&2; exit 1; }

CONFIG=${1:?usage: run_on_box.sh CONFIG [BATCH_SIZE]}
BATCH=${2:-12}
CHECKOUT=${CHECKOUT:-$HOME/eip}
VLLM_VENV=${VLLM_VENV:-$HOME/vllm-venv}

nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 || true
sleep 5

cd "$CHECKOUT"
export PATH="$VLLM_VENV/bin:$HOME/.local/bin:$PATH"
exec sg docker -c "PATH=$PATH $VLLM_VENV/bin/python scripts/run_batch.py \
  --config $CONFIG --batch-size $BATCH --setup-workers 8"
