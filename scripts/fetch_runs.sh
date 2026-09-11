#!/usr/bin/env bash
# Copy trajectories off a rented box.
#
#     scripts/fetch_runs.sh [ssh-host] [local-dir]
#
# Run this after every batch, not once before shutting the machine down.
# Trajectories are the only irreplaceable thing the box holds: sampling is
# stochastic, so a lost batch is not a re-run but a different dataset, and any
# manual auditing done against it is lost with it (D2, D12).
#
# Everything else on the box rebuilds in about two minutes — model weights 65 s,
# images 70 s, code and environment 5 s (see docs/setup-log.md). Activation
# tensors are derived from these trajectories by Pass 2 in minutes, so they are
# not worth copying either.
#
# `runs/` is gitignored: it is too large for the repo and not source. Keep the
# copies somewhere backed up.

set -euo pipefail

HOST="${1:-eip}"
DEST="${2:-./runs}"
REMOTE="${REMOTE:-eip/runs/}"

mkdir -p "$DEST"

echo "pulling $HOST:$REMOTE -> $DEST"
rsync -az --info=stats1,progress2 --partial "$HOST:$REMOTE" "$DEST/"

trajectories=$(find "$DEST" -name meta.json | wc -l | tr -d ' ')
echo
echo "$trajectories trajectories now in $DEST"
du -sh "$DEST"
