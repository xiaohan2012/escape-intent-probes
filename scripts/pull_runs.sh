#!/usr/bin/env bash
# Mirror the box's trajectories to the local runs/, and prove it worked.
#
#     scripts/pull_runs.sh            # pull everything under ~/eip/runs from `eip`
#     scripts/pull_runs.sh eip 'probe-02-*'
#
# Exists because of 2026-09-14: a box was terminated on the assurance that
# "everything durable is local", and the 94 probe-01 trajectories — the frozen
# asset (D12), sole source of labels and token ids — existed only on that box.
# runs/ is gitignored, so git saves nothing here; this script is the mirror.
#
# Run it the moment a batch lands. Trajectories are small (a few MB per batch);
# there is no reason to ever batch this up.
#
# The count check at the end is the point: rsync exiting 0 says the transfer
# ran, not that the local tree now holds what the box holds. Count meta.json
# on both sides and fail loudly on any difference.
set -euo pipefail

HOST=${1:-eip}
GLOB=${2:-*}
REMOTE_ROOT="eip/runs"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)/runs"

mkdir -p "$LOCAL_ROOT"

rsync -a --partial \
  --include="/${GLOB}" --include="/${GLOB}/**" --exclude='/*' \
  "$HOST:$REMOTE_ROOT/" "$LOCAL_ROOT/"

remote=$(ssh "$HOST" "find $REMOTE_ROOT -path \"$REMOTE_ROOT/$GLOB/*\" -name meta.json | wc -l" | tr -d '[:space:]')
local_count=$(find "$LOCAL_ROOT" -path "$LOCAL_ROOT/$GLOB/*" -name meta.json | wc -l | tr -d '[:space:]')

echo "remote meta.json: $remote"
echo "local  meta.json: $local_count"

if [[ "$local_count" -lt "$remote" ]]; then
  echo "MISMATCH: local holds fewer trajectories than the box — do not trust this mirror" >&2
  exit 1
fi
echo "mirrored: every trajectory on the box is now local"
