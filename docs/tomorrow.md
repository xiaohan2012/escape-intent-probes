# Next session

Rewritten 2026-09-14 early morning, after the overnight run. Pass 1 and Pass 2
are **complete**; the first probe has been fitted and reported (PR #7 comment).

## Where things stand

| stage | state |
|---|---|
| Pass 1 | **done** — 72/72 impossible (crossed 20 = 28%), 22/24 benign (2 recoverable failures) |
| Pass 2 | **done** — 94 npz, hooked single-forward, fp32, verified; 8.9 GB, mirrored locally in `act-zst/` |
| Probe | **first fit done** — best cell generated x R2 x L37: AUC 0.951, median lead 5 steps, false alarms 33-52% |

## The result, in one paragraph

The probe fires ~5 steps before the crossing and is not a string detector
(falsifier 1 passed). But a third to a half of pressured non-crossers fire too
(falsifier 2 partially failed): step-level separability is excellent, trajectory
-level specificity is moderate. R1 behaved exactly as the design predicted — a
condition detector — which validates the R1/R2 contrast as a diagnostic.

## Today, in rough order

1. **Probe iteration on the local activations** (no GPU needed, seconds per
   fit): tune the threshold on train folds (it is fixed 0.5/0.9 now, a known
   gap in `train_probe.py`), try layer ensembles, look at score trajectories
   of the false-alarming non-crossers — are they the ones that read the cue?
2. **Decide whether more data is worth it**: the six lost trajectories (2
   benign + earlier setup failures already recovered) resume for ~10 min of
   GPU. The benign arm could grow to seeds 2-3 for a stronger negative class.
3. **Write-up** — the hackathon deadline. `docs/decisions.md` D22-D24 plus the
   PR #4/#6/#7 comments contain every number.
4. CI: one line under `[tool.ty.rules]` for the two `unused-ignore-comment`
   warnings.

## The box

`ssh eip`, up since 2026-09-13 14:36 UTC (~$61 GPU so far). Everything is
mirrored locally except the raw model weights and the run logs
(`/tmp/fit-final-t{05,09}.log` copied down as `fit-final-*.log`).
**Safe to terminate once the final `act-zst/` rsync is confirmed 94/94.**
Rebuild from `docs/setup-log.md` in ~10 min if needed.

## Reading order for someone new

1. `docs/design-doc.md` — the experiment
2. `docs/decisions.md` — D22 screen, D23 descent, D24 probe
3. PR #7 comment — the probe's first numbers
4. `docs/vllm-pitfalls.md` — infrastructure lessons, and PR #9's skill
