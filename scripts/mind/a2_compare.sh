#!/usr/bin/env bash
# A2 comparison on the laptop: the same generated worlds lived with and without language tiers.
# Runs unattended (put it in tmux); a few hours, CPU only, nice 15. Writes docs/mind-a2.md.
#   scripts/mind/a2_compare.sh [TICKS] [SEEDS...]
set -uo pipefail
cd "$(dirname "$0")/../.."
TICKS=${1:-12000}; shift || true
SEEDS=${@:-11}
OUT=$HOME/mind-experiments/a2-$(date -u +%Y%m%dT%H%M)
mkdir -p "$OUT"
DOC=docs/mind-a2.md
{ echo "# Mind A2: language tiers vs algorithms only"; echo; echo "Run $(date -u +%FT%TZ), $TICKS ticks per mind, seeds $SEEDS. Same world per seed in both arms."; } > $DOC
for s in $SEEDS; do
  for arm in algo tiers; do
    flag=""; [ $arm = algo ] && flag="--no-tiers"
    echo "$(date -u +%T) seed $s arm $arm"
    nice -n 15 python3 -m mind run --state "$OUT/$s-$arm" --fast "$TICKS" --seed "$s" $flag > "$OUT/$s-$arm.log" 2>&1
    { echo; echo "## seed $s, $arm"; echo; python3 -m mind report --state "$OUT/$s-$arm" | sed -n '5,40p'; } >> $DOC
  done
done
echo "done: $DOC"
