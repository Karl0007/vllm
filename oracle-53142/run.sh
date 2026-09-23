#!/usr/bin/env bash
# Reproduce the three arms of the #53142 restore-fidelity run.
#
#   ./run.sh A|B|C
#
# The tree is this branch (MaCoredroid's fixture + #53798's implementation).
# Arm A = the tree as it is; B and C overlay a `mamba_hybrid.py` carrying the
# #55507 binding semantics (B: as written, C: with the fallback fix).
set -euo pipefail
ARM="${1:?usage: run.sh A|B|C}"
IMAGE="${IMAGE:-vllm/vllm-backport:cmp170hx-v013}"
# `docker` needs a privileged group on some hosts; override when it does.
DOCKER="${DOCKER:-sudo docker}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TREE="$(dirname "$HERE")"
TARGET="$TREE/vllm/v1/worker/gpu/model_states/mamba_hybrid.py"

case "$ARM" in
  A) : ;;  # tree as-is
  B) cp "$HERE/mamba_hybrid_armB_55507_asis.py" "$TARGET" ;;
  C) cp "$HERE/mamba_hybrid_armC_55507_fixed.py" "$TARGET" ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac

# The checked-out tree has no compiled extensions; take them from the image.
$DOCKER run --rm -v "$TREE:/tree:rw" --entrypoint bash "$IMAGE" -c \
  'cd /usr/local/lib/python3.12/dist-packages && find vllm -name "*.so" -exec cp --parents {} /tree/ \;'

# --noconftest: the test file is self-contained; the image ships no `tblib`.
GPU="${CUDA_VISIBLE_DEVICES:-3}"
$DOCKER run --rm \
  --gpus "\"device=${GPU}\"" -v "$TREE:/tree:rw" -w /tree \
  --entrypoint bash "$IMAGE" -c \
  "python3 -m pytest oracle-53142/test_mamba_hybrid_model_state_adapted.py \
     -q --noconftest -p no:cacheprovider"
