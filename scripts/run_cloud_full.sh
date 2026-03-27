#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
fi

python -m auresnet_dz.train.train \
  data=cloud \
  train=cloud \
  "$@"
