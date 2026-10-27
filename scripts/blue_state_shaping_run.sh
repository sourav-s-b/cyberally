#!/usr/bin/env bash
# Compatibility entry point. New comparisons use the finite-Red native gate.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "$ROOT/scripts/blue_finite_red_check.sh" "$@"
