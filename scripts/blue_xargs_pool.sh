#!/usr/bin/env bash
# Foreground xargs worker pool. Each slot remains occupied until its command exits.
set -euo pipefail
workers=$1
nargs=$2
shift 2
[[ "$workers" =~ ^[1-9][0-9]*$ && "$nargs" =~ ^[1-9][0-9]*$ ]] || {
  echo "workers and args-per-job must be positive integers" >&2
  exit 2
}
exec xargs -r -P "$workers" -n "$nargs" "$@"
