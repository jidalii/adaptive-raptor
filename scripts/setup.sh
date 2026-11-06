#!/usr/bin/env bash
# One-command development setup. Safe to run repeatedly.
#
#   ./scripts/setup.sh                # full setup + offline smoke run
#   ./scripts/setup.sh --no-smoke     # skip the smoke run
#   ./scripts/setup.sh --with-tests   # also run the offline test suite
#   ./scripts/setup.sh --no-install-uv  # fail instead of installing uv if it is missing
#
# Never downloads embedding models or datasets and never calls paid APIs.
set -euo pipefail

RUN_SMOKE=1; RUN_TESTS=0; INSTALL_UV=1
for arg in "$@"; do
  case "$arg" in
    --no-smoke) RUN_SMOKE=0 ;;
    --with-tests) RUN_TESTS=1 ;;
    --no-install-uv) INSTALL_UV=0 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."
step() { printf '\n==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

step "1/7 System requirements"
OS="$(uname -s)"; ARCH="$(uname -m)"
echo "platform: $OS $ARCH"
case "$OS" in
  Darwin|Linux) ;;
  *) die "unsupported OS '$OS' (macOS or Linux required)" ;;
esac
[ "$OS" = "Darwin" ] && [ "$ARCH" != "arm64" ] && echo "note: primary target is macOS arm64; $ARCH is untested"
for cmd in curl tar git; do
  command -v "$cmd" >/dev/null 2>&1 || die "'$cmd' is required but not installed"
done

step "2/7 uv"
find_uv() {
  command -v uv 2>/dev/null && return 0
  for p in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" /opt/homebrew/bin/uv /usr/local/bin/uv; do
    [ -x "$p" ] && { echo "$p"; return 0; }
  done
  return 1
}
if ! UV="$(find_uv)"; then
  [ "$INSTALL_UV" = 1 ] || die "uv not found (install: https://docs.astral.sh/uv/)"
  if [ "$OS" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
    echo "installing uv with Homebrew"; brew install uv
  else
    echo "installing uv with the official installer (https://astral.sh/uv/install.sh)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  UV="$(find_uv)" || die "uv installation failed; install it manually and re-run"
fi
echo "uv: $UV ($("$UV" --version))"

step "3/7 Python environment (.venv from uv.lock)"
"$UV" python install "$(cat .python-version)" >/dev/null
"$UV" sync --frozen
echo "installed: $("$UV" run python --version)"

step "4/7 Local directories and .env"
mkdir -p data indexes outputs .cache
if [ ! -f .env ]; then
  cp .env.example .env
  echo "created .env from .env.example (add your API keys there)"
else
  echo ".env exists (left unchanged)"
fi

step "5/7 Environment check"
"$UV" run python scripts/check_environment.py

if [ "$RUN_SMOKE" = 1 ]; then
  step "6/7 Offline smoke run (mock models, bundled fixture, real RAPTOR clustering)"
  "$UV" run python -m adaptive_raptor.run --config configs/experiments/smoke_mock.yaml \
    || die "smoke run failed"
else
  step "6/7 Smoke run skipped (--no-smoke)"
fi

if [ "$RUN_TESTS" = 1 ]; then
  step "7/7 Offline test suite"
  "$UV" run pytest -q
else
  step "7/7 Tests skipped (run: uv run pytest)"
fi

cat <<'MSG'

Setup complete. Next steps:
  1. Add DEEPSEEK_API_KEY to .env            (needed only for real runs)
  2. ./scripts/download_datasets.sh --dataset qasper
  3. uv run python -m adaptive_raptor.run --config configs/experiments/baseline.yaml --validate-only
  4. uv run python -m adaptive_raptor.run --config configs/experiments/baseline.yaml
     (first real run downloads the SBERT model, ~420 MB, and calls the DeepSeek API)
MSG
