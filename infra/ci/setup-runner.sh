#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.runner.yml"
ENV_FILE="${SCRIPT_DIR}/.env"
ENV_EXAMPLE="${SCRIPT_DIR}/.env.example"
DOCKER_SOCK="/var/run/docker.sock"

echo "=== OspreyDB Pro — Self-Hosted GitHub Actions Runner Setup ==="

# 1. Check Docker & Docker Compose
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: 'docker' CLI not found. Install Docker Engine first (Fedora/Ubuntu)." >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: 'docker compose' plugin (v2+) not found. Install docker-compose-plugin." >&2
  exit 1
fi
echo "[ok] Docker: $(docker --version)"
echo "[ok] Compose: $(docker compose version)"

# 2. Check Docker socket existence & permissions
if [ ! -S "${DOCKER_SOCK}" ]; then
  echo "ERROR: Docker socket '${DOCKER_SOCK}' not found. Ensure the Docker daemon is running." >&2
  exit 1
fi

DOCKER_GID="$(stat -c '%g' "${DOCKER_SOCK}")"
if [ ! -r "${DOCKER_SOCK}" ] || [ ! -w "${DOCKER_SOCK}" ]; then
  echo "ERROR: Current user '$(id -un)' lacks read/write access to '${DOCKER_SOCK}' (gid=${DOCKER_GID})." >&2
  echo "Fix with: sudo usermod -aG docker \"\$USER\" && newgrp docker" >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Unable to connect to Docker daemon via '${DOCKER_SOCK}'." >&2
  exit 1
fi
echo "[ok] Docker socket accessible (${DOCKER_SOCK}, gid=${DOCKER_GID})"

# 3. Check cross-compilation tooling (cross / zig / cargo-zigbuild)
has_cross=0
has_zig=0
has_cargo_zigbuild=0
command -v cross >/dev/null 2>&1 && has_cross=1
command -v zig >/dev/null 2>&1 && has_zig=1
command -v cargo-zigbuild >/dev/null 2>&1 && has_cargo_zigbuild=1

if [ "${has_zig}" -eq 1 ] && [ "${has_cargo_zigbuild}" -eq 1 ]; then
  echo "[ok] Local multi-arch tooling: zig ($(zig version)), cargo-zigbuild"
elif [ "${has_cross}" -eq 1 ]; then
  echo "[ok] Local multi-arch tooling: cross ($(cross --version | head -n 1))"
else
  echo "[info] Local 'zig'/'cargo-zigbuild'/'cross' not found on host."
  echo "       The CI workflow automatically installs and caches zig, cargo-zigbuild, and sccache"
  echo "       inside the runner's persistent '/cargo-cache' Docker volume on first run."
fi

# 4. Prepare infra/ci/.env
if [ ! -f "${ENV_FILE}" ]; then
  cp "${ENV_EXAMPLE}" "${ENV_FILE}"
  chmod 0600 "${ENV_FILE}"
  echo "[created] ${ENV_FILE} from .env.example"
fi

if grep -q '^DOCKER_GID=' "${ENV_FILE}"; then
  sed -i "s/^DOCKER_GID=.*/DOCKER_GID=${DOCKER_GID}/" "${ENV_FILE}"
else
  echo "DOCKER_GID=${DOCKER_GID}" >> "${ENV_FILE}"
fi

# Load configured tokens without exporting unrelated host vars
ACCESS_TOKEN_VAL="$(awk -F= '/^ACCESS_TOKEN=/{print $2}' "${ENV_FILE}" | tr -d '[:space:]')"
RUNNER_TOKEN_VAL="$(awk -F= '/^RUNNER_TOKEN=/{print $2}' "${ENV_FILE}" | tr -d '[:space:]')"

print_github_instructions() {
  cat <<'EOF'

=== GitHub Registration Instructions ===
1. Open the private repository on GitHub:
   Settings -> Actions -> Runners -> New self-hosted runner
2. Copy the short-lived token into `infra/ci/.env` as `RUNNER_TOKEN=...`
   (OR set a PAT with `repo` scope as `ACCESS_TOKEN=...` for auto-renewal on restarts).
3. Verify the runner appears online with label `osprey-pro-builder` under:
   Settings -> Actions -> Runners
EOF
}

if [ -z "${ACCESS_TOKEN_VAL}" ] && [ -z "${RUNNER_TOKEN_VAL}" ]; then
  echo ""
  echo "[action required] Neither ACCESS_TOKEN nor RUNNER_TOKEN is set in '${ENV_FILE}'."
  print_github_instructions
  echo ""
  echo "After updating '${ENV_FILE}', re-run: ${0}"
  exit 0
fi

# 5. Start runner stack in background
echo "=== Starting osprey-pro self-hosted runner ==="
DOCKER_GID="${DOCKER_GID}" docker compose -f "${COMPOSE_FILE}" up -d

echo ""
docker compose -f "${COMPOSE_FILE}" ps
print_github_instructions
echo ""
echo "Tail logs with: docker compose -f ${COMPOSE_FILE} logs -f"
