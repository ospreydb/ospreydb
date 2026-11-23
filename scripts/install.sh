#!/usr/bin/env sh
# OspreyDB Universal Installer (https://ospreydb.ramoslucas-lr.workers.dev/install.sh)
# Detects OS and CPU architecture, downloads the release binary archive from the
# latest GitHub Release, verifies its SHA-256 checksum, and installs `osprey`
# into /usr/local/bin (or ~/.local/bin when unprivileged).

set -eu

REPO="ospreydb/ospreydb"
BIN_NAME="osprey"
VERSION="${OSPREY_VERSION:-latest}"

log() {
    printf '[osprey-install] %s\n' "$1"
}

err() {
    printf '[osprey-install] ERROR: %s\n' "$1" >&2
    exit 1
}

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        err "Required command '$1' was not found in PATH."
    fi
}

detect_target() {
    os="$(uname -s)"
    arch="$(uname -m)"

    case "${os}" in
        Linux)
            case "${arch}" in
                x86_64|amd64)
                    printf 'x86_64-unknown-linux-gnu'
                    ;;
                aarch64|arm64)
                    printf 'aarch64-unknown-linux-gnu'
                    ;;
                *)
                    err "Unsupported Linux architecture: ${arch}"
                    ;;
            esac
            ;;
        Darwin)
            case "${arch}" in
                x86_64|amd64)
                    printf 'x86_64-apple-darwin'
                    ;;
                arm64|aarch64)
                    printf 'aarch64-apple-darwin'
                    ;;
                *)
                    err "Unsupported macOS architecture: ${arch}"
                    ;;
            esac
            ;;
        MINGW*|MSYS*|CYGWIN*)
            err "Windows detected. Build from source with 'cargo install --path crates/ospreydb' or download a release from https://github.com/${REPO}/releases."
            ;;
        *)
            err "Unsupported operating system: ${os}"
            ;;
    esac
}

resolve_tag() {
    if [ "${VERSION}" != "latest" ]; then
        case "${VERSION}" in
            v*) printf '%s' "${VERSION}" ;;
            *) printf 'v%s' "${VERSION}" ;;
        esac
        return
    fi

    api_url="https://api.github.com/repos/${REPO}/releases/latest"
    tag="$(curl -fsSL "${api_url}" 2>/dev/null | sed -n 's/.*"tag_name":[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1 || true)"

    if [ -z "${tag}" ]; then
        # Fallback to GitHub Releases HTTP redirect if the REST API is rate-limited.
        effective_url="$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/${REPO}/releases/latest" 2>/dev/null || true)"
        case "${effective_url}" in
            */releases/tag/*)
                tag="${effective_url##*/}"
                ;;
        esac
    fi

    if [ -z "${tag}" ]; then
        err "Failed to resolve latest release tag from ${api_url}. Set OSPREY_VERSION=vX.Y.Z explicitly."
    fi
    printf '%s' "${tag}"
}

compute_sha256() {
    file="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${file}" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "${file}" | awk '{print $1}'
    elif command -v openssl >/dev/null 2>&1; then
        openssl dgst -sha256 "${file}" | awk '{print $NF}'
    else
        err "No SHA-256 tool found (need sha256sum, shasum, or openssl)."
    fi
}

choose_install_dir() {
    if [ -n "${OSPREY_INSTALL_DIR:-}" ]; then
        printf '%s' "${OSPREY_INSTALL_DIR}"
        return
    fi
    if [ -d "/usr/local/bin" ] && [ -w "/usr/local/bin" ]; then
        printf '/usr/local/bin'
    elif [ "$(id -u)" = "0" ]; then
        printf '/usr/local/bin'
    else
        printf '%s/.local/bin' "${HOME}"
    fi
}

main() {
    need_cmd curl
    need_cmd tar
    need_cmd uname

    target="$(detect_target)"
    tag="$(resolve_tag)"
    ver="${tag#v}"
    archive="${BIN_NAME}-${ver}-${target}.tar.gz"
    base_url="https://github.com/${REPO}/releases/download/${tag}"

    tmpdir="$(mktemp -d)"
    trap 'rm -rf "${tmpdir}"' EXIT INT TERM

    log "Detected target: ${target}"
    log "Downloading ${archive} (${tag})..."
    curl -fsSL "${base_url}/${archive}" -o "${tmpdir}/${archive}"

    log "Verifying SHA-256 checksum..."
    curl -fsSL "${base_url}/${archive}.sha256" -o "${tmpdir}/${archive}.sha256"
    expected_sha="$(awk '{print $1}' "${tmpdir}/${archive}.sha256")"
    actual_sha="$(compute_sha256 "${tmpdir}/${archive}")"
    if [ -z "${expected_sha}" ] || [ "${expected_sha}" != "${actual_sha}" ]; then
        err "SHA-256 mismatch! expected=${expected_sha} actual=${actual_sha}"
    fi

    tar -xzf "${tmpdir}/${archive}" -C "${tmpdir}"
    src_bin="${tmpdir}/${BIN_NAME}-${ver}-${target}/${BIN_NAME}"
    if [ ! -f "${src_bin}" ]; then
        err "Expected binary not found inside archive: ${src_bin}"
    fi

    install_dir="$(choose_install_dir)"
    mkdir -p "${install_dir}"
    dest_bin="${install_dir}/${BIN_NAME}"
    cp "${src_bin}" "${dest_bin}"
    chmod 0755 "${dest_bin}"

    log "Installed ${BIN_NAME} (${tag}) to ${dest_bin}"
    case ":${PATH}:" in
        *":${install_dir}:"*) ;;
        *)
            log "Add '${install_dir}' to your PATH:"
            log "  export PATH=\"${install_dir}:\$PATH\""
            ;;
    esac

    "${dest_bin}" --version || true
    log "Run 'osprey doctor' to inspect host AVX-512/AVX2/NEON capabilities, or 'osprey serve' to start the server."
}

main "$@"
