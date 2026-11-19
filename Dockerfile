# syntax=docker/dockerfile:1.7

FROM --platform=$BUILDPLATFORM rust:1.99-bookworm AS builder

ARG TARGETARCH
ARG BUILDARCH

WORKDIR /usr/src/osprey

# Install cross-compiler toolchain only when cross-compiling between amd64 and arm64
RUN set -eux; \
    if [ "${TARGETARCH}" = "arm64" ] && [ "${BUILDARCH}" != "arm64" ]; then \
        apt-get update && apt-get install -y --no-install-recommends gcc-aarch64-linux-gnu libc6-dev-arm64-cross && rm -rf /var/lib/apt/lists/*; \
        rustup target add aarch64-unknown-linux-gnu; \
    elif [ "${TARGETARCH}" = "amd64" ] && [ "${BUILDARCH}" != "amd64" ]; then \
        apt-get update && apt-get install -y --no-install-recommends gcc-x86-64-linux-gnu libc6-dev-amd64-cross && rm -rf /var/lib/apt/lists/*; \
        rustup target add x86_64-unknown-linux-gnu; \
    fi

# Copy manifests first for dependency layer caching
COPY Cargo.toml Cargo.lock ./
COPY crates/osprey-core/Cargo.toml crates/osprey-core/Cargo.toml
COPY crates/osprey-simd/Cargo.toml crates/osprey-simd/Cargo.toml
COPY crates/osprey-storage/Cargo.toml crates/osprey-storage/Cargo.toml
COPY crates/osprey-cli/Cargo.toml crates/osprey-cli/Cargo.toml
COPY crates/ospreydb/Cargo.toml crates/ospreydb/Cargo.toml
COPY README.md LICENSE ./

# Copy workspace sources
COPY crates crates

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/usr/src/osprey/target \
    set -eux; \
    mkdir -p /out; \
    if [ "${TARGETARCH}" = "arm64" ]; then \
        TARGET="aarch64-unknown-linux-gnu"; \
        export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc; \
        export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc; \
        rustup target add "${TARGET}"; \
    else \
        TARGET="x86_64-unknown-linux-gnu"; \
        if [ "${BUILDARCH}" != "amd64" ]; then \
            export CC_x86_64_unknown_linux_gnu=x86_64-linux-gnu-gcc; \
            export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-linux-gnu-gcc; \
        fi; \
        rustup target add "${TARGET}"; \
    fi; \
    cargo build --release --locked --target "${TARGET}" -p ospreydb --bin osprey; \
    cp "/usr/src/osprey/target/${TARGET}/release/osprey" /out/osprey; \
    chmod 0755 /out/osprey

FROM gcr.io/distroless/cc-debian12

LABEL org.opencontainers.image.title="OspreyDB" \
      org.opencontainers.image.description="100% Apache-2.0 Vector Database Server & CLI (1-Bit RaBitQ, Zstd S3 Slabs, CPU LUT-Rank, WAL/LSM, Arrow Flight, Qdrant/Cohere HTTP)" \
      org.opencontainers.image.url="https://ospreydb.dev" \
      org.opencontainers.image.source="https://github.com/ospreydb/ospreydb" \
      org.opencontainers.image.licenses="Apache-2.0"

COPY --from=builder /out/osprey /usr/local/bin/osprey

USER 65532:65532
WORKDIR /data
EXPOSE 6333 8815
ENTRYPOINT ["osprey"]
CMD ["--help"]
