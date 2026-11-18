# syntax=docker/dockerfile:1.7

FROM gcr.io/distroless/cc-debian12

ARG TARGETARCH

LABEL org.opencontainers.image.title="OspreyDB Pro" \
      org.opencontainers.image.description="OspreyDB Commercial Engine (Arrow Flight, Qdrant/Cohere Drop-In, Auto-Embeddings, WAL/LSM & S3 CAS)" \
      org.opencontainers.image.url="https://ospreydb.dev" \
      org.opencontainers.image.source="https://github.com/ospreydb/osprey-pro" \
      org.opencontainers.image.licenses="LicenseRef-OspreyDB-Commercial"

# Pre-compiled static musl binary staged by CI (`dist-docker/amd64` or `dist-docker/arm64`)
COPY dist-docker/${TARGETARCH}/osprey-pro /usr/local/bin/osprey-pro

USER 65532:65532
WORKDIR /data

EXPOSE 6333 8815

ENTRYPOINT ["/usr/local/bin/osprey-pro"]
CMD ["serve", "--help"]
