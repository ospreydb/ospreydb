#!/usr/bin/env bash
# =============================================================================
# OspreyDB — Receita: RAG sobre um Data Lake Parquet existente
# =============================================================================
#
# Cenário:
#   Você já possui uma tabela Parquet (Iceberg/Delta/Hive ou um dump qualquer)
#   com uma coluna de ID, colunas de metadados e uma coluna de embeddings já
#   calculados (f32/f16). O objetivo é indexar essa tabela DIRETAMENTE para os
#   artefatos compactos do OspreyDB (.osi + .osv), localmente ou no S3/MinIO,
#   SEM reescrever, duplicar ou baixar os dados brutos do Data Lake.
#
# O que o Osprey grava:
#   segment.osi  -> índice 1-bit residente (mmap, baixo RAM) usado no Estágio 1
#   segment.osv  -> slabs de vetores f16 comprimidos com Zstd + minipages
#                   ordenados por localidade TSP (lidos via Range-GET no S3)
#   segment.osr  -> opcional, apenas se você tiver tokens ColBERT (--with-lutrank)
#
# O Parquet original permanece a fonte da verdade. O .osv é um artefato
# colunar compactado (não uma cópia linha-a-linha da sua tabela), e o .osi
# contém apenas códigos de 1 bit + locators.
#
# Pré-requisitos:
#   - `osprey` no PATH (instalador universal, Homebrew, Docker ou cargo)
#   - `aws` CLI somente se você quiser baixar o .osi de um bucket para o host
#   - Credenciais S3 no ambiente (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY)
#
# Uso:
#   export LAKE_INPUT=s3://meu-lago/warehouse/docs/part-00000.parquet
#   export SEGMENTS_URI=s3://meu-lago/osprey/docs
#   export QUERY='[0.12, -0.04, ...]'   # ou um caminho .npy/.json
#   ./examples/parquet_datalake_rag.sh
#
# Se o import for interrompido, reexecute-o: o segmento é regravado a partir
# do Parquet original (a fonte da verdade nunca é modificada).
# =============================================================================

set -euo pipefail

# --- Configuração (sobrescreva via ambiente) --------------------------------
# Um único arquivo Parquet por import. `osprey import` não expande globs de
# S3; para vários shards, importe cada shard em um segmento separado (veja o
# bloco "Múltiplos shards" no final) e consulte o segmento desejado.
LAKE_INPUT="${LAKE_INPUT:-s3://meu-lago/warehouse/docs/part-00000.parquet}"

# Destino dos artefatos: prefixo S3/MinIO ou diretório local.
#   S3:    s3://meu-lago/osprey/docs
#   Local: ./osprey-datalake/segments
SEGMENTS_URI="${SEGMENTS_URI:-s3://meu-lago/osprey/docs}"

WORKDIR="${WORKDIR:-./osprey-datalake}"
ID_COL="${ID_COL:-doc_id}"
VEC_COL="${VEC_COL:-embedding}"
# Colunas escalares preservadas no payload e devolvidas no resultado da busca.
METADATA_COLS="${METADATA_COLS:-title,url,lang,updated_at}"
CLUSTERS="${CLUSTERS:-256}"
TOP_K="${TOP_K:-10}"

# Consulta: JSON inline (`[0.1, -0.2, ...]`) ou caminho para `.npy`/`.json`.
# A dimensionalidade deve ser igual à coluna de embeddings indexada.
# Para `.npy`, use uma matriz 2D (`np.save(path, query.reshape(1, -1))`).
QUERY="${QUERY:-}"

# MinIO / S3-compatível (ex.: http://localhost:9000). Deixe vazio para AWS.
# O object_store lê estas variáveis automaticamente.
if [[ -n "${MINIO_ENDPOINT:-}" ]]; then
  export AWS_ENDPOINT_URL="${MINIO_ENDPOINT}"
  export AWS_ALLOW_HTTP=true
  export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-minioadmin}"
  export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-minioadmin}"
fi

need() { command -v "$1" >/dev/null 2>&1 || { echo "ERRO: '$1' não encontrado no PATH." >&2; exit 1; }; }
need osprey
mkdir -p "$WORKDIR"

echo "==> [1/4] Diagnóstico de SIMD e conectividade com o Object Storage"
# Valida CPU (AVX-512/AVX2/NEON) e alinhamento de 64 bytes. Se a entrada já
# estiver no S3/MinIO, o doctor também faz Head + Range-GET no objeto.
if [[ "$LAKE_INPUT" == s3://* ]]; then
  osprey doctor --s3-uri "$LAKE_INPUT"
else
  osprey doctor
fi

echo
echo "==> [2/4] Indexando '$LAKE_INPUT' -> '$SEGMENTS_URI'"
# Notas de flags:
#   --id-col / --vec-col     evitam a autodetecção (doc_id/id, embedding/vector)
#   --metadata-cols          projeta apenas o payload necessário (não o vetor)
#   --clusters 256           K do 1-Bit Binary K-Means (ajuste pela escala do corpus)
#   --page-size-kb 16        alvo de minipage descomprimida do .osv
#   --sample-size            amostra do Pass-1 para treinar o codebook
# Sem --with-lutrank: esta receita usa dense retrieval puro (sem tokens ColBERT).
osprey import \
  --input "$LAKE_INPUT" \
  --output "$SEGMENTS_URI" \
  --id-col "$ID_COL" \
  --vec-col "$VEC_COL" \
  --metadata-cols "$METADATA_COLS" \
  --clusters "$CLUSTERS" \
  --page-size-kb 16 \
  --batch-size 4096

# Se o destino for remoto, o import já publicou:
#   "$SEGMENTS_URI/segment.osi", "$SEGMENTS_URI/segment.osv"
if [[ "$SEGMENTS_URI" == s3://* ]]; then
  if command -v aws >/dev/null 2>&1; then
    echo "==> Baixando apenas o .osi para mmap local (o .osv fica no S3)"
    AWS_CP=(aws s3 cp --only-show-errors)
    if [[ -n "${AWS_ENDPOINT_URL:-}" ]]; then
      AWS_CP+=(--endpoint-url "$AWS_ENDPOINT_URL")
    fi
    "${AWS_CP[@]}" "$SEGMENTS_URI/segment.osi" "$WORKDIR/segment.osi"
  else
    echo "AVISO: 'aws' não encontrado. Baixe manualmente o segment.osi:" >&2
    echo "       $SEGMENTS_URI/segment.osi -> $WORKDIR/segment.osi" >&2
  fi
  INDEX="$WORKDIR/segment.osi"
  SLABS="$SEGMENTS_URI/segment.osv"
else
  INDEX="$SEGMENTS_URI/segment.osi"
  SLABS="$SEGMENTS_URI/segment.osv"
fi

echo
echo "==> [3/4] Inspecionando integridade (magic OSPR, xxHash64, tombstones)"
# `osprey inspect` abre artefatos locais via mmap; inspecione o .osv local
# quando ele não estiver no bucket.
if [[ -f "$INDEX" ]]; then
  INSPECT_ARGS=(--file "$INDEX" --verify-checksums)
  if [[ "$SLABS" != *"://"* ]]; then
    INSPECT_ARGS+=(--file "$SLABS")
  fi
  osprey inspect "${INSPECT_ARGS[@]}"
else
  echo "AVISO: .osi local ausente; pulando inspect (baixe-o e rode novamente)." >&2
fi

echo
echo "==> [4/4] Busca vetorial (K=$TOP_K)"
if [[ -z "$QUERY" ]]; then
  cat >&2 <<'EOF'
Defina QUERY com um vetor já computado, por exemplo:

  # Inline (dimensão deve bater com a coluna indexada)
  export QUERY='[0.12, -0.04, 0.31, ...]'

  # Ou um arquivo .npy / .json
  export QUERY=./query.npy

  ./examples/parquet_datalake_rag.sh

O `osprey` open-source indexa e busca vetores PRÉ-COMPUTADOS. A geração de
embeddings a partir de texto puro (HTTP /v1/embeddings) é uma feature do
daemon comercial `osprey` (`ospreydb/ospreydb`) e não existe neste CLI público.
EOF
  exit 0
fi

# O .osv pode permanecer no S3: o planejador DP agrupa os Range-GETs por
# localidade TSP e o refinamento exato em f16 acontece na CPU.
osprey search \
  --index "$INDEX" \
  --slabs "$SLABS" \
  --query "$QUERY" \
  --top-k "$TOP_K" \
  --nprobe 16 \
  --candidates 200 \
  --json

# O JSON retornado contém `hits[].doc_id`, `hits[].score` e
# `hits[].payload` com as colunas de metadados projetadas (title, url, ...).
# Use esse payload para montar o contexto do seu pipeline de RAG.

# =============================================================================
# Múltiplos shards (sem duplicar o Data Lake)
# =============================================================================
# Cada shard gera um segmento independente. Importe-os em paralelo ou em lote
# e consulte o segmento relevante; o Parquet original nunca é copiado:
#
#   for part in part-00000 part-00001 part-00002; do
#     osprey import \
#       --input "s3://meu-lago/warehouse/docs/${part}.parquet" \
#       --output "s3://meu-lago/osprey/docs/${part}" \
#       --id-col doc_id --vec-col embedding \
#       --metadata-cols title,url
#   done
#
# Dica de FinOps: mantenha o .osi (pequeno) em instâncias de consulta e o
# .osv no S3. O planejador DP coalesce leituras e reduz egress drasticamente.
# =============================================================================
