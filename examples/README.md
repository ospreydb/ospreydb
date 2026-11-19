# Receitas OspreyDB

Exemplos executáveis de indexação e busca sobre dados que você **já possui**.
Em todas as receitas o OspreyDB grava apenas os artefatos compactos
(`.osi` + `.osv`), mantendo o Data Lake/mídia originais como fonte da verdade.

| Receita | Cenário | Vetores | Destino |
| :--- | :--- | :--- | :--- |
| [`parquet_datalake_rag.sh`](./parquet_datalake_rag.sh) | Tabela Parquet existente com ID, metadados e coluna de embeddings | `f32`/`f16` pré-computados | Local ou S3/MinIO |
| [`video_media_s3_rag.py`](./video_media_s3_rag.py) | Acervo de vídeos no S3, uma cena/frame por vetor com URI + timestamp | `f32` pré-computados (CLIP/SigLIP/etc.) | Local ou S3/MinIO |

## Pré-requisitos

```bash
# Binário open-source Apache-2.0 (Linux/macOS)
curl -fsSL https://ospreydb.dev/install.sh | sh
# ou: brew install ospreydb/tap/osprey / cargo install --path crates/ospreydb

osprey doctor            # SIMD da CPU, alinhamento 64B e conectividade S3
```

- Python: `python3 -m pip install pyarrow numpy` (somente para a receita de vídeo).
- S3/MinIO: credenciais em `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`.
  Para MinIO/S3-compatível, exporte `AWS_ENDPOINT_URL` e `AWS_ALLOW_HTTP=true`
  (ou passe `--endpoint` / `MINIO_ENDPOINT` no script Python).
- A AWS CLI (`aws s3 cp`) é usada apenas para baixar o `.osi` até o host; o
  `.osv` permanece no bucket e é lido por Range-GET.

## 1. Data Lake Parquet

```bash
export LAKE_INPUT=s3://meu-lago/warehouse/docs/part-00000.parquet
export SEGMENTS_URI=s3://meu-lago/osprey/docs
export QUERY='[0.12, -0.04, 0.31, ...]'   # ou ./query.npy (matriz 2D 1xD)
./examples/parquet_datalake_rag.sh
```

O script:

1. valida SIMD e acesso ao bucket (`osprey doctor`);
2. indexa a tabela direto do Parquet com `--id-col`, `--vec-col` e
   `--metadata-cols` (apenas o payload necessário é projetado);
3. baixa somente o `.osi` para mmap local, deixando o `.osv` no S3;
4. inspeciona integridade (`magic OSPR`, `xxHash64`, tombstones);
5. executa `osprey search --json` — os `hits[].payload` trazem os metadados
   para montar o contexto do RAG.

`osprey import` processa **um arquivo Parquet por vez** (não expande globs
S3). Para vários shards, gere um segmento por shard — há um exemplo comentado
no final do script.

## 2. Acervo de vídeos no S3

```bash
python3 examples/video_media_s3_rag.py
# indexando direto no bucket:
python3 examples/video_media_s3_rag.py \
  --segments s3://meu-bucket/osprey/media \
  --media-root s3://meu-bucket/media
```

O script monta uma tabela Parquet de cenas com `doc_id`, `scene_uri`
(`s3://bucket/media/video_01.mp4#t=120s`), `caption` e `embedding`, chama
`osprey import` e depois `osprey search --json`. Nenhum quadro de vídeo é
baixado: o índice guarda o vetor e o locator com timestamp; seu app usa o
`scene_uri` retornado para abrir o vídeo no ponto exato.

## Texto puro? Use `osprey serve` com `--embed-url`

Os subcomandos batch (`osprey import` e `osprey search`) são focados em
indexar e buscar vetores pré-computados. Para ingestão e busca direta por
texto puro via `/v1/embeddings`, suba o servidor `osprey serve` com
`--embed-url` (100% Apache-2.0, sem limite de vetores).

Se você já tem um encoder (OpenAI, Cohere, CLIP, sentence-transformers,
vLLM, etc.), continue no fluxo CLI acima: basta gravar a coluna de vetores
no Parquet e importar.

## Referências

- [`ARCHITECTURE.md`](../ARCHITECTURE.md) — formatos `.osi`/`.osv`/`.osr`,
  orçamento de latência e topologia de crates.
- [`README.md`](../README.md) — instalação e quickstart.
- [`rfcs/RFC-0001`](../rfcs/RFC-0001-osprey-rabitq-range-planner.md) e
  [`rfcs/RFC-0002`](../rfcs/RFC-0002-lutrank-nested-lattice-simd-reranker.md).
