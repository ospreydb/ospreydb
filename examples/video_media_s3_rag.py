#!/usr/bin/env python3
"""OspreyDB — Receita: RAG multimodal sobre cenas de vídeo em S3/MinIO.

Fluxo:
  1. Um encoder de vídeo (CLIP, SigLIP, InternVideo, etc.) já produziu um
     embedding f32/f16 por cena/frame. Este script apenas materializa esses
     vetores pré-computados (aqui sintéticos, para o exemplo rodar de ponta a
     ponta) em uma tabela Parquet com:
         doc_id, scene_uri, start_s, caption, embedding
     O `scene_uri` aponta para o objeto original com timestamp, por exemplo
         s3://meu-bucket/media/video_01.mp4#t=120s
     Nenhum byte de vídeo é baixado, copiado ou reescrito.
  2. `osprey import` indexa a tabela em .osi + .osv (locais ou no S3).
  3. `osprey search --json` retorna os `doc_id`s mais similares e o payload
     com `scene_uri`, que o seu app usa para abrir o vídeo no timestamp certo.

Dependências: apenas `pyarrow` e `numpy` (+ o binário `osprey`), mais `aws`
CLI opcional quando o destino dos segmentos for um bucket.

Uso:
  python3 examples/video_media_s3_rag.py
  python3 examples/video_media_s3_rag.py --segments s3://meu-bucket/osprey/media
  MINIO_ENDPOINT=http://localhost:9000 python3 examples/video_media_s3_rag.py \\
      --segments s3://meu-bucket/osprey/media

A geração de embeddings a partir de texto/imagem puro via HTTP não faz parte
do CLI open-source: vetores entram sempre pré-computados.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCENE_SECONDS = 45
SCENE_OFFSET_SECONDS = 30


def build_scene_table(media_root: str, dim: int) -> pa.Table:
    """Materializa embeddings de cenas (sintéticos) + locators de mídia no S3."""
    rng = np.random.default_rng(seed=7)
    videos = [
        ("video_01.mp4", "palestra sobre indexação vetorial", 4),
        ("video_02.mp4", "demo de busca multimodal em vídeo", 3),
        ("video_03.mp4", "entrevista sobre data lakes e S3", 5),
    ]

    doc_ids: list[int] = []
    scene_uris: list[str] = []
    start_seconds: list[float] = []
    captions: list[str] = []
    vectors: list[np.ndarray] = []

    next_id = 1
    for filename, topic, num_scenes in videos:
        for scene in range(num_scenes):
            start_s = float(SCENE_OFFSET_SECONDS + scene * SCENE_SECONDS)
            # Substitua por `encoder.encode(frame)` no pipeline real.
            vec = rng.normal(size=dim).astype(np.float32)
            vec /= np.linalg.norm(vec)
            doc_ids.append(next_id)
            scene_uris.append(f"{media_root.rstrip('/')}/{filename}#t={int(start_s)}s")
            start_seconds.append(start_s)
            captions.append(f"{topic} — cena {scene + 1}")
            vectors.append(vec)
            next_id += 1

    matrix = np.stack(vectors).astype(np.float32)
    return pa.table(
        {
            "doc_id": pa.array(doc_ids, type=pa.uint64()),
            "scene_uri": pa.array(scene_uris, type=pa.string()),
            "start_s": pa.array(start_seconds, type=pa.float64()),
            "caption": pa.array(captions, type=pa.string()),
            "embedding": pa.FixedSizeListArray.from_arrays(
                pa.array(matrix.reshape(-1), type=pa.float32()), dim
            ),
        }
    )


def osprey_env(endpoint: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if endpoint:
        env["AWS_ENDPOINT_URL"] = endpoint
        env["AWS_ALLOW_HTTP"] = "true"
        env.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
        env.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
    return env


def run(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), file=sys.stderr)
    return subprocess.run(cmd, check=True, text=True, env=env)


def is_remote(uri: str) -> bool:
    return "://" in uri


def download_index(
    segments_uri: str, workdir: Path, env: dict[str, str], endpoint: str | None
) -> Path:
    """O .osi é mapeado via mmap e precisa ser local; o .osv pode ficar no S3."""
    target = workdir / "segment.osi"
    source = f"{segments_uri.rstrip('/')}/segment.osi"
    if shutil.which("aws") is None:
        raise SystemExit(
            f"O índice precisa estar local. Baixe {source} para {target} "
            "(instale a AWS CLI ou passe --segments local)."
        )
    cmd = ["aws", "s3", "cp", source, str(target), "--only-show-errors"]
    if endpoint:
        cmd += ["--endpoint-url", endpoint]
    run(cmd, env)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default="./osprey-media", type=Path)
    parser.add_argument("--parquet", type=Path, help="Saída da tabela de cenas")
    parser.add_argument(
        "--segments",
        default=None,
        help="Destino .osi/.osv: diretório local ou s3://bucket/prefixo",
    )
    parser.add_argument("--media-root", default="s3://meu-bucket/media")
    parser.add_argument("--endpoint", default=os.environ.get("MINIO_ENDPOINT"))
    parser.add_argument("--dim", default=512, type=int)
    parser.add_argument("--top-k", default=5, type=int)
    args = parser.parse_args()

    if shutil.which("osprey") is None:
        raise SystemExit("Binário `osprey` não encontrado no PATH.")

    workdir: Path = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    parquet_path = args.parquet or (workdir / "scenes.parquet")
    segments = args.segments or str(workdir / "segments")
    env = osprey_env(args.endpoint)

    # 1) Tabela Parquet com vetores pré-computados e locators com timestamp.
    table = build_scene_table(args.media_root, args.dim)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet_path)
    print(f"[1/3] {table.num_rows} cenas -> {parquet_path}", file=sys.stderr)
    print(f"      mídia permanece em {args.media_root}", file=sys.stderr)

    # 2) Indexação: payload mínimo (URI + timestamp), sem duplicar o acervo.
    run(
        [
            "osprey",
            "import",
            "--input",
            str(parquet_path),
            "--output",
            segments,
            "--id-col",
            "doc_id",
            "--vec-col",
            "embedding",
            "--metadata-cols",
            "scene_uri,start_s,caption",
            "--clusters",
            str(min(64, max(2, table.num_rows // 2))),
            # .osv já usa minipages Zstd; dispensa o dicionário para payloads
            # pequenos como este (metadados escalares + locators).
            "--no-zstd-dict",
        ],
        env,
    )

    index = (
        download_index(segments, workdir, env, args.endpoint)
        if is_remote(segments)
        else Path(segments) / "segment.osi"
    )
    slabs = f"{segments.rstrip('/')}/segment.osv" if is_remote(segments) else str(
        Path(segments) / "segment.osv"
    )

    # 3) Consulta: no pipeline real este vetor vem do mesmo encoder da cena
    #    consultada (texto -> CLIP, frame -> encoder de vídeo, etc.).
    query = np.asarray(
        table.column("embedding").combine_chunks()[0].as_py(), dtype=np.float32
    )
    query = query + np.random.default_rng(1).normal(
        0.0, 0.02, size=args.dim
    ).astype(np.float32)
    query /= np.linalg.norm(query)
    query_path = workdir / "query.npy"
    # NpyMatrixReader exige matriz 2D (N, D); salvamos como (1, D).
    np.save(query_path, query.reshape(1, -1))

    result = subprocess.run(
        [
            "osprey",
            "search",
            "--index",
            str(index),
            "--slabs",
            slabs,
            "--query",
            str(query_path),
            "--top-k",
            str(args.top_k),
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    report = json.loads(result.stdout)

    print(f"[3/3] Top-{args.top_k} cenas por similaridade:", file=sys.stderr)
    for hit in report["hits"]:
        payload = hit.get("payload") or {}
        print(
            f"  #{hit['rank']:>2}  score={hit['score']:.4f}  "
            f"doc_id={hit['doc_id']}  {payload.get('scene_uri', '?')}  "
            f"| {payload.get('caption', '')}"
        )


if __name__ == "__main__":
    main()
