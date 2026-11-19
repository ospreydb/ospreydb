#!/usr/bin/env python3
"""
Deterministic Golden Vector Generator for OspreyDB (Phase 2).

Generates Little-Endian binary fixtures under reference/golden/ alongside
reference/golden/golden_manifest.json with exact byte offsets, shapes,
dtypes, seeds, and SHA-256 digests so Rust unit/integration tests (Phases 4-7)
verify bit-exact / epsilon-exact parity with the Python mathematical oracle.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Allow running both as `python3 reference/generate_golden_vectors.py` and module import
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reference.poc_reference import (  # noqa: E402
    build_nested_lattice_codebook,
    encode_tokens_nested_lattice,
    fwht_orthogonal_rotation,
    next_power_of_two,
    pack_bits_u64,
    popcount_u64,
    splitmix64_step,
)

GOLDEN_DIR = REPO_ROOT / "reference" / "golden"

FWHT_DIMS = [64, 128, 256, 512, 768, 1024, 1536]
SEED_FWHT = 0x4F535052_0001
SEED_BINARY_KMEANS = 0x4F535052_0002
SEED_RABITQ = 0x4F535052_0003
SEED_LUTRANK = 0x4F535052_0004


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def splitmix64_words(padded_dim: int, fwht_seed: int) -> np.ndarray:
    """Return exact uint64 SplitMix64 words used for Rademacher sign generation."""
    n_words = (padded_dim + 63) // 64
    state = fwht_seed & 0xFFFFFFFFFFFFFFFF
    words = np.empty(n_words, dtype="<u8")
    for i in range(n_words):
        state, w = splitmix64_step(state)
        words[i] = w
    return words


def write_binary_sections(
    path: Path,
    sections: list[tuple[str, np.ndarray]],
) -> dict[str, Any]:
    """
    Concatenate arrays in strict Little-Endian order and return manifest metadata
    with exact byte offsets, lengths, shapes, dtypes, and SHA-256 digest.
    """
    offset = 0
    sec_meta: list[dict[str, Any]] = []
    with path.open("wb") as f:
        for name, arr in sections:
            le_dtype = arr.dtype.newbyteorder("<")
            le_arr = np.ascontiguousarray(arr, dtype=le_dtype)
            raw = le_arr.tobytes(order="C")
            f.write(raw)
            sec_meta.append(
                {
                    "name": name,
                    "offset": offset,
                    "byte_len": len(raw),
                    "shape": list(le_arr.shape),
                    "dtype": le_arr.dtype.str,
                }
            )
            offset += len(raw)

    return {
        "file": path.name,
        "total_bytes": offset,
        "sha256": sha256_file(path),
        "sections": sec_meta,
    }


def generate_fwht_golden() -> list[dict[str, Any]]:
    """Vetor Ouro 1: Multi-dimension FWHT fixtures (fwht_d{d}.bin)."""
    entries: list[dict[str, Any]] = []
    n_vecs = 16

    for d in FWHT_DIMS:
        padded_dim = next_power_of_two(d)
        rng = np.random.default_rng(SEED_FWHT + d)

        # Normalized anisotropic float32 input vectors
        aniso = 1.0 / np.sqrt(1.0 + np.arange(d, dtype=np.float64) * 0.20)
        vecs = rng.normal(size=(n_vecs, d)) * aniso
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        inputs_f32 = vecs.astype("<f4")

        # Rotate the exact float32 inputs so Rust receives identical inputs
        outputs_f32 = fwht_orthogonal_rotation(inputs_f32.astype(np.float64), SEED_FWHT).astype("<f4")
        words_u64 = splitmix64_words(padded_dim, SEED_FWHT)

        out_path = GOLDEN_DIR / f"fwht_d{d}.bin"
        meta = write_binary_sections(
            out_path,
            [
                ("sign_words", words_u64),
                ("inputs", inputs_f32),
                ("outputs", outputs_f32),
            ],
        )
        meta.update(
            {
                "kind": "fwht",
                "seed": f"0x{SEED_FWHT:016x}",
                "orig_dim": d,
                "padded_dim": padded_dim,
                "num_vectors": n_vecs,
            }
        )
        entries.append(meta)
    return entries


def train_binary_kmeans_deterministic(
    x_rot: np.ndarray,
    num_clusters: int = 16,
    iters: int = 12,
    seed: int = SEED_BINARY_KMEANS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic 1-Bit Binary K-Means returning:
      (init_centroids_bits, assignments, final_centroids_bits, final_centroids_f16)
    """
    rng = np.random.default_rng(seed)
    n_vecs, dim = x_rot.shape
    x_bits = pack_bits_u64(x_rot >= 0.0)

    init_idx = rng.choice(n_vecs, size=num_clusters, replace=False)
    centroids = x_rot[init_idx].copy()
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    init_centroids_bits = pack_bits_u64(centroids >= 0.0).astype("<u8")

    assignments = np.zeros(n_vecs, dtype="<u4")
    for _ in range(iters):
        c_bits = pack_bits_u64(centroids >= 0.0)
        best_dist = np.full(n_vecs, dim + 1, dtype=np.int32)
        for k in range(num_clusters):
            xor_words = np.bitwise_xor(x_bits, c_bits[k : k + 1, :])
            dist_k = popcount_u64(xor_words).sum(axis=1)
            mask = dist_k < best_dist
            best_dist[mask] = dist_k[mask]
            assignments[mask] = k

        x_signed = np.where(x_rot >= 0.0, 1.0, -1.0)
        for k in range(num_clusters):
            members = x_signed[assignments == k]
            if len(members) > 0:
                mean_vec = members.mean(axis=0)
                norm = np.linalg.norm(mean_vec)
                if norm > 1e-12:
                    centroids[k] = mean_vec / norm

    final_centroids_f16 = np.empty((num_clusters, dim), dtype="<f2")
    for k in range(num_clusters):
        members = x_rot[assignments == k]
        if len(members) > 0:
            final_centroids_f16[k] = members.mean(axis=0).astype("<f2")
        else:
            final_centroids_f16[k] = centroids[k].astype("<f2")

    # Align final assignments and final binary centroids with materialized f16 centroids
    c_f64 = final_centroids_f16.astype(np.float64)
    dists_sq = (
        np.sum(x_rot**2, axis=1, keepdims=True)
        - 2.0 * (x_rot @ c_f64.T)
        + np.sum(c_f64**2, axis=1, keepdims=True).T
    )
    assignments = np.argmin(dists_sq, axis=1).astype("<u4")
    final_centroids_bits = pack_bits_u64(c_f64 >= 0.0).astype("<u8")
    return init_centroids_bits, assignments, final_centroids_bits, final_centroids_f16


def generate_binary_kmeans_and_rabitq_golden() -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Vetor Ouro 2 (binary_kmeans_golden.bin) and Vetor Ouro 3 (rabitq_golden.bin)
    over N=512, d=128, K=16, N_q=16.
    """
    n_vecs = 512
    dim = 128
    num_clusters = 16
    n_queries = 16

    rng = np.random.default_rng(SEED_BINARY_KMEANS)
    aniso = 1.0 / np.sqrt(1.0 + np.arange(dim, dtype=np.float64) * 0.15)
    centers = rng.normal(size=(num_clusters, dim)) * aniso
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    topics = rng.integers(0, num_clusters, size=n_vecs)
    db_raw = centers[topics] + 0.28 * rng.normal(size=(n_vecs, dim)) * aniso
    db_raw /= np.linalg.norm(db_raw, axis=1, keepdims=True)
    db_raw_f16 = db_raw.astype("<f2")

    db_rot = fwht_orthogonal_rotation(db_raw, SEED_BINARY_KMEANS)
    db_rot_f32 = db_rot.astype("<f4")

    init_bits, assignments, final_bits, final_f16 = train_binary_kmeans_deterministic(
        db_rot_f32.astype(np.float64), num_clusters=num_clusters, iters=12, seed=SEED_BINARY_KMEANS
    )

    skm_path = GOLDEN_DIR / "binary_kmeans_golden.bin"
    skm_meta = write_binary_sections(
        skm_path,
        [
            ("input_vectors", db_rot_f32),
            ("init_centroids_bits", init_bits),
            ("assignments", assignments),
            ("final_centroids_bits", final_bits),
            ("final_centroids_f16", final_f16),
        ],
    )
    skm_meta.update(
        {
            "kind": "binary_kmeans",
            "seed": f"0x{SEED_BINARY_KMEANS:016x}",
            "num_vectors": n_vecs,
            "dim": dim,
            "num_clusters": num_clusters,
        }
    )

    # Vetor Ouro 3: Asymmetric RaBitQ over the same N=512, d=128, K=16 dataset
    c_f64 = final_f16.astype(np.float64)
    residuals = db_rot_f32.astype(np.float64) - c_f64[assignments]
    res_norms = np.linalg.norm(residuals, axis=1)
    safe_norms = np.where(res_norms > 1e-12, res_norms, 1.0)
    unit_res = residuals / safe_norms[:, np.newaxis]

    sign_bool = unit_res >= 0.0
    rabitq_code = pack_bits_u64(sign_bool).astype("<u8")  # (512, 2)
    res_norm_f16 = res_norms.astype("<f2")
    l1_norm_factor_f16 = (np.sum(np.abs(unit_res), axis=1) / dim).astype("<f2")

    # Queries (seed = SEED_RABITQ)
    q_rng = np.random.default_rng(SEED_RABITQ)
    q_topics = q_rng.integers(0, num_clusters, size=n_queries)
    q_raw = centers[q_topics] + 0.25 * q_rng.normal(size=(n_queries, dim)) * aniso
    q_raw /= np.linalg.norm(q_raw, axis=1, keepdims=True)
    q_raw_f16 = q_raw.astype("<f2")
    q_rot_f32 = fwht_orthogonal_rotation(q_raw, SEED_BINARY_KMEANS).astype("<f4")

    q_scales = np.empty(n_queries, dtype="<f4")
    q_int4 = np.empty((n_queries, dim), dtype="i1")
    q_bit_planes = np.empty((n_queries, 4, 2), dtype="<u8")
    asym_scores = np.empty((n_queries, n_vecs), dtype="<f4")
    top10_ids_f16 = np.empty((n_queries, 10), dtype="<u4")

    sign_pm1 = np.where(sign_bool, 1.0, -1.0)
    rn_f64 = res_norm_f16.astype(np.float64)
    l1_f64 = np.maximum(l1_norm_factor_f16.astype(np.float64), 1e-6)

    for qi in range(n_queries):
        qr = q_rot_f32[qi].astype(np.float64)
        q_max = np.max(np.abs(qr))
        scale = q_max / 7.0 if q_max > 1e-12 else 1.0
        q_scales[qi] = np.float32(scale)
        qi4 = np.clip(np.round(qr / scale), -7, 7).astype("i1")
        q_int4[qi] = qi4

        q_u4 = (qi4.astype(np.int16) + 7).astype(np.uint8)
        for bit in range(4):
            q_bit_planes[qi, bit] = pack_bits_u64(((q_u4 >> bit) & 1).astype(bool)[np.newaxis, :])[0]

        # Verify bit-plane popcount equivalence against direct dot(q_int4, sign_pm1)
        sum_u4 = int(np.sum(q_u4))
        pop_b = popcount_u64(rabitq_code).sum(axis=1)
        dot_u4_b = np.zeros(n_vecs, dtype=np.int32)
        for bit in range(4):
            and_w = np.bitwise_and(rabitq_code, q_bit_planes[qi, bit : bit + 1, :])
            dot_u4_b += (1 << bit) * popcount_u64(and_w).sum(axis=1)
        dot_bitwise = 2 * dot_u4_b - sum_u4 - 14 * pop_b + 7 * dim
        dot_direct = sign_pm1 @ qi4.astype(np.float64)
        assert np.array_equal(dot_bitwise, dot_direct.astype(np.int32)), "Bit-plane mismatch!"

        c_dots = c_f64 @ qr
        scores = c_dots[assignments] + rn_f64 * ((float(q_scales[qi]) * dot_bitwise) / (dim * l1_f64))
        asym_scores[qi] = scores.astype("<f4")

        # Top-64 asymmetric candidates refined with exact f16 inner product -> Top-10 IDs
        top64 = np.argpartition(asym_scores[qi], -64)[-64:]
        refined = db_raw_f16[top64].astype(np.float32) @ q_raw_f16[qi].astype(np.float32)
        top10_ids_f16[qi] = top64[np.argsort(refined)[-10:][::-1]].astype("<u4")

    rabitq_path = GOLDEN_DIR / "rabitq_golden.bin"
    rabitq_meta = write_binary_sections(
        rabitq_path,
        [
            ("cluster_id", assignments),
            ("res_norm", res_norm_f16),
            ("l1_norm_factor", l1_norm_factor_f16),
            ("rabitq_code", rabitq_code),
            ("db_raw_f16", db_raw_f16),
            ("centroids_f16", final_f16),
            ("q_raw_f16", q_raw_f16),
            ("q_rot", q_rot_f32),
            ("q_scale", q_scales),
            ("q_int4", q_int4),
            ("q_bit_planes", q_bit_planes),
            ("asym_scores", asym_scores),
            ("top10_ids_f16", top10_ids_f16),
        ],
    )
    rabitq_meta.update(
        {
            "kind": "rabitq",
            "db_seed": f"0x{SEED_BINARY_KMEANS:016x}",
            "query_seed": f"0x{SEED_RABITQ:016x}",
            "num_vectors": n_vecs,
            "dim": dim,
            "num_clusters": num_clusters,
            "num_queries": n_queries,
        }
    )
    return skm_meta, rabitq_meta


def generate_lutrank_golden() -> dict[str, Any]:
    """
    Vetor Ouro 4: LUT-Rank codebooks, nibble-packed tokens, quantized int8 LUTs,
    and exact vs LUT MaxSim scores (lutrank_golden.bin).
    """
    n_docs = 16
    l_d = 32
    l_q = 8
    dim = 128
    subspace_dim = 2
    n_subspaces = dim // subspace_dim
    gamma = 0.28

    rng = np.random.default_rng(SEED_LUTRANK)
    cb1_f64, sigma = build_nested_lattice_codebook(dim)
    cb2_f64 = gamma * cb1_f64
    cb_layer1 = cb1_f64.astype("<f4")
    cb_layer2 = cb2_f64.astype("<f4")

    # Query tokens
    q_tok = rng.normal(size=(l_q, dim))
    q_tok /= np.linalg.norm(q_tok, axis=1, keepdims=True)
    q_tok_f32 = q_tok.astype("<f4")
    q_rot = fwht_orthogonal_rotation(q_tok_f32.astype(np.float64), SEED_LUTRANK)
    q_tokens_rot = q_rot.astype("<f4")

    # Build 16 documents with graded semantic relevance
    doc_tokens_raw = np.empty((n_docs, l_d, dim), dtype="<f4")
    doc_tokens_rot = np.empty((n_docs, l_d, dim), dtype="<f4")
    doc_nibbles = np.empty((n_docs, l_d, 64), dtype="u1")

    perm = rng.permutation(n_docs)
    for d_idx in range(n_docs):
        tier = int(perm[d_idx])
        overlap = max(0.10, 0.90 - 0.05 * tier)
        d_tok = rng.normal(size=(l_d, dim))
        d_tok /= np.linalg.norm(d_tok, axis=1, keepdims=True)
        n_aligned = max(1, int(l_q * overlap))
        for k in range(n_aligned):
            iq = k % l_q
            id_tok = (k * 3) % l_d
            d_tok[id_tok] = overlap * q_tok[iq] + (1.0 - overlap) * d_tok[id_tok]
            d_tok[id_tok] /= np.linalg.norm(d_tok[id_tok])

        d_f32 = d_tok.astype("<f4")
        doc_tokens_raw[d_idx] = d_f32
        d_rot = fwht_orthogonal_rotation(d_f32.astype(np.float64), SEED_LUTRANK)
        doc_tokens_rot[d_idx] = d_rot.astype("<f4")

        l1_bytes, l2_bytes = encode_tokens_nested_lattice(
            doc_tokens_rot[d_idx].astype(np.float64), cb1_f64, gamma=gamma
        )
        doc_nibbles[d_idx, :, :32] = l1_bytes
        doc_nibbles[d_idx, :, 32:] = l2_bytes

    # Synthesize int8 LUTs for Layer 1 and Layer 2 independently (shape [8, 64, 16])
    q_sub = q_tokens_rot.astype(np.float64).reshape(l_q, n_subspaces, subspace_dim)
    lut1_f64 = np.einsum("qsd,cd->qsc", q_sub, cb_layer1.astype(np.float64))
    lut2_f64 = np.einsum("qsd,cd->qsc", q_sub, cb_layer2.astype(np.float64))

    max1 = np.max(np.abs(lut1_f64), axis=(1, 2))
    max2 = np.max(np.abs(lut2_f64), axis=(1, 2))
    scale1 = np.where(max1 > 1e-12, max1 / 127.0, 1.0).astype("<f4")
    scale2 = np.where(max2 > 1e-12, max2 / 127.0, 1.0).astype("<f4")

    q_lut1_i8 = np.clip(
        np.round(lut1_f64 / scale1.astype(np.float64)[:, None, None]), -127, 127
    ).astype("i1")
    q_lut2_i8 = np.clip(
        np.round(lut2_f64 / scale2.astype(np.float64)[:, None, None]), -127, 127
    ).astype("i1")

    maxsim_exact_f32 = np.empty(n_docs, dtype="<f4")
    maxsim_lut_f32 = np.empty(n_docs, dtype="<f4")
    sub_idx = np.arange(n_subspaces, dtype=np.int32)

    for d_idx in range(n_docs):
        sim_mat = q_tok_f32 @ doc_tokens_raw[d_idx].T
        maxsim_exact_f32[d_idx] = np.float32(np.sum(np.max(sim_mat, axis=1)))

        l1_b = doc_nibbles[d_idx, :, :32]
        l2_b = doc_nibbles[d_idx, :, 32:]
        c1 = np.empty((l_d, n_subspaces), dtype=np.uint8)
        c1[:, 0::2] = l1_b & 0x0F
        c1[:, 1::2] = (l1_b >> 4) & 0x0F

        c2 = np.empty((l_d, n_subspaces), dtype=np.uint8)
        c2[:, 0::2] = l2_b & 0x0F
        c2[:, 1::2] = (l2_b >> 4) & 0x0F

        tot = 0.0
        for qi in range(l_q):
            acc1 = q_lut1_i8[qi, sub_idx, c1].astype(np.int32).sum(axis=1)
            acc2 = q_lut2_i8[qi, sub_idx, c2].astype(np.int32).sum(axis=1)
            tok_scores = float(scale1[qi]) * acc1 + float(scale2[qi]) * acc2
            tot += float(np.max(tok_scores))
        maxsim_lut_f32[d_idx] = np.float32(tot)

    pearson = float(np.corrcoef(maxsim_exact_f32, maxsim_lut_f32)[0, 1])
    assert pearson > 0.99, f"LUT-Rank golden Pearson {pearson:.5f} <= 0.99"

    lut_path = GOLDEN_DIR / "lutrank_golden.bin"
    lut_meta = write_binary_sections(
        lut_path,
        [
            ("cb_layer1", cb_layer1),
            ("cb_layer2", cb_layer2),
            ("doc_tokens_raw", doc_tokens_raw),
            ("doc_tokens_rot", doc_tokens_rot),
            ("doc_nibbles", doc_nibbles),
            ("q_tokens_raw", q_tok_f32),
            ("q_tokens_rot", q_tokens_rot),
            ("q_lut1_i8", q_lut1_i8),
            ("q_lut2_i8", q_lut2_i8),
            ("scale1", scale1),
            ("scale2", scale2),
            ("maxsim_exact_f32", maxsim_exact_f32),
            ("maxsim_lut_f32", maxsim_lut_f32),
        ],
    )
    lut_meta.update(
        {
            "kind": "lutrank",
            "seed": f"0x{SEED_LUTRANK:016x}",
            "num_docs": n_docs,
            "doc_tokens": l_d,
            "query_tokens": l_q,
            "token_dim": dim,
            "subspace_dim": subspace_dim,
            "num_subspaces": n_subspaces,
            "gamma": gamma,
            "sigma": sigma,
            "pearson_vs_fp32": pearson,
        }
    )
    return lut_meta


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    fwht_entries = generate_fwht_golden()
    skm_meta, rabitq_meta = generate_binary_kmeans_and_rabitq_golden()
    lut_meta = generate_lutrank_golden()

    manifest = {
        "schema_version": 1,
        "endianness": "little",
        "prng": "splitmix64",
        "artifacts": fwht_entries + [skm_meta, rabitq_meta, lut_meta],
    }

    manifest_path = GOLDEN_DIR / "golden_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Generated {len(manifest['artifacts'])} golden vector binaries in {GOLDEN_DIR}:")
    for art in manifest["artifacts"]:
        print(
            f"  - {art['file']:24s} | {art['total_bytes']:8d} B | sha256={art['sha256'][:16]}..."
        )
    print(f"LUT-Rank Pearson (exact vs int8 LUT MaxSim): {lut_meta['pearson_vs_fp32']:.5f}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
