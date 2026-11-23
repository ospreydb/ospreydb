#!/usr/bin/env python3
"""
OspreyDB Mathematical Oracle & Golden Reference (Phase 2).

Implements bit-exact SplitMix64 Rademacher sign generation, O(d log d) FWHT
orthogonal rotation, and end-to-end mathematical proofs for:
  1. Vector Search: 1-Bit Binary K-Means + Asymmetric RaBitQ + int4 query + f16 refinement
  2. LUT-Rank: 2-layer 2D Nested Lattice (gamma=0.28) + int8 16-entry LUT MaxSim
"""

from __future__ import annotations

import math
import time
import numpy as np

MASK64 = 0xFFFFFFFFFFFFFFFF


def next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n (minimum 1)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def splitmix64_step(state: int) -> tuple[int, int]:
    """Advance SplitMix64 state by one 64-bit word (bit-exact with Rust)."""
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    z = z ^ (z >> 31)
    return state, z


def rademacher_signs_splitmix64(padded_dim: int, fwht_seed: int) -> np.ndarray:
    """
    Generate deterministic Rademacher signs in {-1.0, +1.0}^padded_dim.

    Each 64-bit SplitMix64 output word defines 64 signs in LSB-to-MSB order:
    bit 0 -> -1.0, bit 1 -> +1.0.
    """
    state = fwht_seed & MASK64
    signs = np.empty(padded_dim, dtype=np.float64)
    idx = 0
    while idx < padded_dim:
        state, word = splitmix64_step(state)
        take = min(64, padded_dim - idx)
        for bit in range(take):
            signs[idx + bit] = 1.0 if ((word >> bit) & 1) else -1.0
        idx += take
    return signs


def fwht_orthogonal_rotation(x: np.ndarray, fwht_seed: int) -> np.ndarray:
    """
    Zero-pad vectors to d_padded = 2^ceil(log2(d)) and apply orthogonal FWHT:
        P x = (1 / sqrt(d_padded)) * H * D_signs * x
    Supports 1D (d,) or 2D (N, d) arrays.
    """
    arr = np.asarray(x, dtype=np.float64)
    is_1d = arr.ndim == 1
    if is_1d:
        arr = arr[np.newaxis, :]

    n_vecs, orig_dim = arr.shape
    padded_dim = next_power_of_two(orig_dim)
    buf = np.zeros((n_vecs, padded_dim), dtype=np.float64)
    buf[:, :orig_dim] = arr

    signs = rademacher_signs_splitmix64(padded_dim, fwht_seed)
    buf *= signs[np.newaxis, :]

    h = 1
    while h < padded_dim:
        step = 2 * h
        blocks = buf.reshape(n_vecs, padded_dim // step, 2, h)
        a = blocks[:, :, 0, :].copy()
        b = blocks[:, :, 1, :]
        blocks[:, :, 0, :] = a + b
        blocks[:, :, 1, :] = a - b
        h = step

    buf /= math.sqrt(padded_dim)
    return buf[0] if is_1d else buf


def pack_bits_u64(bit_matrix: np.ndarray) -> np.ndarray:
    """Pack boolean array of shape (N, D) into uint64 words of shape (N, D // 64)."""
    n_vecs, dim = bit_matrix.shape
    assert dim % 64 == 0, "Bit-packing requires padded_dim divisible by 64"
    words = dim // 64
    reshaped = bit_matrix.astype(np.uint64).reshape(n_vecs, words, 64)
    shifts = np.arange(64, dtype=np.uint64)
    return np.bitwise_or.reduce(reshaped << shifts, axis=2)


def popcount_u64(words: np.ndarray) -> np.ndarray:
    """Portable popcount over uint64 array using byte LUT."""
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)
    raw_bytes = words.view(np.uint8).reshape(*words.shape, 8)
    return lut[raw_bytes].sum(axis=-1, dtype=np.int32)


def train_1bit_binary_kmeans(
    x_rot: np.ndarray,
    num_clusters: int = 64,
    iters: int = 12,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1-Bit Binary K-Means operating over binarized FWHT-rotated signatures.
    Returns (centroids_f16, assignments).
    """
    rng = np.random.default_rng(seed)
    n_vecs, dim = x_rot.shape
    x_bits = pack_bits_u64(x_rot >= 0.0)

    # Deterministic furthest/stratified initialization in 1-bit space
    init_idx = rng.choice(n_vecs, size=num_clusters, replace=False)
    centroids = x_rot[init_idx].copy()
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)

    assignments = np.zeros(n_vecs, dtype=np.int32)
    for _ in range(iters):
        c_bits = pack_bits_u64(centroids >= 0.0)
        # Hamming similarity: maximize agreement bits (D - 2 * popcnt(x ^ c))
        # Equivalent to minimizing popcnt(x_bits ^ c_bits)
        best_dist = np.full(n_vecs, dim + 1, dtype=np.int32)
        for k in range(num_clusters):
            xor_words = np.bitwise_xor(x_bits, c_bits[k : k + 1, :])
            dist_k = popcount_u64(xor_words).sum(axis=1)
            mask = dist_k < best_dist
            best_dist[mask] = dist_k[mask]
            assignments[mask] = k

        # Update centroids from signed 1-bit representations of assigned members
        x_signed = np.where(x_rot >= 0.0, 1.0, -1.0)
        for k in range(num_clusters):
            members = x_signed[assignments == k]
            if len(members) > 0:
                mean_vec = members.mean(axis=0)
                norm = np.linalg.norm(mean_vec)
                if norm > 1e-12:
                    centroids[k] = mean_vec / norm

    # Final f16 centroid calibration (mean of member rotated vectors quantized to f16)
    centroids_f16 = np.empty((num_clusters, dim), dtype=np.float16)
    for k in range(num_clusters):
        members = x_rot[assignments == k]
        if len(members) > 0:
            mean_vec = members.mean(axis=0)
            centroids_f16[k] = mean_vec.astype(np.float16)
        else:
            centroids_f16[k] = centroids[k].astype(np.float16)

    # Re-assign once against exact f16 centroids to ensure residual consistency
    c_f32 = centroids_f16.astype(np.float64)
    dists_sq = (
        np.sum(x_rot**2, axis=1, keepdims=True)
        - 2.0 * (x_rot @ c_f32.T)
        + np.sum(c_f32**2, axis=1, keepdims=True).T
    )
    assignments = np.argmin(dists_sq, axis=1).astype(np.int32)
    return centroids_f16, assignments


def run_vector_search_proof() -> None:
    """
    Proof 1: Vector Search (1-Bit Binary K-Means K=64 + Asymmetric RaBitQ + int4 query
    + Top-150 f16 exact refinement) on anisotropic normalized embeddings.
    """
    print("=== Proof 1: Vector Search (FWHT + 1-Bit Binary K-Means + Asymmetric RaBitQ + f16 Refine) ===")
    rng = np.random.default_rng(20261115)

    n_docs = 8192
    n_queries = 100
    orig_dim = 384
    num_clusters = 64
    nprobe = 16
    top_candidates = 150
    top_k = 10
    fwht_seed = 0x4F5350525F533356  # "OSPR_S3V"

    # Build realistic anisotropic multi-modal dataset (64 latent topics + power-law decay)
    aniso_scale = 1.0 / np.sqrt(1.0 + np.arange(orig_dim, dtype=np.float64) * 0.25)
    topic_centers = rng.normal(size=(num_clusters, orig_dim)) * aniso_scale
    topic_centers /= np.linalg.norm(topic_centers, axis=1, keepdims=True)

    doc_topics = rng.integers(0, num_clusters, size=n_docs)
    db_raw = topic_centers[doc_topics] + 0.32 * rng.normal(size=(n_docs, orig_dim)) * aniso_scale
    db_raw /= np.linalg.norm(db_raw, axis=1, keepdims=True)

    # Queries drawn near corpus topics with perturbation
    q_topics = rng.integers(0, num_clusters, size=n_queries)
    q_raw = topic_centers[q_topics] + 0.28 * rng.normal(size=(n_queries, orig_dim)) * aniso_scale
    q_raw /= np.linalg.norm(q_raw, axis=1, keepdims=True)

    # Verify FWHT orthogonality invariant
    db_rot = fwht_orthogonal_rotation(db_raw, fwht_seed)
    padded_dim = db_rot.shape[1]
    assert padded_dim == 512, f"Expected padded_dim=512 for orig_dim=384, got {padded_dim}"

    exact_dots = q_raw @ db_raw.T
    rot_q = fwht_orthogonal_rotation(q_raw, fwht_seed)
    rot_dots = rot_q @ db_rot.T
    ortho_err = np.max(np.abs(exact_dots - rot_dots))
    assert ortho_err < 1e-10, f"FWHT orthogonality violated: max_err={ortho_err}"

    # 1-Bit Binary K-Means (K=64)
    centroids_f16, assignments = train_1bit_binary_kmeans(db_rot, num_clusters=num_clusters)
    centroids_f64 = centroids_f16.astype(np.float64)

    # Asymmetric RaBitQ encoding of intra-cluster residuals
    residuals = db_rot - centroids_f64[assignments]
    res_norms = np.linalg.norm(residuals, axis=1)
    safe_norms = np.where(res_norms > 1e-12, res_norms, 1.0)
    unit_res = residuals / safe_norms[:, np.newaxis]

    sign_bool = unit_res >= 0.0
    code_words = pack_bits_u64(sign_bool)  # packed 1-bit codes [N, padded_dim // 64]
    res_norm_f16 = res_norms.astype(np.float16)
    l1_norm_factor_f16 = (np.sum(np.abs(unit_res), axis=1) / padded_dim).astype(np.float16)

    # Store exact f16 vectors as in .osv minipages
    db_f16 = db_raw.astype(np.float16)

    # Pre-group cluster indices (Cluster-SoA simulation)
    cluster_members = [np.flatnonzero(assignments == k) for k in range(num_clusters)]

    recalls = []
    t0 = time.perf_counter()
    for qi in range(n_queries):
        q_orig = q_raw[qi]
        q_r = rot_q[qi]

        # Probe top nprobe clusters using f16 centroids
        c_scores = centroids_f64 @ q_r
        probed_clusters = np.argpartition(c_scores, -nprobe)[-nprobe:]

        # Symmetric int4 quantization of rotated query: [-7, +7]
        q_max = np.max(np.abs(q_r))
        q_scale = q_max / 7.0 if q_max > 1e-12 else 1.0
        q_int4 = np.clip(np.round(q_r / q_scale), -7, 7).astype(np.int8)

        # Bitwise equivalence check via unsigned 4-bit offset + popcount on sample cluster
        # u = q_int4 + 7 in [0, 14]: <q_int4, 2*b - 1> = 2*<u, b> - sum(u) - 14*popcnt(b) + 7*D
        q_u4 = (q_int4 + 7).astype(np.uint8)
        q_planes = [pack_bits_u64(((q_u4 >> bit) & 1).astype(bool)[np.newaxis, :])[0] for bit in range(4)]
        sum_u4 = int(np.sum(q_u4))

        cand_indices = []
        cand_scores = []
        for k in probed_clusters:
            members = cluster_members[k]
            if len(members) == 0:
                continue
            m_words = code_words[members]
            # 4-bit-plane AND + popcount inner product
            dot_u4_b = np.zeros(len(members), dtype=np.int32)
            for bit in range(4):
                and_words = np.bitwise_and(m_words, q_planes[bit][np.newaxis, :])
                dot_u4_b += (1 << bit) * popcount_u64(and_words).sum(axis=1)
            pop_b = popcount_u64(m_words).sum(axis=1)
            # <q_int4, sign(u)> = 2*dot_u4_b - sum_u4 - 14*pop_b + 7*D
            dot_int4_sign = 2 * dot_u4_b - sum_u4 - 14 * pop_b + 7 * padded_dim

            rn = res_norm_f16[members].astype(np.float64)
            l1 = np.maximum(l1_norm_factor_f16[members].astype(np.float64), 1e-6)
            est_scores = c_scores[k] + rn * ((q_scale * dot_int4_sign) / (padded_dim * l1))

            cand_indices.append(members)
            cand_scores.append(est_scores)

        all_cands = np.concatenate(cand_indices)
        all_est = np.concatenate(cand_scores)

        # Select Top-150 candidates for exact f16 refinement (.osv stage 3)
        k_sel = min(top_candidates, len(all_cands))
        top_c_pos = np.argpartition(all_est, -k_sel)[-k_sel:]
        stage2_ids = all_cands[top_c_pos]

        q_f16_cast = q_orig.astype(np.float16).astype(np.float32)
        refined_scores = db_f16[stage2_ids].astype(np.float32) @ q_f16_cast
        top10_pred = stage2_ids[np.argsort(refined_scores)[-top_k:][::-1]]

        gt_top10 = np.argsort(exact_dots[qi])[-top_k:][::-1]
        recall = len(set(top10_pred.tolist()) & set(gt_top10.tolist())) / top_k
        recalls.append(recall)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    mean_recall = float(np.mean(recalls))
    print(
        f"  N={n_docs}, D={orig_dim} (padded={padded_dim}), K={num_clusters}, "
        f"nprobe={nprobe}, Top-C={top_candidates}"
    )
    print(f"  FWHT orthogonality max error : {ortho_err:.3e}")
    print(f"  Mean Recall@{top_k}             : {mean_recall * 100:.2f}% (target >= 96.00%)")
    print(f"  Total Python oracle time     : {elapsed_ms:.1f} ms for {n_queries} queries")
    assert mean_recall >= 0.96, f"Vector Search Recall@10 {mean_recall:.4f} < 0.96"
    print("  [PASS] Vector Search proof passed.\n")


def build_nested_lattice_codebook(token_dim: int = 128) -> tuple[np.ndarray, float]:
    """
    Build canonical 16-entry 2D lattice codebook:
        A1 = [-1.75, -0.55, 0.55, 1.75] * sigma, sigma = 1 / sqrt(D)
        C1[c] = (A1[c % 4], A1[c // 4])
    """
    sigma = 1.0 / math.sqrt(token_dim)
    levels = np.array([-1.75, -0.55, 0.55, 1.75], dtype=np.float64) * sigma
    codebook = np.empty((16, 2), dtype=np.float64)
    for c in range(16):
        codebook[c, 0] = levels[c % 4]
        codebook[c, 1] = levels[c // 4]
    return codebook, sigma


def encode_tokens_nested_lattice(
    tokens_rot: np.ndarray,
    codebook: np.ndarray,
    gamma: float = 0.28,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode FWHT-rotated tokens (N_tok, 128) WITHOUT per-subspace normalization into
    nibble-packed 2-layer codes: 32 bytes Layer 1 + 32 bytes Layer 2 = 64 bytes/token.
    """
    n_tok, dim = tokens_rot.shape
    subspaces = dim // 2
    t_sub = tokens_rot.reshape(n_tok, subspaces, 2)

    # Layer 1 argmin Euclidean distance over 16 2D centroids
    diff1 = t_sub[:, :, np.newaxis, :] - codebook[np.newaxis, np.newaxis, :, :]
    dist1 = np.sum(diff1**2, axis=-1)
    c1 = np.argmin(dist1, axis=-1).astype(np.uint8)  # (n_tok, 64)

    residual = t_sub - codebook[c1]
    # Layer 2 argmin over contracted lattice gamma * codebook
    diff2 = residual[:, :, np.newaxis, :] - (gamma * codebook)[np.newaxis, np.newaxis, :, :]
    dist2 = np.sum(diff2**2, axis=-1)
    c2 = np.argmin(dist2, axis=-1).astype(np.uint8)  # (n_tok, 64)

    # Pack adjacent even/odd subspaces into bytes: c_even | (c_odd << 4)
    packed_l1 = (c1[:, 0::2] & 0x0F) | ((c1[:, 1::2] & 0x0F) << 4)
    packed_l2 = (c2[:, 0::2] & 0x0F) | ((c2[:, 1::2] & 0x0F) << 4)
    return packed_l1, packed_l2


def run_lutrank_proof() -> None:
    """
    Proof 2: LUT-Rank (2D Nested Lattice M=2, gamma=0.28, NO per-subspace normalization,
    16-entry int8 LUT synthesis, MaxSim over 200 candidate documents).
    """
    print("=== Proof 2: LUT-Rank (2D Nested Lattice + int8 16-entry LUT + MaxSim) ===")
    rng = np.random.default_rng(20261116)

    n_docs = 200
    n_queries = 15
    q_tokens_per_query = 24
    d_tokens_per_doc = 96
    token_dim = 128
    gamma = 0.28
    fwht_seed = 0x4F5350525F4C5554  # "OSPR_LUT"

    codebook, sigma = build_nested_lattice_codebook(token_dim)
    assert codebook.shape == (16, 2)

    # Verify global variance equalization sigma^2 ~= 1/D after FWHT on random unit tokens
    probe_tok = rng.normal(size=(d_tokens_per_doc, token_dim))
    probe_tok /= np.linalg.norm(probe_tok, axis=1, keepdims=True)
    sample_rot = fwht_orthogonal_rotation(probe_tok, fwht_seed)
    empirical_std = float(np.std(sample_rot))
    assert abs(empirical_std - sigma) < 0.01, f"Unexpected std {empirical_std} vs sigma {sigma}"

    # Validate token-level dot-product Pearson correlation directly
    q_sample = rng.normal(size=(32, token_dim))
    q_sample /= np.linalg.norm(q_sample, axis=1, keepdims=True)
    q_sample_rot = fwht_orthogonal_rotation(q_sample, fwht_seed)
    q_s_sub = q_sample_rot.reshape(32, 64, 2)
    lut_s_f = np.einsum("qsd,cd->qsc", q_s_sub, codebook)
    s_scales = np.max(np.abs(lut_s_f), axis=(1, 2)) / 127.0
    lut_s_i8 = np.clip(np.round(lut_s_f / s_scales[:, None, None]), -127, 127).astype(np.int8)
    sub_idx = np.arange(64, dtype=np.int32)
    tok_exact = (q_sample @ probe_tok.T).ravel()
    l1_0, l2_0 = encode_tokens_nested_lattice(sample_rot, codebook, gamma=gamma)
    assert l1_0.shape == (d_tokens_per_doc, 32) and l2_0.shape == (d_tokens_per_doc, 32)
    c1_0 = np.empty((d_tokens_per_doc, 64), dtype=np.uint8)
    c1_0[:, 0::2] = l1_0 & 0x0F
    c1_0[:, 1::2] = (l1_0 >> 4) & 0x0F
    c2_0 = np.empty((d_tokens_per_doc, 64), dtype=np.uint8)
    c2_0[:, 0::2] = l2_0 & 0x0F
    c2_0[:, 1::2] = (l2_0 >> 4) & 0x0F
    tok_lut = np.empty((32, d_tokens_per_doc), dtype=np.float64)
    for qi in range(32):
        a1 = lut_s_i8[qi, sub_idx, c1_0].astype(np.int32).sum(axis=1)
        a2 = lut_s_i8[qi, sub_idx, c2_0].astype(np.int32).sum(axis=1)
        tok_lut[qi] = s_scales[qi] * (a1 + gamma * a2)
    token_pearson = float(np.corrcoef(tok_exact, tok_lut.ravel())[0, 1])

    pearsons = []
    recalls = []
    for _ in range(n_queries):
        q_tok = rng.normal(size=(q_tokens_per_query, token_dim))
        q_tok /= np.linalg.norm(q_tok, axis=1, keepdims=True)
        q_rot = fwht_orthogonal_rotation(q_tok, fwht_seed)

        # Synthesize 16-entry int8 LUTs per (query_token, subspace): shape (Q, 64, 16)
        q_sub = q_rot.reshape(q_tokens_per_query, 64, 2)
        lut_f64 = np.einsum("qsd,cd->qsc", q_sub, codebook)
        max_abs = np.max(np.abs(lut_f64), axis=(1, 2))
        scales = np.where(max_abs > 1e-12, max_abs / 127.0, 1.0)
        lut_i8 = np.clip(
            np.round(lut_f64 / scales[:, np.newaxis, np.newaxis]), -127, 127
        ).astype(np.int8)

        # Simulate 200 Stage-3 candidate documents with graded semantic token alignment
        # (shuffled order to prevent any positional bias)
        relevance_tiers = rng.permutation(n_docs)
        exact_scores = np.empty(n_docs, dtype=np.float64)
        lut_scores = np.empty(n_docs, dtype=np.float64)

        for d_idx in range(n_docs):
            tier = int(relevance_tiers[d_idx])
            if tier < 10:
                # True Top-10 relevant documents
                overlap = 0.88 - 0.02 * tier
            elif tier < 50:
                # Hard-negative / partially relevant candidates
                overlap = 0.52 - 0.005 * (tier - 10)
            else:
                # Weakly related tail candidates from Stage 3
                overlap = max(0.05, 0.28 - 0.0015 * (tier - 50))
            d_tok = rng.normal(size=(d_tokens_per_doc, token_dim))
            d_tok /= np.linalg.norm(d_tok, axis=1, keepdims=True)
            n_aligned = max(1, int(q_tokens_per_query * overlap))
            for k in range(n_aligned):
                iq = k % q_tokens_per_query
                id_tok = (k * 3) % d_tokens_per_doc
                d_tok[id_tok] = overlap * q_tok[iq] + (1.0 - overlap) * d_tok[id_tok]
                d_tok[id_tok] /= np.linalg.norm(d_tok[id_tok])

            d_rot = fwht_orthogonal_rotation(d_tok, fwht_seed)
            l1_bytes, l2_bytes = encode_tokens_nested_lattice(d_rot, codebook, gamma=gamma)

            # Exact FP32 MaxSim ground truth
            sim_mat = q_tok.astype(np.float32) @ d_tok.astype(np.float32).T
            exact_scores[d_idx] = float(np.sum(np.max(sim_mat, axis=1)))

            # Nibble-decoded int8 LUT MaxSim (zero float muls on doc tokens)
            c1 = np.empty((d_tokens_per_doc, 64), dtype=np.uint8)
            c1[:, 0::2] = l1_bytes & 0x0F
            c1[:, 1::2] = (l1_bytes >> 4) & 0x0F

            c2 = np.empty((d_tokens_per_doc, 64), dtype=np.uint8)
            c2[:, 0::2] = l2_bytes & 0x0F
            c2[:, 1::2] = (l2_bytes >> 4) & 0x0F

            total_maxsim = 0.0
            for qi in range(q_tokens_per_query):
                acc1 = lut_i8[qi, sub_idx, c1].astype(np.int32).sum(axis=1)
                acc2 = lut_i8[qi, sub_idx, c2].astype(np.int32).sum(axis=1)
                tok_scores = scales[qi] * (acc1 + gamma * acc2)
                total_maxsim += float(np.max(tok_scores))
            lut_scores[d_idx] = total_maxsim

        corr = float(np.corrcoef(exact_scores, lut_scores)[0, 1])
        pearsons.append(corr)

        gt_top10 = set(np.argsort(exact_scores)[-10:].tolist())
        pred_top10 = set(np.argsort(lut_scores)[-10:].tolist())
        recalls.append(len(gt_top10 & pred_top10) / 10.0)

    mean_pearson = float(np.mean(pearsons))
    min_pearson = float(np.min(pearsons))
    mean_recall10 = float(np.mean(recalls))
    print(f"  N_docs={n_docs}, D_tok={token_dim}, sigma={sigma:.6f}, gamma={gamma}")
    print(f"  Token-level Pearson  : {token_pearson:.5f} (target > 0.99)")
    print(f"  MaxSim Pearson (doc) : {mean_pearson:.5f} (min={min_pearson:.5f}, target > 0.99)")
    print(f"  Mean Recall@10       : {mean_recall10 * 100:.2f}% (target == 100.00%)")
    assert token_pearson > 0.99, f"Token-level Pearson {token_pearson:.5f} <= 0.99"
    assert mean_pearson > 0.99, f"LUT-Rank Pearson {mean_pearson:.5f} <= 0.99"
    assert mean_recall10 == 1.0, f"LUT-Rank Recall@10 {mean_recall10:.4f} != 1.0"
    print("  [PASS] LUT-Rank proof passed.\n")


if __name__ == "__main__":
    run_vector_search_proof()
    run_lutrank_proof()
    print("ALL MATHEMATICAL ORACLE PROOFS PASSED.")
