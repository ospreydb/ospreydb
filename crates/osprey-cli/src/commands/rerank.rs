//! `osprey rerank`: standalone CPU ColBERT `LUT-Rank` reranking over
//! external query token tensors and candidate `doc_id`s using `.osr`, plus optional
//! end-to-end Vector Search + LUT-Rank execution.

use std::collections::HashSet;
use std::time::Instant;

use comfy_table::{Cell, CellAlignment};
use osprey_core::{OspreyError, OsrStore, QuantizedQueryLut};
use osprey_simd::{SimdBackend, SimdEngine};
use serde::Serialize;

use crate::cli::{RerankArgs, SearchArgs};
use crate::commands::search;
use crate::output::{
    new_rounded_table, paint_accent, paint_dim, paint_pass, paint_score, parse_doc_ids_file,
    parse_query_tokens_input, print_indented_table, print_json, print_kv, print_section,
    CliContext, CliError, CliResult,
};

#[derive(Debug, Serialize)]
struct RerankHit {
    rank: usize,
    doc_id: u64,
    maxsim_score: f32,
    found_in_osr: bool,
    doc_token_count: u16,
}

#[derive(Debug, Serialize)]
struct RerankReport {
    rerank_store: String,
    simd_tier: String,
    query_tokens: usize,
    token_dim: u16,
    gamma: f32,
    candidates_evaluated: usize,
    lut_synthesis_us: u64,
    maxsim_ms: f64,
    hits: Vec<RerankHit>,
}

/// Executes `osprey rerank`.
pub fn run(args: &RerankArgs, ctx: &CliContext) -> CliResult<()> {
    let mut candidate_ids = args.doc_ids.clone();
    if let Some(cfile) = &args.candidates_file {
        candidate_ids.extend(parse_doc_ids_file(cfile)?);
    }

    // Deduplicate candidate IDs preserving order.
    let mut seen = HashSet::with_capacity(candidate_ids.len());
    candidate_ids.retain(|&id| seen.insert(id));

    if !candidate_ids.is_empty() {
        return run_standalone_cpu_rerank(args, &candidate_ids, ctx);
    }

    // Fall back to end-to-end Search + LUT-Rank if `--index`, `--slabs`, and `--query` are provided.
    if let (Some(index), Some(slabs), Some(query)) = (&args.index, &args.slabs, &args.query) {
        let search_args = SearchArgs {
            index: index.clone(),
            slabs: slabs.clone(),
            query: query.clone(),
            top_k: args.top_k,
            nprobe: args.nprobe,
            candidates: args.candidates,
            refine_multiplier: args.refine_multiplier,
            cache_mb: args.cache_mb,
            rerank_store: Some(args.rerank_store.clone()),
            query_tokens: Some(args.query_tokens.clone()),
            rerank_top_m: args.rerank_top_m,
        };
        return search::run(&search_args, ctx);
    }

    Err(CliError::InvalidArgs(
        "provide candidate doc IDs via --doc-ids / --candidates-file, or provide --index, --slabs, and --query for end-to-end search+rerank"
            .to_string(),
    ))
}

fn run_standalone_cpu_rerank(
    args: &RerankArgs,
    candidate_ids: &[u64],
    ctx: &CliContext,
) -> CliResult<()> {
    if args.top_k == 0 {
        return Err(CliError::InvalidArgs(
            "--top-k must be greater than 0".to_string(),
        ));
    }

    ctx.log_info(format!(
        "Memory-mapping .osr store '{}' for standalone CPU LUT-Rank rerank...",
        args.rerank_store.display()
    ));
    let osr = OsrStore::open_mmap(&args.rerank_store)?;
    let hdr = *osr.header();
    let token_dim = usize::from(hdr.token_dim);

    let q_tokens = parse_query_tokens_input(&args.query_tokens)?;
    let mut flat_tokens = Vec::with_capacity(q_tokens.len() * token_dim);
    for tok in &q_tokens {
        if tok.len() != token_dim {
            return Err(CliError::Core(OspreyError::InvalidDimension {
                dim: tok.len(),
            }));
        }
        flat_tokens.extend_from_slice(tok);
    }

    let simd = SimdEngine::new_auto();

    let t_lut = Instant::now();
    let query_lut = QuantizedQueryLut::synthesize(
        &flat_tokens,
        q_tokens.len(),
        token_dim,
        hdr.gamma,
        hdr.fwht_seed,
        &simd,
    )?;
    let lut_synthesis_us = u64::try_from(t_lut.elapsed().as_micros()).unwrap_or(u64::MAX);

    let t_score = Instant::now();
    let mut scored: Vec<(u64, f32, bool, u16)> = Vec::with_capacity(candidate_ids.len());
    for (idx, &doc_id) in candidate_ids.iter().enumerate() {
        if let Some(&next_id) = candidate_ids.get(idx + 1) {
            osr.prefetch_doc_id(next_id, &simd);
        }
        match (
            osr.lookup_doc(doc_id)?,
            osr.score_doc_id_maxsim(doc_id, &query_lut, &simd)?,
        ) {
            (Some(range), Some(score)) => {
                scored.push((doc_id, score, true, range.token_count));
            }
            _ => {
                scored.push((doc_id, f32::NEG_INFINITY, false, 0));
            }
        }
    }
    let maxsim_ms = t_score.elapsed().as_secs_f64() * 1_000.0;

    scored.sort_by(|a, b| {
        b.2.cmp(&a.2)
            .then_with(|| b.1.total_cmp(&a.1))
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.truncate(args.top_k);

    let hits: Vec<RerankHit> = scored
        .into_iter()
        .enumerate()
        .map(
            |(idx, (doc_id, maxsim_score, found_in_osr, doc_token_count))| RerankHit {
                rank: idx + 1,
                doc_id,
                maxsim_score,
                found_in_osr,
                doc_token_count,
            },
        )
        .collect();

    let report = RerankReport {
        rerank_store: args.rerank_store.display().to_string(),
        simd_tier: format!("{:?}", simd.level()),
        query_tokens: q_tokens.len(),
        token_dim: hdr.token_dim,
        gamma: hdr.gamma,
        candidates_evaluated: candidate_ids.len(),
        lut_synthesis_us,
        maxsim_ms,
        hits,
    };

    if ctx.json {
        print_json(&report)?;
    } else {
        print_rerank_human(&report);
    }

    Ok(())
}

fn print_rerank_human(report: &RerankReport) {
    let mut table = new_rounded_table(&["Rank", "Doc ID", "MaxSim", "Doc Tokens"]);
    for h in &report.hits {
        let score_cell = if h.found_in_osr {
            paint_score(h.maxsim_score, &format!("{:.5}", h.maxsim_score))
        } else {
            paint_dim("missing")
        };
        table.add_row(vec![
            Cell::new(h.rank).set_alignment(CellAlignment::Right),
            Cell::new(h.doc_id).set_alignment(CellAlignment::Right),
            Cell::new(score_cell).set_alignment(CellAlignment::Right),
            Cell::new(h.doc_token_count).set_alignment(CellAlignment::Right),
        ]);
    }
    print_indented_table(&table);

    print_section("Telemetry");
    print_kv("SIMD kernel", &paint_pass(&report.simd_tier));
    print_kv(
        "LUT int8 synth",
        &paint_accent(&format!("{} us", report.lut_synthesis_us)),
    );
    print_kv(
        "MaxSim scoring",
        &format!(
            "{:.3} ms ({} docs)",
            report.maxsim_ms, report.candidates_evaluated
        ),
    );
    println!();
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use osprey_core::{OsrDocumentTokens, DEFAULT_LUTRANK_GAMMA, LUTRANK_TOKEN_DIM};

    #[test]
    fn standalone_cpu_rerank_ranks_matching_doc_first() {
        let dir = tempfile::tempdir().unwrap();
        let osr_path = dir.path().join("tokens.osr");

        let mut d10 = vec![0.0_f32; 4 * LUTRANK_TOKEN_DIM];
        let mut d20 = vec![0.0_f32; 4 * LUTRANK_TOKEN_DIM];
        for t in 0..4 {
            d10[t * LUTRANK_TOKEN_DIM + t] = 1.0;
            d20[t * LUTRANK_TOKEN_DIM + 20 + t] = 1.0;
        }
        let docs = vec![
            OsrDocumentTokens {
                doc_id: 10,
                tokens: &d10,
                num_tokens: 4,
            },
            OsrDocumentTokens {
                doc_id: 20,
                tokens: &d20,
                num_tokens: 4,
            },
        ];
        let simd = SimdEngine::new_auto();
        let osr =
            OsrStore::build(&docs, LUTRANK_TOKEN_DIM, DEFAULT_LUTRANK_GAMMA, 55, &simd).unwrap();
        std::fs::write(&osr_path, osr.to_aligned_bytes().unwrap().as_slice()).unwrap();

        let mut q = vec![vec![0.0_f32; LUTRANK_TOKEN_DIM]; 2];
        q[0][20] = 1.0;
        q[1][21] = 1.0;
        let qt_path = dir.path().join("q_tokens.json");
        std::fs::write(&qt_path, serde_json::to_string(&q).unwrap()).unwrap();

        let cand_path = dir.path().join("cands.json");
        std::fs::write(&cand_path, "[10, 20, 999]").unwrap();

        let args = RerankArgs {
            rerank_store: osr_path,
            query_tokens: qt_path.display().to_string(),
            doc_ids: vec![],
            candidates_file: Some(cand_path),
            top_k: 3,
            index: None,
            slabs: None,
            query: None,
            nprobe: 16,
            candidates: 200,
            refine_multiplier: 0,
            cache_mb: 64,
            rerank_top_m: 200,
        };
        let ctx = CliContext {
            json: true,
            verbose: 0,
            threads: 1,
        };
        run(&args, &ctx).unwrap();
    }
}
