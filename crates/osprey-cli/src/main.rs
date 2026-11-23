//! OspreyDB standalone CLI binary (`import`, `search`, `rerank`, `delete`, `inspect`, `doctor`, `bench`, `completions`).

use std::process::ExitCode;

use clap::{CommandFactory, FromArgMatches};
use osprey_cli::cli::{OspreyCli, OspreyCommand};
use osprey_cli::commands;
use osprey_cli::output::{self, CliContext, CliError, CliResult};
use osprey_simd::{active_simd_backend, detect_best_simd_level, SimdLevel};
use serde::Serialize;
use yansi::{Condition, Paint};

#[derive(Debug, Serialize)]
struct JsonErrorPayload {
    error: String,
    exit_code: u8,
}

fn build_cli_version() -> &'static str {
    match detect_best_simd_level() {
        SimdLevel::Avx512 => concat!(env!("CARGO_PKG_VERSION"), " (simd: avx512_vpopcntdq)"),
        SimdLevel::Avx2 => concat!(env!("CARGO_PKG_VERSION"), " (simd: avx2)"),
        SimdLevel::Neon => concat!(env!("CARGO_PKG_VERSION"), " (simd: neon)"),
        SimdLevel::Scalar => concat!(env!("CARGO_PKG_VERSION"), " (simd: scalar)"),
    }
}

fn build_cli_command() -> clap::Command {
    let ver = build_cli_version();
    OspreyCli::command().version(ver).long_version(ver)
}

fn main() -> ExitCode {
    yansi::whenever(Condition::TTY_AND_COLOR);
    let matches = build_cli_command().get_matches();
    let cli = OspreyCli::from_arg_matches(&matches).unwrap_or_else(|err| err.exit());
    let ctx = CliContext {
        json: cli.json,
        verbose: cli.verbose,
        threads: cli.threads,
    };

    if let Err(err) = dispatch(cli, &ctx) {
        if ctx.json {
            let payload = JsonErrorPayload {
                error: err.to_string(),
                exit_code: err.numeric_code(),
            };
            if let Ok(json) = serde_json::to_string_pretty(&payload) {
                eprintln!("{json}");
            } else {
                eprintln!(
                    "{{\"error\":\"{err}\",\"exit_code\":{}}}",
                    err.numeric_code()
                );
            }
        } else {
            eprintln!("{} {err}", "error:".red().bright().bold());
        }
        return err.exit_code();
    }

    ExitCode::from(output::EXIT_OK)
}

fn dispatch(cli: OspreyCli, ctx: &CliContext) -> CliResult<()> {
    if cli.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global()
            .map_err(|err| {
                CliError::Io(format!("failed to initialize rayon thread pool: {err}"))
            })?;
    }

    ctx.log_debug(format!(
        "osprey initialized (simd={}, json={}, verbose={}, threads={})",
        active_simd_backend(),
        ctx.json,
        ctx.verbose,
        ctx.threads
    ));

    match cli.command {
        OspreyCommand::Doctor(args) => commands::doctor::run(&args, ctx),
        OspreyCommand::Import(args) => commands::import::run(&args, ctx),
        OspreyCommand::Search(args) => commands::search::run(&args, ctx),
        OspreyCommand::Rerank(args) => commands::rerank::run(&args, ctx),
        OspreyCommand::Delete(args) => commands::delete::run(&args, ctx),
        OspreyCommand::Inspect(args) => commands::inspect::run(&args, ctx),
        OspreyCommand::Bench(args) => commands::bench::run(&args, ctx),
        OspreyCommand::Completions(args) => {
            let mut cmd = build_cli_command();
            clap_complete::generate(args.shell, &mut cmd, "osprey", &mut std::io::stdout());
            Ok(())
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn clap_parses_all_seven_subcommands_with_global_flags() {
        let doc = OspreyCli::try_parse_from(["osprey", "--json", "-vv", "doctor"]).unwrap();
        assert!(doc.json);
        assert_eq!(doc.verbose, 2);
        assert!(matches!(doc.command, OspreyCommand::Doctor(_)));

        let imp = OspreyCli::try_parse_from([
            "osprey",
            "import",
            "--input",
            "data.parquet",
            "--output",
            "./out",
            "--clusters",
            "64",
            "--page-size-kb",
            "16",
            "--with-lutrank",
        ])
        .unwrap();
        assert!(matches!(imp.command, OspreyCommand::Import(_)));

        let srch = OspreyCli::try_parse_from([
            "osprey",
            "search",
            "--index",
            "seg.osi",
            "--slabs",
            "seg.osv",
            "--query",
            "[0.1,0.2]",
        ])
        .unwrap();
        assert!(matches!(srch.command, OspreyCommand::Search(_)));

        let rrk = OspreyCli::try_parse_from([
            "osprey",
            "rerank",
            "--rerank-store",
            "seg.osr",
            "--query-tokens",
            "[[0.1]]",
            "--doc-ids",
            "1,2,3",
        ])
        .unwrap();
        assert!(matches!(rrk.command, OspreyCommand::Rerank(_)));

        let del =
            OspreyCli::try_parse_from(["osprey", "delete", "--index", "seg.osi", "--ids", "10,20"])
                .unwrap();
        assert!(matches!(del.command, OspreyCommand::Delete(_)));

        let insp =
            OspreyCli::try_parse_from(["osprey", "inspect", "-f", "seg.osi", "seg.osv"]).unwrap();
        assert!(matches!(insp.command, OspreyCommand::Inspect(_)));

        let bnc = OspreyCli::try_parse_from(["osprey", "bench", "--with-lutrank"]).unwrap();
        assert!(matches!(bnc.command, OspreyCommand::Bench(_)));

        let comp = OspreyCli::try_parse_from(["osprey", "completions", "bash"]).unwrap();
        assert!(matches!(comp.command, OspreyCommand::Completions(_)));

        let ver = build_cli_version();
        assert!(ver.contains(env!("CARGO_PKG_VERSION")));
        assert!(ver.contains(active_simd_backend()));
    }
}
