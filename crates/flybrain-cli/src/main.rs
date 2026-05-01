use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use flybrain_graph::builder::{
    build, build_default_with, BuildRequest, BuildSource, CompressionMethod,
};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "flybrain",
    version,
    about = "FlyBrain Optimizer CLI",
    long_about = "\
Build, run, benchmark, and report on the FlyBrain Optimizer pipeline.

Phase 1 ships the `build` subcommand. The `sim`, `bench`, and `report`
subcommands are placeholders for later phases — see PLAN.md for the roadmap."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Build a compressed FlyBrain graph from a connectome source.
    Build(BuildArgs),
    /// Run simulation pretraining (Phase 6; not yet implemented).
    Sim,
    /// Run the benchmark suite (Phase 10; not yet implemented).
    Bench,
    /// Build the final results report (Phase 11; not yet implemented).
    Report,
}

#[derive(Parser)]
struct BuildArgs {
    /// Connectome source.
    #[arg(long, value_enum, default_value_t = SourceKind::Synthetic)]
    source: SourceKind,

    /// Synthetic only: number of source-graph nodes (default 2048).
    #[arg(long, default_value_t = 2048)]
    num_nodes: usize,

    /// Zenodo only: directory with `neurons.csv` and `connections.csv`.
    #[arg(long)]
    zenodo_dir: Option<PathBuf>,

    /// Zenodo only: explicit path to neurons CSV (overrides --zenodo-dir).
    #[arg(long, requires = "zenodo_connections")]
    zenodo_neurons: Option<PathBuf>,

    /// Zenodo only: explicit path to connections CSV (overrides --zenodo-dir).
    #[arg(long, requires = "zenodo_neurons")]
    zenodo_connections: Option<PathBuf>,

    /// Compression method.
    #[arg(long, value_enum, default_value_t = MethodKind::Louvain)]
    method: MethodKind,

    /// Target number of clusters K.
    #[arg(long, short = 'k', default_value_t = 64)]
    k: usize,

    /// Random seed (used by Louvain / Leiden / Spectral).
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Output `.fbg` path. Defaults to `data/flybrain/fly_graph_<K>.fbg`.
    #[arg(long, short = 'o')]
    output: Option<PathBuf>,

    /// Run the K∈{32,64,128,256} batch instead of a single build.
    #[arg(long)]
    all: bool,

    /// Output directory for `--all` (defaults to `data/flybrain`).
    #[arg(long, default_value = "data/flybrain")]
    out_dir: PathBuf,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum SourceKind {
    Synthetic,
    Zenodo,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum MethodKind {
    RegionAgg,
    CelltypeAgg,
    Louvain,
    Leiden,
    Spectral,
}

impl From<MethodKind> for CompressionMethod {
    fn from(k: MethodKind) -> Self {
        match k {
            MethodKind::RegionAgg => CompressionMethod::RegionAgg,
            MethodKind::CelltypeAgg => CompressionMethod::CelltypeAgg,
            MethodKind::Louvain => CompressionMethod::Louvain,
            MethodKind::Leiden => CompressionMethod::Leiden,
            MethodKind::Spectral => CompressionMethod::Spectral,
        }
    }
}

fn run_build(args: BuildArgs) -> Result<()> {
    if args.all {
        let reports =
            build_default_with(&args.out_dir, args.num_nodes, args.seed, args.method.into())?;
        for r in &reports {
            println!(
                "[K={:>3}] {} src={} edges={} compressed_edges={} Q={:.4} -> {}",
                r.target_k,
                r.method,
                r.source_num_nodes,
                r.source_num_edges,
                r.compressed_num_edges,
                r.modularity_directed,
                r.fbg_path.display()
            );
        }
        return Ok(());
    }

    let source = match args.source {
        SourceKind::Synthetic => BuildSource::Synthetic {
            num_nodes: args.num_nodes,
            seed: args.seed,
        },
        SourceKind::Zenodo => {
            if let (Some(neurons), Some(connections)) =
                (args.zenodo_neurons.clone(), args.zenodo_connections.clone())
            {
                BuildSource::ZenodoCsv {
                    neurons,
                    connections,
                }
            } else if let Some(dir) = args.zenodo_dir.clone() {
                BuildSource::ZenodoDir { dir }
            } else {
                anyhow::bail!(
                    "--source zenodo requires either --zenodo-dir or both --zenodo-neurons and --zenodo-connections"
                );
            }
        }
    };

    let output = args
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("data/flybrain/fly_graph_{}.fbg", args.k)));

    let req = BuildRequest {
        source,
        method: args.method.into(),
        target_k: args.k,
        seed: args.seed,
        output,
    };
    let report = build(&req)?;

    let json = serde_json::to_string_pretty(&report)?;
    println!("{json}");
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Some(Command::Build(args)) => run_build(args),
        Some(Command::Sim) | Some(Command::Bench) | Some(Command::Report) => {
            eprintln!("not yet implemented; see PLAN.md");
            return ExitCode::from(2);
        }
        None => {
            // Mimic the previous default: print help + version on bare invocation.
            println!(
                "flybrain {} — FlyBrain Optimizer CLI",
                env!("CARGO_PKG_VERSION")
            );
            println!("run `flybrain --help` for usage");
            return ExitCode::SUCCESS;
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
