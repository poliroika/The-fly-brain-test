use std::process::ExitCode;

fn print_help() {
    println!(
        "flybrain {} — FlyBrain Optimizer CLI\n\n\
         Usage:\n  \
         flybrain <command> [args]\n\n\
         Commands:\n  \
         build       Build a compressed FlyBrain graph (Phase 1; not yet implemented)\n  \
         sim         Run simulation pretraining (Phase 6; not yet implemented)\n  \
         bench       Run benchmark suite (Phase 10; not yet implemented)\n  \
         report      Build the final results report (Phase 11; not yet implemented)\n  \
         help        Show this help\n  \
         version     Show version\n\n\
         Phase 0 ships only the skeleton. See PLAN.md for the full roadmap.",
        env!("CARGO_PKG_VERSION")
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("help");
    match cmd {
        "help" | "--help" | "-h" => {
            print_help();
            ExitCode::SUCCESS
        }
        "version" | "--version" | "-V" => {
            println!("flybrain {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        "build" | "sim" | "bench" | "report" => {
            eprintln!(
                "flybrain {cmd}: not yet implemented in Phase 0. See PLAN.md for the roadmap."
            );
            ExitCode::from(2)
        }
        other => {
            eprintln!("unknown command: {other}");
            print_help();
            ExitCode::FAILURE
        }
    }
}
