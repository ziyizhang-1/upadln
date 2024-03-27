mod cnn;
mod utils;

extern crate clap;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(author = "Ziyi Zhang", version = "0.1.0", about = None, long_about = None)]
struct Cmds {
    #[command(subcommand)]
    commands: SubCommands,
}

#[derive(Subcommand, Debug)]
enum SubCommands {
    Infer(Infer),
    Bench(Bench),
}

#[derive(Parser, Debug)]
struct Infer {
    #[arg(long, short)]
    model: String,
    #[arg(long, short)]
    input: String,
    #[arg(long, short, action, default_value_t = true)]
    real_time: bool,
    #[arg(long, short)]
    batch_size: Option<u32>,
}

#[derive(Parser, Debug)]
struct Bench {
    #[arg(long, short)]
    model: String,
    #[arg(long, short, action, default_value_t = true)]
    dummy: bool,
    #[arg(long, short)]
    input: String,
    #[arg(long, short, action, default_value_t = true)]
    real_time: bool,
    #[arg(long, short)]
    batch_size: Option<u32>,
}

fn main() {
    let cmds = Cmds::parse();
    match cmds.commands {
        SubCommands::Infer(ref args) => {
            cnn::types::run().unwrap();
        }
        SubCommands::Bench(ref args) => {}
    }
}
