use std::env;
use std::fs;
use std::process;

use catgrad_mlir::codegen::codegen;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        eprintln!("Usage: {} <input.mlir> <output.so>", args[0]);
        process::exit(1);
    }

    let mlir_file = &args[1];
    let output_so = &args[2];

    // Read MLIR text from file
    let mlir_text = match fs::read_to_string(mlir_file) {
        Ok(content) => content,
        Err(err) => {
            eprintln!("Error reading {}: {}", mlir_file, err);
            process::exit(1);
        }
    };

    // Compile to shared library
    match codegen(&mlir_text, output_so) {
        Ok(()) => {
            println!("Successfully compiled {} to {}", mlir_file, output_so);
        }
        Err(err) => {
            eprintln!("Compilation failed: {:?}", err);
            process::exit(1);
        }
    }
}
