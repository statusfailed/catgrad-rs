{
  description = "Development environment with MLIR tools for catgrad-core";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # MLIR and LLVM tools
            llvmPackages_21.mlir
            llvmPackages_21.llvm
            llvmPackages_21.clang
            
            # Rust toolchain
            rustc
            cargo
            rustfmt
            clippy
            
            # Python for ezmlir script
            python3
          ];
        };
      });
}
