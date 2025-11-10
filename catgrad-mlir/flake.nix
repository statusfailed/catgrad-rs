{
  description = "Development environment with MLIR tools for catgrad-core";

  inputs = {
    root.url = "path:../";           # parent flake
    nixpkgs.follows = "root/nixpkgs";  # use parent's nixpkgs pin
    flake-utils.follows = "root/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
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
            lld

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
