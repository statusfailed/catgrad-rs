{
  description = "Catgrad - A Categorical Deep Learning Compiler";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      manifest = (pkgs.lib.importTOML ./Cargo.toml).package;
    in {
      packages.default = pkgs.rustPlatform.buildRustPackage {
        pname = manifest.name;
        version = manifest.version;
        src = ./.;
        cargoLock.lockFile = ./Cargo.lock;

        meta = with pkgs.lib; {
          description = manifest.description;
          license = licenses.mit;
        };
      };

      devShells.default = pkgs.mkShell {
        nativeBuildInputs = with pkgs; [
          cargo
          rustc
          rustfmt
          clippy
        ];
      };
    });
}
