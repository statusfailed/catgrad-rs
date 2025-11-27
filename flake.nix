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
      manifest = (pkgs.lib.importTOML ./Cargo.toml).workspace.package;

      mkCatgrad = {
        withExamples ? true,
        withMlir ? false,
        llvmPackages ? pkgs.llvmPackages_21,
      }:
        pkgs.rustPlatform.buildRustPackage {
          pname = "catgrad";
          version = manifest.version;
          src = ./.;
          cargoDeps = pkgs.rustPlatform.importCargoLock {
            lockFile = ./Cargo.lock;
          };

          cargoBuildFlags =
            ["--workspace"]
            ++ pkgs.lib.optionals withExamples ["--examples"];

          # libraries
          propagatedBuildInputs =
            []
            ++ pkgs.lib.optionals withMlir (with llvmPackages; [
              mlir # for mlir_c_runner_utils
            ]);

          # executables
          propagatedNativeBuildInputs =
            []
            ++ pkgs.lib.optionals withMlir (with llvmPackages; [
              mlir
              llvm
            ]);

          # include examples
          postInstall = pkgs.lib.optionalString withExamples ''
            mkdir -p $out/bin
            find target -path '*/release/examples/*' -executable -type f \
              ! -name '*-????????????????' \
              -exec install -Dm755 {} $out/bin/ \;
          '';

          meta = with pkgs.lib; {
            description = manifest.description;
            license = licenses.mit;
            mainProgram = "llama";
          };
        };
    in {
      packages = {
        default = mkCatgrad {};
        minimal = mkCatgrad {withExamples = false;};
        withMlir = mkCatgrad {withMlir = true;};
      };
    });
}
