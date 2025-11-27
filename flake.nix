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
      }: let
        mlirLibs = with llvmPackages; [mlir llvm];
        mlirBins = with llvmPackages; [mlir llvm clang];
      in
        pkgs.rustPlatform.buildRustPackage {
          pname = "catgrad";
          version = manifest.version;
          src = ./.;
          cargoDeps = pkgs.rustPlatform.importCargoLock {
            lockFile = ./Cargo.lock;
          };

          # disable cargo-auditable wrapper
          auditable = false;

          cargoBuildFlags =
            ["--workspace"]
            ++ pkgs.lib.optionals withExamples ["--examples"];

          # if withMlir, we need to wrap binaries to add mlir to PATH/LIBRARY_PATH
          nativeBuildInputs =
            if withMlir
            then [pkgs.makeWrapper]
            else [];

          propagatedBuildInputs =
            []
            ++ pkgs.lib.optionals withMlir mlirLibs;

          propagatedNativeBuildInputs =
            []
            ++ pkgs.lib.optionals withMlir mlirBins;

          # include examples and wrap MLIR entrypoints when available
          postInstall =
            # copy examples if requested (except test binaries)
            pkgs.lib.optionalString withExamples ''
              mkdir -p $out/bin
              find target -path '*/release/examples/*' -executable -type f \
                ! -name '*-????????????????' \
                -exec install -Dm755 {} $out/bin/ \;
            ''
            # mlir-llm needs mlir toolchain + libs at runtime, wrap binary to add them to env
            + pkgs.lib.optionalString withMlir ''
              if [ -x "$out/bin/mlir-llm" ]; then
                wrapProgram "$out/bin/mlir-llm" \
                  --prefix PATH : "${pkgs.lib.makeBinPath mlirBins}" \
                  --prefix LIBRARY_PATH : "${pkgs.lib.makeLibraryPath mlirLibs}" \
                  --prefix LD_LIBRARY_PATH : "${pkgs.lib.makeLibraryPath mlirLibs}" \
                  --prefix DYLD_LIBRARY_PATH : "${pkgs.lib.makeLibraryPath mlirLibs}" \
                  --prefix NIX_LDFLAGS " " "-L${pkgs.lib.makeLibraryPath mlirLibs}"
              fi
            '';

          meta = with pkgs.lib; {
            description = manifest.description;
            license = licenses.mit;
            mainProgram =
              if withMlir
              then "mlir-llm"
              else "llm";
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
