# TODO: try https://github.com/nix-community/poetry2nix
{
  description = "A Python 3 project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShell = pkgs.mkShell {

          buildInputs = [
            pkgs.python311
            pkgs.poetry
            #            pkgs.python3Packages.click
            #            pkgs.python3Packages.opencv4
            #            pkgs.python3Packages.numpy
            #            pkgs.python3Packages.pytorch
            #            pkgs.python3Packages.easyocr
            darwin.apple_sdk.frameworks.Vision
                darwin.apple_sdk.frameworks.CoreML
                darwin.apple_sdk.frameworks.Quartz
          ];
          shellHook = ''
            mkdir -p .local
            export SHELL=${pkgs.zsh}/bin/zsh
            export PATH=.local/python3/bin:$PATH
            exec ${pkgs.zsh}/bin/zsh
            echo "Python 3 development environment (Nix for SDK, everything else impure)"
          '';
        };

        packages = {
          default = pkgs.python3Packages.buildPythonPackage {
            pname = "screenpear";
            version = "0.1.0";
            src = ./.;

            propagatedBuildInputs = [
              pkgs.python3Packages.click
              pkgs.python3Packages.opencv4
              pkgs.python3Packages.numpy
              pkgs.python3Packages.pytorch
              pkgs.python3Packages.easyocr
            ];

            meta = with pkgs.lib; {
              description = "A Python 3 project for OCR and image processing";
              license = licenses.asl20;
              maintainers = [ maintainers.dmitrykireev ];
            };
          };
        };
      }
    );
}
