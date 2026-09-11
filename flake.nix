{
  description = "Python development environment with uv";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.my-config.url = "git+file:///home/pjjimiso/nixos-config"; # bootdev cli

  outputs = { self, nixpkgs, my-config }:
    let
      system = "x86_64-linux"; # Adjust for your architecture
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python312;
      bootdev = my-config.packages.${system}.bootdev;
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ pkgs.uv python bootdev ];

        # Use the Nix-provided interpreter instead of uv's prebuilt downloads,
        # which are generic dynamically linked binaries NixOS cannot execute.
        UV_PYTHON = "${python}/bin/python3";
        UV_PYTHON_DOWNLOADS = "never";

        # Let wheels with bundled native code find their C libraries.
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
          stdenv.cc.cc
          zlib
          # Add extra native C libraries here (e.g., glib, libglvnd for OpenCV/PyTorch)
        ]);
      };
    };
}
