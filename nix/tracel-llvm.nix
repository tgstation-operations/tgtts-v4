{ pkgs }:
# The LLVM that cubecl's CUDA runtime links to compile its kernels to PTX. Its
# build script, by tracel-llvm-bundler, would download this archive into the
# user's data directory and run the llvm-config in it; laid out as it leaves
# it, with the sentinel it checks, and pointed at by XDG_DATA_HOME, the
# download is skipped and the patched llvm-config runs on NixOS.
let
  version = "23.1.0-4";
in
pkgs.stdenv.mkDerivation {
  pname = "tracel-llvm";
  inherit version;
  src = pkgs.fetchurl {
    url = "https://github.com/tracel-ai/tracel-llvm/releases/download/v${version}/linux-x64.tar.xz";
    hash = "sha256-w5S7GxCseF8W09ktaIqTQ7qGn3IOnPnKPxq/L0AfFl8=";
  };
  nativeBuildInputs = [ pkgs.autoPatchelfHook ];
  buildInputs = [ pkgs.stdenv.cc.cc.lib ];
  dontConfigure = true;
  dontBuild = true;
  dontStrip = true;
  installPhase = ''
    dir=$out/tracel/tracel-llvm-${version}
    mkdir -p "$dir/bin"
    # The lld links beside it dangle: the archive holds no lld.
    cp bin/llvm-config "$dir/bin"
    cp -r include lib "$dir"
    touch "$dir/.tracel-llvm-installed"
  '';
}
