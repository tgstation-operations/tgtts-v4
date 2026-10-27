{ pkgs }:
# The hand-written GLSL compute kernels of kernels/glsl, compiled ahead of time to SPIR-V for
# Vulkan 1.3: per kernel, <name>.spv and its disassembly <name>.spvasm. The build fails unless
# every module passes spirv-val. Whether they compute the right thing is the device tests'.
let
  inherit (pkgs) lib;
in
pkgs.stdenvNoCC.mkDerivation {
  pname = "glsl-kernels";
  version = "0.1.0";

  src = lib.fileset.toSource {
    root = ../kernels/glsl;
    fileset = ../kernels/glsl;
  };

  nativeBuildInputs = [
    pkgs.glslang
    pkgs.spirv-tools
  ];

  buildPhase = ''
    runHook preBuild
    mkdir -p $out
    for source in *.comp; do
      name=''${source%.comp}
      glslangValidator --quiet --target-env vulkan1.3 -o "$out/$name.spv" "$source"
      spirv-val --target-env vulkan1.3 "$out/$name.spv"
      spirv-dis "$out/$name.spv" -o "$out/$name.spvasm"
    done
    runHook postBuild
  '';

  dontInstall = true;

  meta.license = lib.licenses.agpl3Only;
}
