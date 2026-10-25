{
  pkgs,
  rustToolchain,
  scripts,
}:
pkgs.mkShell {
  # A worktree's target starts as a clone of the main checkout's
  # (.githooks/post-checkout), newer than every source in it, so cargo has to
  # judge freshness by content: checksums need both of the first two, and
  # rustdoc reads dep-info only with the third. The gate sets them too: they
  # reach every unit's fingerprint, and a shell and a gate that disagreed would
  # rebuild everything at each alternation.
  CARGO_UNSTABLE_CHECKSUM_FRESHNESS = "true";
  CARGO_BUILD_FINGERPRINT = "content";
  CARGO_UNSTABLE_RUSTDOC_DEPINFO = "true";

  packages = [
    rustToolchain
    pkgs.deadnix
    pkgs.nixfmt
    pkgs.statix
  ]
  ++ builtins.attrValues scripts;

  shellHook = ''
    # Otherwise coreutils' fmt, the paragraph reflower, can shadow ours and
    # sit reading stdin.
    export PATH="${pkgs.lib.makeBinPath (builtins.attrValues scripts)}:$PATH"

    git config core.hooksPath .githooks 2>/dev/null || true
  '';
}
