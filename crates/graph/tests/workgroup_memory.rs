//! The workgroup memory the kernels' pipelines take, as the runtime computes it from their SPIR-V
//! before the driver sees one (`vulkan::spirv::workgroup_memory`), and refuses past the device's
//! limit. Built with the `device` feature, and run with `GLSL_KERNELS` and `VULKAN_KERNELS` naming
//! the compiled kernels, as `nix run .#device-tests` sets them; at VoxCPM2's shapes with
//! `VOXCPM2_DIR` naming the checkpoint.
#![cfg(feature = "device")]

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    ffi::CString,
    fs,
    path::PathBuf,
};

use graph::{
    decode::{Batch, SUBGROUPS, Tuning},
    graph::Graph,
    kernels::{KERNELS_VAR, MLIR_KERNELS_VAR, Manifest},
    lm_side::{self, LmSideShape, lm_side_batch_graph},
    local::{self, PatchConfig, dit::sampler_graph, encoder::encoder_graph},
    patch::patch_graph,
    prefill::MAX_ROWS,
    prompt::prompt_graphs,
    vae::{VaeConfig, VaeShape, vae_graph},
};
use serde::Deserialize;
use testkit::checkpoint_dir;
use vulkan::{Context, SpecValue, spirv};

fn kernels_dir(var: &str) -> PathBuf {
    let dir = std::env::var_os(var).unwrap_or_else(|| panic!("{var} names no directory"));
    PathBuf::from(dir)
}

/// The SPIR-V of the kernels by name, each read once: GLSL's where `GLSL_KERNELS` has it, and
/// otherwise from `VULKAN_KERNELS` with the entry point its manifest names.
#[derive(Default)]
struct Modules(HashMap<String, (Vec<u32>, CString)>);

impl Modules {
    fn get(&mut self, name: &str) -> &(Vec<u32>, CString) {
        self.0.entry(name.to_string()).or_insert_with(|| {
            let glsl = kernels_dir(KERNELS_VAR).join(format!("{name}.spv"));
            let (path, entry) = if glsl.exists() {
                (glsl, c"main".to_owned())
            } else {
                let mlir = kernels_dir(MLIR_KERNELS_VAR);
                let json = fs::read(mlir.join(format!("{name}.json"))).unwrap();
                let manifest: Manifest = serde_json::from_slice(&json).unwrap();
                let entry = CString::new(manifest.entry_point).unwrap();
                (mlir.join(format!("{name}.spv")), entry)
            };
            let bytes = fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            (spirv::words(&bytes).unwrap().0, entry)
        })
    }

    /// The bytes of workgroup memory `name` takes specialized with `constants`.
    fn bytes(&mut self, name: &str, constants: &[(u32, SpecValue)]) -> u64 {
        let (words, entry) = self.get(name);
        spirv::workgroup_memory(words, entry, constants)
            .unwrap_or_else(|e| panic!("{name} at {constants:?}: {e}"))
    }
}

/// The LM side for each batch and cut the decode step is tested and timed at, as far as it
/// takes them.
fn lm_side_cuts(side: &LmSideShape) -> Vec<Graph> {
    let mut graphs = Vec::new();
    for subgroup in SUBGROUPS {
        for rows in [1, 2, 3, 4, 8, 16] {
            let batch = Batch { rows, slots: rows };
            let base = Tuning::for_batch(subgroup, rows);
            let mut cuts = vec![base];
            for workgroup in [subgroup, 2 * subgroup, 256] {
                for pairs in [1, 2] {
                    for ahead in [0, 2] {
                        for stage in [16384, 32768, 49152, 65536] {
                            for vectors in [1, 2, 4, 8, 16] {
                                cuts.push(Tuning {
                                    workgroup,
                                    qkv_pairs: pairs,
                                    gate_up_pairs: pairs,
                                    residual_rows: pairs,
                                    ahead,
                                    stage,
                                    vectors,
                                    ..base
                                });
                            }
                        }
                    }
                }
            }
            let built = cuts
                .iter()
                .filter_map(|t| lm_side_batch_graph(side, t, batch).ok());
            graphs.extend(built.map(|g| g.graph));
        }
    }
    graphs
}

/// The parts of VoxCPM2's `config.json` the audio VAE is recorded from.
#[derive(Deserialize)]
struct AudioVae {
    audio_vae_config: VaeConfig,
}

/// The local stacks' tunings at `subgroup` as tests/local.rs checks them, those that fit
/// `max_shared_memory` as it requires, and the device's default.
fn local_cuts(subgroup: u32, report: &vulkan::DeviceReport) -> Vec<local::Tuning> {
    let base = local::Tuning::for_subgroup(subgroup);
    let cut = |workgroup, pairs, tiles, chunk, block, depth, groups| local::Tuning {
        workgroup,
        qkv_pairs: pairs,
        gate_up_pairs: pairs,
        o_tiles: tiles,
        down_tiles: tiles,
        chunk,
        block,
        depth,
        groups,
        ..base
    };
    let parts = |o_parts, down_parts, tuning| local::Tuning {
        o_parts,
        down_parts,
        ..tuning
    };
    let device = local::Tuning {
        subgroup,
        ..local::Tuning::for_device(report).unwrap()
    };
    let cuts = [
        base,
        device,
        cut(64, 1, 1, 256, 32, 4, 0),
        cut(128, 2, 2, 128, 32, 2, 0),
        cut(256, 2, 4, 64, 16, 1, 0),
        cut(256, 1, 1, 512, 16, 8, 40),
        cut(512, 1, 2, 128, 64, 4, 8),
        parts(1, 1, cut(256, 1, 1, 512, 32, 4, 0)),
        parts(4, 8, cut(256, 1, 1, 128, 32, 2, 0)),
        parts(1, 2, cut(128, 1, 2, 256, 48, 4, 3)),
    ];
    let fits = |t: &local::Tuning| t.shared_bytes() <= report.max_shared_memory as usize;
    cuts.into_iter().filter(fits).collect()
}

#[test]
#[ignore = "needs VOXCPM2_DIR to name the VoxCPM2 checkpoint"]
fn every_kernel_takes_no_more_workgroup_memory_than_the_device_has() {
    let ctx = Context::new().unwrap_or_else(|e| panic!("{e}"));
    let report = ctx.report().clone();
    let max = u64::from(report.max_shared_memory);
    let json = fs::read_to_string(checkpoint_dir().join("config.json")).unwrap();
    let side_config: lm_side::Config = serde_json::from_str(&json).unwrap();
    let config: PatchConfig = serde_json::from_str(&json).unwrap();
    let AudioVae { audio_vae_config } = serde_json::from_str(&json).unwrap();
    let side = LmSideShape::new(&side_config, 2048).unwrap();
    let mut modules = Modules::default();

    // Each kernel at the defaults its source declares.
    let mut defaults: BTreeMap<String, u64> = BTreeMap::new();
    for var in [KERNELS_VAR, MLIR_KERNELS_VAR] {
        for entry in fs::read_dir(kernels_dir(var)).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "spv") {
                let name = path.file_stem().unwrap().to_string_lossy().into_owned();
                let bytes = modules.bytes(&name, &[]);
                defaults.insert(name, bytes);
            }
        }
    }

    // Each kernel in every graph the service records, at the cuts the devices take by default
    // and those the tests and timings run.
    let mut graphs = lm_side_cuts(&side);
    let chunk = voxcpm::Params::default().decode_chunk_frames.get();
    graphs.push(
        vae_graph(&VaeShape::new(&audio_vae_config, chunk).unwrap())
            .unwrap()
            .graph,
    );
    for subgroup in SUBGROUPS {
        let lm = Tuning::for_batch(subgroup, 1);
        for local in local_cuts(subgroup, &report) {
            let patch = patch_graph(&side, &config, 10, &lm, &local).unwrap();
            let prompt = prompt_graphs(&patch, MAX_ROWS, &lm).unwrap();
            graphs.extend(prompt.programs);
            graphs.push(prompt.embed);
            graphs.push(patch.graph);
            for rows in [5, 16] {
                graphs.push(encoder_graph(&config, rows, &local).unwrap().graph);
                graphs.push(sampler_graph(&config, 10, rows, &local).unwrap().graph);
            }
        }
    }
    let mut largest: BTreeMap<String, (u64, Vec<(u32, SpecValue)>)> = BTreeMap::new();
    let mut past = Vec::new();
    let mut seen = HashSet::new();
    for graph in &graphs {
        for op in &graph.ops {
            let p = &op.pipeline;
            if !seen.insert(p.key()) {
                continue;
            }
            let bytes = modules.bytes(p.kernel.name, &p.constants);
            if bytes > max {
                past.push(format!("{} at {:?}: {bytes}", op.label, p.constants));
            }
            let at = largest.entry(p.kernel.name.to_string()).or_default();
            if bytes >= at.0 {
                *at = (bytes, p.constants.clone());
            }
        }
    }
    eprintln!(
        "{} distinct pipelines; workgroup memory per kernel, of the {max} bytes {} has: at its \
         defaults, and the most at any cut",
        seen.len(),
        report.name
    );
    for (name, bytes) in &defaults {
        let most = largest.get(name);
        eprintln!(
            "  {name:<40} {bytes:>6} {:>6} {}",
            most.map_or(String::from("-"), |m| m.0.to_string()),
            most.map_or(String::new(), |m| format!("{:?}", m.1))
        );
    }
    eprintln!(
        "cuts past the device's, which the runtime refuses ({}):\n  {}",
        past.len(),
        past.join("\n  ")
    );
    for name in largest.keys() {
        assert!(
            defaults.contains_key(name),
            "{name} is in no kernel directory"
        );
    }
    // The cuts the devices take by default fit.
    for subgroup in SUBGROUPS {
        for rows in [1, 2, 4, 8, 16] {
            let batch = Batch { rows, slots: rows };
            let tuning = Tuning::for_batch(subgroup, rows);
            let g = lm_side_batch_graph(&side, &tuning, batch).unwrap();
            for op in &g.graph.ops {
                let p = &op.pipeline;
                let bytes = modules.bytes(p.kernel.name, &p.constants);
                assert!(bytes <= max, "{} at {rows} rows: {bytes}", op.label);
            }
        }
    }
}
