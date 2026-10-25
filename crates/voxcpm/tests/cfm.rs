mod common;

use burn::tensor::{FloatDType, Tensor};
use common::tiny;
use testkit::{Fixture, assert_close, device, fixture};
use voxcpm::{Cfm, Sampling, schedule};

const CASES: [&str; 2] = ["service", "long"];

fn sampling(fixture: &Fixture, case: &str) -> Sampling {
    let values = fixture.tensor::<1>(&format!("input.{case}.sampling"), &device());
    let [timesteps, temperature, cfg_value] =
        values.into_data().iter::<f32>().collect::<Vec<_>>()[..]
    else {
        panic!("sampling is [timesteps, temperature, cfg_value]");
    };
    Sampling {
        timesteps: timesteps as usize,
        temperature,
        cfg_value,
    }
}

fn floats(tensor: Tensor<1>) -> Vec<f32> {
    tensor.into_data().iter::<f32>().collect()
}

/// torch's vectorized cos rounds some of the cosines an ulp away from the nearest, and that ulp of a
/// cosine under 1 is at most 2^-24 of the time.
#[test]
fn schedule_is_upstreams_but_for_an_ulp_of_a_cosine() {
    let (fixture, device) = (fixture!("voxcpm_cfm"), device());
    for case in CASES {
        let timesteps = sampling(&fixture, case).timesteps;
        let expected = floats(fixture.tensor(&format!("input.{case}.t_span"), &device));
        let t_span = schedule(timesteps);
        assert_eq!(t_span.len(), expected.len());
        for (i, (t, e)) in t_span.iter().zip(&expected).enumerate() {
            let within = (t - e).abs() <= 2f32.powi(-24);
            assert!(within, "{case}: t_span[{i}] is {t}, upstream's {e}");
        }
    }
}

#[test]
fn solves_as_upstream_on_its_schedule() {
    let (fixture, device) = (fixture!("voxcpm_cfm"), device());
    let cfm = Cfm::load(&tiny(), fixture.path(), FloatDType::F32, &device).unwrap();
    for case in CASES {
        let input = |name: &str| format!("input.{case}.{name}");
        let t_span = floats(fixture.tensor(&input("t_span"), &device));
        let patches = cfm.solve_euler(
            fixture.tensor(&input("x"), &device),
            &t_span,
            fixture.tensor(&input("mu"), &device),
            fixture.tensor(&input("cond"), &device),
            sampling(&fixture, case).cfg_value,
        );
        // f32 here and in upstream, summing in other orders: these patches, of magnitude under
        // 3, came within 4.8e-7 (2 ulps) of upstream's.
        let expected = fixture.tensor(&format!("output.{case}.patches"), &device);
        assert_close(patches, expected, 2e-6, 1e-6);
    }
}

#[test]
fn samples_as_upstream() {
    let (fixture, device) = (fixture!("voxcpm_cfm"), device());
    let cfm = Cfm::load(&tiny(), fixture.path(), FloatDType::F32, &device).unwrap();
    for case in CASES {
        let input = |name: &str| format!("input.{case}.{name}");
        let patches = cfm.sample(
            fixture.tensor(&input("mu"), &device),
            fixture.tensor(&input("cond"), &device),
            fixture.tensor(&input("noise"), &device),
            &sampling(&fixture, case),
        );
        // As close as on upstream's own schedule: at this size the schedule's differences move
        // nothing past the rounding of the rest.
        let expected = fixture.tensor(&format!("output.{case}.patches"), &device);
        assert_close(patches, expected, 2e-6, 1e-6);
    }
}
