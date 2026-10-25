//! librosa's `resample` at its default `res_type="soxr_hq"`, through the libsoxr that python-soxr
//! drives, called as python-soxr calls it so that the samples come out the same.

use std::{
    ffi::{CStr, c_char, c_uint, c_ulong, c_void},
    ptr,
};

/// A sample type libsoxr resamples.
pub trait Sample: Copy + Default + sealed::Sealed {
    /// libsoxr's datatype for split channels of this type, which python-soxr gives it.
    #[doc(hidden)]
    const SPLIT: c_uint;
}

impl Sample for f32 {
    const SPLIT: c_uint = 4;
}

impl Sample for f64 {
    const SPLIT: c_uint = 5;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// `librosa.resample(x, orig_sr=from, target_sr=to)`: `x` resampled from `from` Hz to `to` Hz by
/// libsoxr at its "high quality" recipe, then cut or padded with zeros to
/// `ceil(len * to / from)` samples.
///
/// # Panics
///
/// If either rate is not positive and finite.
pub fn resample<T: Sample>(x: &[T], from: f64, to: f64) -> Vec<T> {
    assert!(
        from > 0.0 && from.is_finite() && to > 0.0 && to.is_finite(),
        "resampling from {from} Hz to {to} Hz"
    );
    if from == to {
        return x.to_vec();
    }
    let len = (x.len() as f64 * (to / from)).ceil() as usize;
    let mut y = soxr_hq(x, from, to);
    y.resize(len, T::default());
    y
}

/// What python-soxr's `resample` gives for a mono array: the input, then a flush, into an output
/// buffer of the size it allocates. It feeds the input in chunks, which libsoxr's output does not
/// depend on.
fn soxr_hq<T: Sample>(x: &[T], from: f64, to: f64) -> Vec<T> {
    let olen = (x.len() as f64 * to / from + 1.0) as usize;
    let mut y = vec![T::default(); olen];
    let mut done = 0;
    // SAFETY: the specs are what libsoxr's constructors return; every buffer pointer is to a
    // live slice of the length passed beside it, and the resampler is deleted before return.
    unsafe {
        let io = soxr_io_spec(T::SPLIT, T::SPLIT);
        let quality = soxr_quality_spec(SOXR_HQ, 0);
        let mut error = ptr::null();
        let soxr = soxr_create(from, to, 1, &mut error, &io, &quality, ptr::null());
        check(error);
        let process = |input: Option<&[T]>, out: &mut [T]| {
            let ibuf = input.map_or(ptr::null(), <[T]>::as_ptr).cast::<c_void>();
            let obuf = out.as_mut_ptr().cast::<c_void>();
            // End of input is a null pointer where the array of channel pointers would be.
            let ibufs = input.map_or(ptr::null(), |_| &raw const ibuf);
            let mut odone = 0;
            let ilen = input.map_or(0, <[T]>::len);
            let error = soxr_process(
                soxr,
                ibufs,
                ilen,
                ptr::null_mut(),
                &raw const obuf,
                out.len(),
                &mut odone,
            );
            check(error);
            odone
        };
        done += process(Some(x), &mut y);
        done += process(None, &mut y[done..]);
        soxr_delete(soxr);
    }
    y.truncate(done);
    y
}

/// Panics with libsoxr's error, which it gives only for arguments this module checks or
/// memory it could not have.
fn check(error: *const c_char) {
    if !error.is_null() {
        // SAFETY: libsoxr's errors are static C strings.
        let error = unsafe { CStr::from_ptr(error) };
        panic!("libsoxr: {}", error.to_string_lossy());
    }
}

const SOXR_HQ: c_ulong = 4;

#[repr(C)]
struct IoSpec {
    itype: c_uint,
    otype: c_uint,
    scale: f64,
    e: *mut c_void,
    flags: c_ulong,
}

#[repr(C)]
struct QualitySpec {
    precision: f64,
    phase_response: f64,
    passband_end: f64,
    stopband_begin: f64,
    e: *mut c_void,
    flags: c_ulong,
}

unsafe extern "C" {
    fn soxr_io_spec(itype: c_uint, otype: c_uint) -> IoSpec;
    fn soxr_quality_spec(recipe: c_ulong, flags: c_ulong) -> QualitySpec;
    fn soxr_create(
        input_rate: f64,
        output_rate: f64,
        num_channels: c_uint,
        error: *mut *const c_char,
        io_spec: *const IoSpec,
        quality_spec: *const QualitySpec,
        runtime_spec: *const c_void,
    ) -> *mut c_void;
    fn soxr_process(
        resampler: *mut c_void,
        input: *const *const c_void,
        ilen: usize,
        idone: *mut usize,
        output: *const *mut c_void,
        olen: usize,
        odone: *mut usize,
    ) -> *const c_char;
    fn soxr_delete(resampler: *mut c_void);
}
