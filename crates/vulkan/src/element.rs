//! What a buffer holds, element by element: [`F16`], `f32` and `u32`, each an [`Element`], and
//! [`BF16`], which a checkpoint stores weights in and no kernel reads.
//!
//! A 16-bit float is its bits, with no arithmetic: it is made from an `f32` and read back as one,
//! by the functions here, named for how they round. They round as half computes, whose
//! conversions from and to `f32` agree between its hardware and software paths on every input,
//! so a value's bits do not depend on the build. Its conversion from `f64` does, rounding through
//! `f32` in some builds and once in others, and nothing here calls it.
//!
//! A rounding is never implicit:
//!
//! ```compile_fail,E0277
//! use vulkan::F16;
//! let x: F16 = 0.1f32.into();
//! ```

use std::fmt;

/// An element type as a value, as GLSL and the kernels' manifests name it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ElementType {
    F16,
    F32,
    U32,
}

impl ElementType {
    pub fn name(self) -> &'static str {
        match self {
            ElementType::F16 => "f16",
            ElementType::F32 => "f32",
            ElementType::U32 => "u32",
        }
    }

    /// The bytes of one element.
    pub fn bytes(self) -> u64 {
        match self {
            ElementType::F16 => 2,
            ElementType::F32 | ElementType::U32 => 4,
        }
    }
}

impl fmt::Display for ElementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A type a buffer holds: one whose bytes in memory, little-endian, are its value as the GPU
/// reads it, so that a buffer is written and read by copying them.
///
/// # Safety
///
/// Implement only for types of [`TYPE`](Self::TYPE)'s size that have no padding and for which
/// every bit pattern of their size is a valid value.
pub unsafe trait Element: Copy + Send + Sync + 'static {
    /// The type, as a kernel declares a binding to hold it.
    const TYPE: ElementType;
}

// SAFETY: four bytes, no padding, and any bits are an f32.
unsafe impl Element for f32 {
    const TYPE: ElementType = ElementType::F32;
}

// SAFETY: four bytes, no padding, and any bits are a u32.
unsafe impl Element for u32 {
    const TYPE: ElementType = ElementType::U32;
}

/// An IEEE 754 binary16 float, as its bits: what the kernels hold weights, activations and caches
/// in. Two are equal when their bits are, so a NaN equals itself and 0 does not equal -0.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct F16(u16);

// SAFETY: a transparent u16, any bits of which are an f16.
unsafe impl Element for F16 {
    const TYPE: ElementType = ElementType::F16;
}

impl F16 {
    pub const ZERO: F16 = F16(0);

    pub const fn from_bits(bits: u16) -> Self {
        F16(bits)
    }

    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// The float whose bits `bytes` are, little-endian, as a checkpoint stores them.
    pub const fn from_le_bytes(bytes: [u8; 2]) -> Self {
        F16(u16::from_le_bytes(bytes))
    }

    /// `value` rounded to the nearest f16, ties to even: past the largest finite f16, 65504, to
    /// infinity, and below the smallest subnormal to zero, keeping the sign; a NaN to a quiet NaN.
    pub fn from_f32(value: f32) -> Self {
        F16(half::f16::from_f32(value).to_bits())
    }

    /// `value` rounded to f32, then to f16 as [`from_f32`](Self::from_f32) rounds that: twice,
    /// which leaves a few values an ulp from where rounding once would, in every build alike.
    pub fn from_f64(value: f64) -> Self {
        Self::from_f32(value as f32)
    }

    /// The bfloat16 `value` rounded to f16 as [`from_f32`](Self::from_f32) rounds the f32 it is,
    /// exactly, a NaN quiet.
    pub fn from_bf16(value: BF16) -> Self {
        Self::from_f32(value.to_f32())
    }

    /// The value, exactly.
    pub fn to_f32(self) -> f32 {
        half::f16::from_bits(self.0).to_f32()
    }
}

impl fmt::Debug for F16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.to_f32(), f)
    }
}

/// A bfloat16, the upper half of an f32's bits: what VoxCPM2's checkpoint stores its weights in,
/// which the kernels take as [`F16::from_bf16`] rounds them.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct BF16(u16);

impl BF16 {
    pub const fn from_bits(bits: u16) -> Self {
        BF16(bits)
    }

    /// The float whose bits `bytes` are, little-endian, as a checkpoint stores them.
    pub const fn from_le_bytes(bytes: [u8; 2]) -> Self {
        BF16(u16::from_le_bytes(bytes))
    }

    /// The value, exactly, a NaN quiet.
    pub fn to_f32(self) -> f32 {
        half::bf16::from_bits(self.0).to_f32()
    }
}

impl fmt::Debug for BF16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.to_f32(), f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_f16_reads_as_the_f32_it_is_and_rounds_back_to_itself() {
        for bits in 0..=u16::MAX {
            let h = F16::from_bits(bits);
            let v = h.to_f32();
            if v.is_nan() {
                assert!(F16::from_f32(v).to_f32().is_nan(), "{bits:#06x}");
                continue;
            }
            assert_eq!(F16::from_f32(v), h, "{bits:#06x}");
            assert_eq!(F16::from_f64(f64::from(v)), h, "{bits:#06x}");
        }
    }

    #[test]
    fn an_f32_rounds_to_the_nearest_f16_and_a_tie_to_the_even_one() {
        let ulp = 2f32.powi(-10);
        let cases = [
            (1.0 + ulp / 2.0, 0x3c00),
            (1.0 + 1.5 * ulp, 0x3c02),
            (1.0 + ulp / 2.0 + ulp / 64.0, 0x3c01),
            (-1.0 - ulp / 2.0, 0xbc00),
            (65504.0, 0x7bff),
            (65519.0, 0x7bff),
            (65520.0, 0x7c00),
            (f32::MAX, 0x7c00),
            (f32::NEG_INFINITY, 0xfc00),
            (2f32.powi(-24), 0x0001),
            (2f32.powi(-25), 0x0000),
            (1.5 * 2f32.powi(-24), 0x0002),
            (-2f32.powi(-26), 0x8000),
        ];
        for (value, bits) in cases {
            assert_eq!(F16::from_f32(value).to_bits(), bits, "{value:e}");
        }
        assert!(F16::from_f32(f32::NAN).to_f32().is_nan());
    }

    /// The four of the real model's DiT time constants that an f64 rounded once to f16 puts an
    /// ulp below where rounding through f32 does.
    #[test]
    fn an_f64_rounds_through_f32() {
        for (value, bits) in [
            (0x4036_fa00_e1ad_d5dd, 0x4dbf),
            (0xc011_fa00_5296_9974, 0xc47f),
            (0xc004_ba00_794e_d8ca, 0xc12f),
            (0x402c_da00_d74d_552a, 0x4b37),
        ] {
            let value = f64::from_bits(value);
            assert_eq!(F16::from_f64(value).to_bits(), bits, "{value:e}");
        }
    }

    #[test]
    fn a_bfloat16_is_an_f32_cut_short_and_rounds_to_f16_once() {
        for bits in 0..=u16::MAX {
            let v = BF16::from_bits(bits).to_f32();
            assert!(
                v.is_nan() || v.to_bits() == u32::from(bits) << 16,
                "{bits:#06x}"
            );
        }
        let f16 = |v: f32| F16::from_bf16(BF16::from_bits((v.to_bits() >> 16) as u16)).to_bits();
        assert_eq!(f16(1.5), 0x3e00);
        assert_eq!(f16(1e5), 0x7c00);
        assert_eq!(f16(1.25 * 2f32.powi(-24)), 0x0001);
        assert_eq!(f16(1.5 * 2f32.powi(-24)), 0x0002);
        assert_eq!(f16(2f32.powi(-30)), 0x0000);
    }

    #[test]
    fn an_element_type_is_as_large_as_its_type() {
        assert_eq!(F16::TYPE.bytes(), size_of::<F16>() as u64);
        assert_eq!(f32::TYPE.bytes(), size_of::<f32>() as u64);
        assert_eq!(u32::TYPE.bytes(), size_of::<u32>() as u64);
    }
}
