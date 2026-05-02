//! Color math: RGB <-> HSV, RGB -> CIELAB (D65), ΔE76.
//!
//! Hand-rolled, no allocations on the hot path. All operations on f32.

#[derive(Clone, Copy, Debug)]
pub struct Rgb(pub u8, pub u8, pub u8);

#[derive(Clone, Copy, Debug)]
pub struct Hsv {
    /// degrees in [0, 360)
    pub h: f32,
    /// [0, 1]
    pub s: f32,
    /// [0, 1]
    pub v: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Lab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

impl Rgb {
    pub fn from_hex(hex: &str) -> Option<Self> {
        let s = hex.trim_start_matches('#');
        if s.len() != 6 {
            return None;
        }
        let r = u8::from_str_radix(&s[0..2], 16).ok()?;
        let g = u8::from_str_radix(&s[2..4], 16).ok()?;
        let b = u8::from_str_radix(&s[4..6], 16).ok()?;
        Some(Rgb(r, g, b))
    }

    pub fn to_hsv(self) -> Hsv {
        let r = self.0 as f32 / 255.0;
        let g = self.1 as f32 / 255.0;
        let b = self.2 as f32 / 255.0;
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let d = max - min;
        let v = max;
        let s = if max <= 1e-6 { 0.0 } else { d / max };
        let h = if d <= 1e-6 {
            0.0
        } else if (max - r).abs() < 1e-6 {
            60.0 * (((g - b) / d).rem_euclid(6.0))
        } else if (max - g).abs() < 1e-6 {
            60.0 * ((b - r) / d + 2.0)
        } else {
            60.0 * ((r - g) / d + 4.0)
        };
        let h = if h < 0.0 { h + 360.0 } else { h };
        Hsv { h, s, v }
    }

    pub fn to_lab(self) -> Lab {
        // sRGB -> linear
        let r = srgb_to_linear(self.0 as f32 / 255.0);
        let g = srgb_to_linear(self.1 as f32 / 255.0);
        let b = srgb_to_linear(self.2 as f32 / 255.0);
        // linear sRGB -> XYZ (D65)
        let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
        let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
        let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
        // XYZ -> LAB (D65 reference white)
        const XN: f32 = 0.95047;
        const YN: f32 = 1.00000;
        const ZN: f32 = 1.08883;
        let fx = lab_f(x / XN);
        let fy = lab_f(y / YN);
        let fz = lab_f(z / ZN);
        Lab {
            l: 116.0 * fy - 16.0,
            a: 500.0 * (fx - fy),
            b: 200.0 * (fy - fz),
        }
    }
}

#[inline]
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn lab_f(t: f32) -> f32 {
    // (6/29)^3 ≈ 0.008856
    if t > 0.008856 {
        t.cbrt()
    } else {
        // (1/3)*(29/6)^2 * t + 4/29
        7.787 * t + 16.0 / 116.0
    }
}

/// Euclidean ΔE76 between two LAB colors.
#[inline]
pub fn delta_e76(a: Lab, b: Lab) -> f32 {
    let dl = a.l - b.l;
    let da = a.a - b.a;
    let db = a.b - b.b;
    (dl * dl + da * da + db * db).sqrt()
}

/// Hue distance accounting for 360° wraparound, in degrees.
#[inline]
pub fn hue_distance(h1: f32, h2: f32) -> f32 {
    let d = (h1 - h2).abs() % 360.0;
    if d > 180.0 { 360.0 - d } else { d }
}

/// Test if a hue h (deg) lies inside a [min, max] band, allowing wraparound
/// when min > max (e.g., 340..20 means 340..360 ∪ 0..20).
#[inline]
pub fn hue_in_band(h: f32, min: f32, max: f32) -> bool {
    if min <= max {
        h >= min && h <= max
    } else {
        h >= min || h <= max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn hex_roundtrip() {
        let c = Rgb::from_hex("#5B4B7A").unwrap();
        assert_eq!((c.0, c.1, c.2), (0x5B, 0x4B, 0x7A));
    }

    #[test]
    fn misato_hair_hsv() {
        // #5B4B7A -> roughly H=258°, S=0.39, V=0.48
        let hsv = Rgb(0x5B, 0x4B, 0x7A).to_hsv();
        assert!(approx(hsv.h, 258.0, 4.0), "h was {}", hsv.h);
        assert!(approx(hsv.s, 0.39, 0.05), "s was {}", hsv.s);
        assert!(approx(hsv.v, 0.48, 0.05), "v was {}", hsv.v);
    }

    #[test]
    fn red_jacket_hue_wrap() {
        // #B91C2C -> H ~ 355°
        let hsv = Rgb(0xB9, 0x1C, 0x2C).to_hsv();
        assert!(approx(hsv.h, 355.0, 4.0), "h was {}", hsv.h);
    }

    #[test]
    fn black_lab() {
        let lab = Rgb(0, 0, 0).to_lab();
        assert!(approx(lab.l, 0.0, 0.5));
    }

    #[test]
    fn white_lab() {
        let lab = Rgb(255, 255, 255).to_lab();
        assert!(approx(lab.l, 100.0, 0.5));
    }

    #[test]
    fn hue_dist_wrap() {
        assert!(approx(hue_distance(355.0, 5.0), 10.0, 0.001));
        assert!(approx(hue_distance(10.0, 350.0), 20.0, 0.001));
    }

    #[test]
    fn hue_band_wrap() {
        assert!(hue_in_band(355.0, 340.0, 20.0));
        assert!(hue_in_band(10.0, 340.0, 20.0));
        assert!(!hue_in_band(100.0, 340.0, 20.0));
    }
}
