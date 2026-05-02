//! Per-pixel classification: does a pixel match a `ColorTarget`?
//!
//! Match criteria (all must hold):
//!   - Pixel HSV hue is in the target's hue band (with wraparound).
//!   - Saturation in [sat_min, 1] and value in [val_min, val_max].
//!   - LAB ΔE76 vs target ≤ de_max.
//!
//! Both HSV and LAB are used because each one alone has a failure mode:
//! HSV is loose for low-sat colors (Rei hair, anything near grey), and LAB
//! alone admits perceptually close but wrong-hue grays. Combined, they
//! produce a clean hair-class cluster.

use crate::color::{Lab, Rgb, delta_e76, hue_in_band};
use crate::config::ColorTarget;

#[inline]
pub fn pixel_matches(rgb: Rgb, target: &ColorTarget, target_lab: Lab) -> bool {
    let hsv = rgb.to_hsv();
    if !hue_in_band(hsv.h, target.hue_min, target.hue_max) {
        return false;
    }
    if hsv.s < target.sat_min || hsv.v < target.val_min || hsv.v > target.val_max {
        return false;
    }
    let lab = rgb.to_lab();
    delta_e76(lab, target_lab) <= target.de_max
}

/// Build a binary mask (1 byte per pixel; 1 = match, 0 = no match) for a
/// single target across an image's RGB pixel array (row-major, RGBRGB...).
pub fn build_mask(rgb_pixels: &[u8], width: u32, height: u32, target: &ColorTarget) -> Vec<u8> {
    let target_lab = target.rgb.to_lab();
    let n = (width as usize) * (height as usize);
    let mut mask = vec![0u8; n];
    for i in 0..n {
        let r = rgb_pixels[3 * i];
        let g = rgb_pixels[3 * i + 1];
        let b = rgb_pixels[3 * i + 2];
        if pixel_matches(Rgb(r, g, b), target, target_lab) {
            mask[i] = 1;
        }
    }
    mask
}

/// Count of matching pixels in a mask.
#[inline]
pub fn mask_count(mask: &[u8]) -> u32 {
    mask.iter().map(|&b| b as u32).sum()
}
