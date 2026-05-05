//! Pipeline 1 — rank artbook pages by certainty that a given character is on
//! them. Operates on downscaled pages (cap longest side at `MAX_DIM`) for
//! speed; the full-res image is only revisited in `extract.rs`.
//!
//! Scoring (per character, per page):
//!   - hair_pixel_ratio:    fraction of pixels matching primary hair color
//!   - hair_blob_ratio:     largest hair blob's area / image area
//!   - secondary_ratio:     fraction of pixels matching any secondary color
//!   - coupling_pixels:     secondary pixels near a primary blob's centroid
//!
//! Final certainty is a weighted log-scale combination, designed so that a
//! single decisive cue (a clear blob of hair near a confirming color) easily
//! beats many scattered pixels with no spatial structure.

use crate::classify::{build_mask, mask_count};
use crate::config::Character;
use crate::connected::{Blob, coupling_score, label};
use crate::io::Page;
use anyhow::Context;
use image::imageops::FilterType;
use image::{GenericImageView, ImageReader};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

pub const MAX_DIM: u32 = 1500;
/// Smallest blob (in % of total pixels) that we'll treat as a real cluster.
pub const MIN_BLOB_FRAC: f32 = 0.0008; // ~0.08% — about a head-sized region

/// Tunable weights inside `combine_score`. Lifted out of literals so the
/// label-driven tuner can search over them per character.
///
/// All defaults match the hand-tuned values that have been in production —
/// see comments in `combine_score` for what each one does.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ScoringWeights {
    pub blob_weight: f32,
    pub mass_weight: f32,
    pub mult_weight: f32,
    pub secondary_weight: f32,
    pub coupling_weight: f32,
    pub density_floor: f32,
    pub density_slope: f32,
    pub aspect_log_div: f32,
    pub coupling_cap: f32,
    pub competition_floor: f32,
    pub competition_slope: f32,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            blob_weight: 2.0,
            mass_weight: 0.6,
            mult_weight: 0.25,
            secondary_weight: 0.3,
            coupling_weight: 0.6,
            density_floor: 0.4,
            density_slope: 1.1,
            aspect_log_div: 1.4,
            coupling_cap: 0.15,
            competition_floor: 0.25,
            competition_slope: 0.75,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PageScore {
    pub book: String,
    pub filename: String,
    pub path: String,
    pub character: String,
    pub width: u32,
    pub height: u32,
    pub orig_width: u32,
    pub orig_height: u32,
    pub hair_pixels: u32,
    pub hair_pixel_ratio: f32,
    pub hair_blobs: u32,
    pub largest_blob_area: u32,
    pub largest_blob_ratio: f32,
    pub second_blob_area: u32,
    pub largest_blob_density: f32,
    pub largest_blob_aspect: f32,
    pub secondary_pixels: u32,
    pub secondary_ratio: f32,
    pub coupling_pixels: u32,
    pub certainty: f32,
}

#[derive(Clone, Debug)]
pub struct RankConfig {
    pub min_certainty: f32,
}

impl Default for RankConfig {
    fn default() -> Self {
        Self {
            min_certainty: 0.0,
        }
    }
}

/// Rank a list of pages for a list of characters. Returns one PageScore per
/// (character, page). Internally parallelized — each page is loaded and
/// scored once, then evaluated for every character (all share the downscale).
///
/// `weights_per_char` lets tuning override the default weights per character.
/// Pass `&BTreeMap::new()` for defaults.
pub fn rank_pages(
    pages: &[Page],
    characters: &[Character],
    weights_per_char: &std::collections::BTreeMap<String, ScoringWeights>,
    progress: Option<&indicatif::ProgressBar>,
) -> Vec<PageScore> {
    let counter = AtomicUsize::new(0);
    let default_w = ScoringWeights::default();

    let scores: Vec<Vec<PageScore>> = pages
        .par_iter()
        .map(|page| {
            let res = score_page(page, characters, weights_per_char, &default_w);
            let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
            if let Some(pb) = progress {
                pb.set_position(n as u64);
            }
            res.unwrap_or_else(|e| {
                eprintln!("warn: failed to score {}: {}", page.path.display(), e);
                Vec::new()
            })
        })
        .collect();

    scores.into_iter().flatten().collect()
}

/// Load one page, downscale, then score it once per character.
///
/// Inter-character competition: a single hair-color match alone is ambiguous
/// (e.g., violet shadows on Asuka's red plugsuit register as Misato hair).
/// We resolve by computing every character's primary signal up front, then
/// during scoring we apply a multiplicative competition factor — if some
/// other character has a much stronger hair blob on this page, demote.
fn score_page(
    page: &Page,
    characters: &[Character],
    weights_per_char: &std::collections::BTreeMap<String, ScoringWeights>,
    default_w: &ScoringWeights,
) -> anyhow::Result<Vec<PageScore>> {
    let img = ImageReader::open(&page.path)
        .with_context(|| format!("open {}", page.path.display()))?
        .with_guessed_format()?
        .decode()
        .with_context(|| format!("decode {}", page.path.display()))?;
    let (orig_w, orig_h) = img.dimensions();
    // Downscale longest side to MAX_DIM.
    let scale = (MAX_DIM as f32) / (orig_w.max(orig_h) as f32);
    let (w, h) = if scale < 1.0 {
        (
            (orig_w as f32 * scale) as u32,
            (orig_h as f32 * scale) as u32,
        )
    } else {
        (orig_w, orig_h)
    };
    let small = if scale < 1.0 {
        img.resize_exact(w, h, FilterType::Lanczos3)
    } else {
        img
    };
    let rgb = small.to_rgb8();
    let pixels = rgb.as_raw().as_slice();

    let total_pixels = (w as u32) * (h as u32);
    let min_blob = ((total_pixels as f32) * MIN_BLOB_FRAC).max(80.0) as u32;

    // Pass 1: per-character primary mask + blobs + combined secondary mask.
    // Both primary and secondary mass enter the inter-character competition,
    // because some characters (Asuka in plugsuit) have a much bigger
    // secondary signature (red plugsuit) than primary (small head of hair).
    struct Pass1 {
        hair_pixels: u32,
        blobs: Vec<Blob>,
        secondary_mask: Vec<u8>,
        secondary_pixels: u32,
    }
    let pass1: Vec<Pass1> = characters
        .iter()
        .map(|ch| {
            let primary = ch.primary();
            let primary_mask = build_mask(pixels, w, h, primary);
            let hair_pixels = mask_count(&primary_mask);
            let blobs = label(&primary_mask, w, h, min_blob);

            let mut secondary_mask = vec![0u8; (w as usize) * (h as usize)];
            for sec in ch.secondaries() {
                let sm = build_mask(pixels, w, h, sec);
                for (a, b) in secondary_mask.iter_mut().zip(sm.iter()) {
                    *a |= *b;
                }
            }
            let secondary_pixels = mask_count(&secondary_mask);

            Pass1 {
                hair_pixels,
                blobs,
                secondary_mask,
                secondary_pixels,
            }
        })
        .collect();

    // Per-page max metrics (for competition).
    let max_largest: u32 = pass1
        .iter()
        .map(|p| p.blobs.first().map(|b| b.area).unwrap_or(0))
        .max()
        .unwrap_or(0);
    // Combined signature mass per character: hair pixels + 0.7 × secondary
    // pixels. Secondary is weighted < 1.0 because hair is a more specific
    // identifier than jacket/plugsuit colors.
    let combined_mass: Vec<u32> = pass1
        .iter()
        .map(|p| p.hair_pixels + (p.secondary_pixels as f32 * 0.7) as u32)
        .collect();
    let max_combined: u32 = *combined_mass.iter().max().unwrap_or(&0);

    // Pass 2: scoring with competition factor.
    let mut out = Vec::with_capacity(characters.len());
    for ((ch, p1), &mass_combined) in
        characters.iter().zip(pass1.iter()).zip(combined_mass.iter())
    {
        let weights: &ScoringWeights = weights_per_char
            .get(ch.key)
            .unwrap_or(default_w);
        let largest = p1.blobs.first().map(|b| b.area).unwrap_or(0);
        let second = p1.blobs.get(1).map(|b| b.area).unwrap_or(0);
        let largest_density = p1.blobs.first().map(|b| b.density()).unwrap_or(0.0);
        let largest_aspect = p1.blobs.first().map(|b| b.aspect()).unwrap_or(0.0);

        let coupling = if p1.blobs.is_empty() {
            0
        } else {
            coupling_score(&p1.secondary_mask, w, &p1.blobs)
        };

        let hair_pixel_ratio = p1.hair_pixels as f32 / total_pixels as f32;
        let largest_blob_ratio = largest as f32 / total_pixels as f32;
        let secondary_ratio = p1.secondary_pixels as f32 / total_pixels as f32;

        // Inter-character competition: a page that has a far stronger
        // signature for another character (e.g. Asuka's plugsuit red filling
        // most of the image) is almost certainly that character's page, not
        // ours. We use combined mass (primary + 0.7 × secondary) so that
        // characters with a body-sized signature (plugsuits) compete on
        // equal footing with characters whose signature is mostly hair.
        let blob_share = if max_largest == 0 {
            1.0
        } else {
            (largest as f32) / (max_largest as f32)
        };
        let mass_share = if max_combined == 0 {
            1.0
        } else {
            (mass_combined as f32) / (max_combined as f32)
        };
        // Both blob and mass must be competitive. We take the min and scale
        // into [0.25, 1.0] so a clear loser is heavily demoted.
        let competition =
            (weights.competition_floor + weights.competition_slope * (blob_share.min(mass_share)))
                .min(1.0);

        let certainty = combine_score(
            hair_pixel_ratio,
            largest_blob_ratio,
            p1.blobs.len() as u32,
            secondary_ratio,
            coupling,
            total_pixels,
            largest_density,
            largest_aspect,
            weights,
        ) * competition;

        out.push(PageScore {
            book: page.book.clone(),
            filename: page.filename.clone(),
            path: page.path.to_string_lossy().to_string(),
            character: ch.key.to_string(),
            width: w,
            height: h,
            orig_width: orig_w,
            orig_height: orig_h,
            hair_pixels: p1.hair_pixels,
            hair_pixel_ratio,
            hair_blobs: p1.blobs.len() as u32,
            largest_blob_area: largest,
            largest_blob_ratio,
            second_blob_area: second,
            largest_blob_density: largest_density,
            largest_blob_aspect: largest_aspect,
            secondary_pixels: p1.secondary_pixels,
            secondary_ratio,
            coupling_pixels: coupling,
            certainty,
        });
    }
    Ok(out)
}

/// Combine evidence into a single certainty score in roughly [0, ~16].
///
/// We weight the log of the largest blob's area heavily (a coherent
/// hair-shaped cluster is the strongest cue), then modulate by the blob's
/// shape — compactness (area / bbox_area) and aspect ratio. A real head/hair
/// region is roughly compact (>= 0.35 fill) and not extreme in aspect; a
/// washed-out background of stray violet pixels is sparse and elongated.
pub fn combine_score(
    hair_pixel_ratio: f32,
    largest_blob_ratio: f32,
    blob_count: u32,
    secondary_ratio: f32,
    coupling_pixels: u32,
    total_pixels: u32,
    largest_density: f32,
    largest_aspect: f32,
    w: &ScoringWeights,
) -> f32 {
    if largest_blob_ratio < 1e-6 && hair_pixel_ratio < 1e-5 {
        return 0.0;
    }
    let density_factor = (w.density_floor + largest_density * w.density_slope).min(1.0);
    let aspect = if largest_aspect == 0.0 {
        1.0
    } else {
        let log_a = largest_aspect.ln().abs();
        (1.0 - (log_a / w.aspect_log_div).min(0.5)).max(0.5)
    };
    let shape_factor = density_factor * aspect;

    let blob_term = (largest_blob_ratio * 1000.0).ln_1p() * w.blob_weight * shape_factor;
    let mass_term = (hair_pixel_ratio * 1000.0).ln_1p() * w.mass_weight;
    let mult_term = (blob_count as f32).min(5.0).sqrt() * w.mult_weight;
    let secondary_term = (secondary_ratio * 1000.0).ln_1p() * w.secondary_weight;
    let coupling_ratio = (coupling_pixels as f32 / total_pixels as f32).min(w.coupling_cap);
    let coupling_term = (coupling_ratio * 1000.0).ln_1p() * w.coupling_weight;

    blob_term + mass_term + mult_term + secondary_term + coupling_term
}

/// Re-evaluate a `PageScore`'s certainty under different scoring weights,
/// using only the cached scalar features (no image decode needed). This is
/// what makes label-driven tuning fast: we're searching over weights, not
/// re-running the pixel pipeline.
pub fn rescore_with(score: &PageScore, w: &ScoringWeights, comp: f32) -> f32 {
    combine_score(
        score.hair_pixel_ratio,
        score.largest_blob_ratio,
        score.hair_blobs,
        score.secondary_ratio,
        score.coupling_pixels,
        score.width * score.height,
        score.largest_blob_density,
        score.largest_blob_aspect,
        w,
    ) * comp
}

/// Sort scores descending by certainty, then by largest blob area.
pub fn sort_descending(scores: &mut [PageScore]) {
    scores.sort_by(|a, b| {
        b.certainty
            .partial_cmp(&a.certainty)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.largest_blob_area.cmp(&a.largest_blob_area))
    });
}

pub fn write_csv(path: &Path, scores: &[PageScore]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut wtr = csv::Writer::from_path(path)?;
    for s in scores {
        wtr.serialize(s)?;
    }
    wtr.flush()?;
    Ok(())
}

pub fn write_json(path: &Path, scores: &[PageScore]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let f = std::fs::File::create(path)?;
    serde_json::to_writer_pretty(f, scores)?;
    Ok(())
}

/// Group scores by character. Returns a vec of (char_key, scores_for_char)
/// with each list already sorted descending.
pub fn group_by_character(scores: Vec<PageScore>) -> Vec<(String, Vec<PageScore>)> {
    let mut by_char: std::collections::BTreeMap<String, Vec<PageScore>> =
        std::collections::BTreeMap::new();
    for s in scores {
        by_char.entry(s.character.clone()).or_default().push(s);
    }
    let mut out: Vec<(String, Vec<PageScore>)> = by_char.into_iter().collect();
    for (_, v) in out.iter_mut() {
        sort_descending(v);
    }
    out
}

/// Convenience to write per-character outputs into `out_dir/<char>/rankings.{csv,json}`.
pub fn write_per_character(
    out_dir: &Path,
    grouped: &[(String, Vec<PageScore>)],
) -> anyhow::Result<()> {
    for (key, scores) in grouped {
        let dir: PathBuf = out_dir.join(key);
        std::fs::create_dir_all(&dir)?;
        write_csv(&dir.join("rankings.csv"), scores)?;
        write_json(&dir.join("rankings.json"), scores)?;
    }
    Ok(())
}
