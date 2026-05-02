//! Pipeline 2 — given the top-N ranked pages per character, load full-res
//! and extract padded crops around the detected hair clusters. Also drops a
//! full-res copy of the page for posterity (the user asked for the highest
//! quality images of Misato).
//!
//! Strategy:
//!   1. Re-classify primary hair pixels at full resolution.
//!   2. Connected-component blobs, filtered by min size (% of full image).
//!   3. Group nearby blobs into "characters" by hierarchical merging:
//!      blobs whose bounding boxes (expanded by 2x) intersect get merged.
//!   4. For each merged group, expand the bbox 2.5x in each dim around the
//!      centroid (capped to image bounds), then crop and save.

use crate::classify::build_mask;
use crate::config::Character;
use crate::connected::{Blob, label};
use crate::rank::PageScore;
use anyhow::Context;
use image::{DynamicImage, GenericImageView, ImageReader};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// Smallest blob (frac of full-res pixels) we keep when extracting. Smaller
/// than rank's threshold because we want to capture small heads in dense
/// composition pages too.
const MIN_BLOB_FRAC_FULL: f32 = 0.0004;

/// Multiplicative expansion of a hair bbox to capture face + body context.
/// Tuned up from 2.5 after seeing the Asuka p081 crop that captured only
/// chin pixels.
const BBOX_EXPAND: f32 = 3.5;

/// Two blobs are merged into one "character" if their bboxes (each expanded
/// 3x around their centroid) intersect. Tuned up from 2.0 so split hair
/// regions on the same head (e.g. Asuka's twin tails + crown) get joined.
const MERGE_EXPAND: f32 = 3.0;

/// Floor on the smaller crop dimension as a fraction of the smaller image
/// dimension, so even a small detected blob produces a usable crop.
const MIN_CROP_FRAC: f32 = 0.30;

#[derive(Clone, Debug)]
pub struct ExtractedCrop {
    pub source_path: PathBuf,
    pub character: String,
    pub crop_index: usize,
    pub bbox: (u32, u32, u32, u32), // x, y, w, h in source coords
    pub area: u32,
}

pub struct ExtractConfig {
    pub top_n: usize,
    pub also_save_full_page: bool,
}

impl Default for ExtractConfig {
    fn default() -> Self {
        Self {
            top_n: 30,
            also_save_full_page: true,
        }
    }
}

pub fn extract_for_character(
    character: &Character,
    ranked: &[PageScore],
    out_dir: &Path,
    config: &ExtractConfig,
    progress: Option<&indicatif::ProgressBar>,
) -> anyhow::Result<Vec<ExtractedCrop>> {
    let crops_dir = out_dir.join(character.key).join("extracted");
    let pages_dir = out_dir.join(character.key).join("top_pages");
    std::fs::create_dir_all(&crops_dir)?;
    if config.also_save_full_page {
        std::fs::create_dir_all(&pages_dir)?;
    }

    let pick: Vec<&PageScore> = ranked
        .iter()
        .filter(|s| s.certainty > 0.0)
        .take(config.top_n)
        .collect();

    let results: Vec<Vec<ExtractedCrop>> = pick
        .par_iter()
        .enumerate()
        .map(|(rank, score)| {
            let r = process_one(
                rank,
                score,
                character,
                &crops_dir,
                &pages_dir,
                config.also_save_full_page,
            );
            if let Some(pb) = progress {
                pb.inc(1);
            }
            r.unwrap_or_else(|e| {
                eprintln!("warn: extract failed for {}: {}", score.path, e);
                Vec::new()
            })
        })
        .collect();

    Ok(results.into_iter().flatten().collect())
}

fn process_one(
    rank: usize,
    score: &PageScore,
    character: &Character,
    crops_dir: &Path,
    pages_dir: &Path,
    save_full_page: bool,
) -> anyhow::Result<Vec<ExtractedCrop>> {
    let path = PathBuf::from(&score.path);
    let img = ImageReader::open(&path)
        .with_context(|| format!("open {}", path.display()))?
        .with_guessed_format()?
        .decode()?;
    let (w, h) = img.dimensions();
    let total_pixels = (w as u64) * (h as u64);
    let min_blob = ((total_pixels as f32) * MIN_BLOB_FRAC_FULL).max(200.0) as u32;

    // Re-detect hair at full resolution.
    let rgb = img.to_rgb8();
    let primary = character.primary();
    let mask = build_mask(rgb.as_raw(), w, h, primary);
    let blobs = label(&mask, w, h, min_blob);

    // Stem for output filenames.
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("page")
        .to_string();
    let book_short = sanitize_book(&score.book);

    // Save full-res page copy (top-of-rank symbol of "highest quality").
    if save_full_page {
        let dest = pages_dir.join(format!(
            "rank{:03}_score{:.3}_{}_{}.jpg",
            rank + 1,
            score.certainty,
            book_short,
            stem
        ));
        img.save_with_format(&dest, image::ImageFormat::Jpeg)
            .with_context(|| format!("save full page {}", dest.display()))?;
    }

    // Merge nearby blobs into character-sized groups.
    let groups = merge_blobs(&blobs);

    let mut out = Vec::new();
    for (gi, group) in groups.iter().enumerate() {
        let bbox = group_bbox(group);
        let expanded = expand_bbox(bbox, w, h, BBOX_EXPAND);
        let area = expanded.2 * expanded.3;
        // Skip tiny crops that ended up under our floor anyway.
        if (expanded.2 as u64) * (expanded.3 as u64) < (total_pixels / 200) as u64 {
            continue;
        }
        let crop = crop_image(&img, expanded);
        let dest = crops_dir.join(format!(
            "rank{:03}_score{:.3}_{}_{}_crop{:02}.jpg",
            rank + 1,
            score.certainty,
            book_short,
            stem,
            gi
        ));
        crop.save_with_format(&dest, image::ImageFormat::Jpeg)
            .with_context(|| format!("save crop {}", dest.display()))?;
        out.push(ExtractedCrop {
            source_path: path.clone(),
            character: character.key.to_string(),
            crop_index: gi,
            bbox: expanded,
            area,
        });
    }

    Ok(out)
}

fn sanitize_book(book: &str) -> String {
    book.chars()
        .map(|c| match c {
            ' ' | '/' | '\\' | '[' | ']' | '.' => '_',
            c => c,
        })
        .collect::<String>()
        .chars()
        .take(28)
        .collect()
}

/// Hierarchical merge of blobs whose expanded bboxes overlap. Each merge
/// pass is O(n^2) in blobs; n is small (we already filtered by min area).
fn merge_blobs(blobs: &[Blob]) -> Vec<Vec<&Blob>> {
    let n = blobs.len();
    if n == 0 {
        return Vec::new();
    }
    // parent[i] = i initially.
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }

    let expanded: Vec<(f32, f32, f32, f32)> = blobs
        .iter()
        .map(|b| expand_blob_bbox(b, MERGE_EXPAND))
        .collect();

    for i in 0..n {
        for j in (i + 1)..n {
            if rect_overlap(expanded[i], expanded[j]) {
                union(&mut parent, i, j);
            }
        }
    }
    let mut groups: std::collections::BTreeMap<usize, Vec<&Blob>> =
        std::collections::BTreeMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        groups.entry(r).or_default().push(&blobs[i]);
    }
    groups.into_values().collect()
}

fn rect_overlap(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> bool {
    let (ax0, ay0, ax1, ay1) = a;
    let (bx0, by0, bx1, by1) = b;
    !(ax1 < bx0 || bx1 < ax0 || ay1 < by0 || by1 < ay0)
}

fn expand_blob_bbox(b: &Blob, factor: f32) -> (f32, f32, f32, f32) {
    let cx = b.cx;
    let cy = b.cy;
    let half_w = b.width() as f32 * factor * 0.5;
    let half_h = b.height() as f32 * factor * 0.5;
    (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
}

fn group_bbox(group: &[&Blob]) -> (u32, u32, u32, u32) {
    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    for b in group {
        if b.min_x < min_x {
            min_x = b.min_x;
        }
        if b.min_y < min_y {
            min_y = b.min_y;
        }
        if b.max_x > max_x {
            max_x = b.max_x;
        }
        if b.max_y > max_y {
            max_y = b.max_y;
        }
    }
    (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
}

fn expand_bbox(
    bbox: (u32, u32, u32, u32),
    img_w: u32,
    img_h: u32,
    factor: f32,
) -> (u32, u32, u32, u32) {
    let (x, y, w, h) = bbox;
    let cx = x as f32 + w as f32 / 2.0;
    let cy = y as f32 + h as f32 / 2.0;
    // Multiplicative expansion, then enforce a minimum-size floor relative
    // to the page so a tight chin-blob still yields a usable portrait crop.
    let min_dim = (img_w.min(img_h) as f32) * MIN_CROP_FRAC;
    let new_w = ((w as f32 * factor).max(min_dim)).min(img_w as f32);
    let new_h = ((h as f32 * factor).max(min_dim)).min(img_h as f32);
    let nx = (cx - new_w / 2.0).max(0.0);
    let ny = (cy - new_h / 2.0).max(0.0);
    let nx = nx.min((img_w as f32) - new_w);
    let ny = ny.min((img_h as f32) - new_h);
    (nx as u32, ny as u32, new_w as u32, new_h as u32)
}

fn crop_image(img: &DynamicImage, bbox: (u32, u32, u32, u32)) -> DynamicImage {
    img.crop_imm(bbox.0, bbox.1, bbox.2, bbox.3)
}
