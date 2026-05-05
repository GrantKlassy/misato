//! Label-driven tuning.
//!
//! Optimize per-character `ScoringWeights` (and competition_floor/slope)
//! against human-supplied labels. Operates on the cached scalar features in
//! `output/<char>/rankings.json` — no image decode needed, so each candidate
//! evaluation is ~ms even across the full label set.
//!
//! Objective: ranking AUC per character. AUC is the probability that a
//! random Yes-labeled page outranks a random No-labeled page under the
//! candidate weights. Robust to the absolute scale of certainty (we only
//! care about ordering) and naturally handles class imbalance. `Unsure`
//! labels are excluded.
//!
//! Color-target tuning (the per-pixel `hue_min/max`, `sat_min`, etc.) is a
//! separate, more expensive problem: each candidate requires re-running
//! `build_mask` + connected components on the actual pixels. Out of scope
//! here — see TODO at bottom.

use crate::config::Character;
use crate::label::{LabelStore, Verdict, default_labels_path};
use crate::rank::{PageScore, ScoringWeights, combine_score};
use anyhow::{Context, Result};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Minimum positives and negatives per character before we'll tune. Below
/// this we'd be overfitting to a handful of labels.
pub const MIN_POS: usize = 10;
pub const MIN_NEG: usize = 10;

/// Number of random-search candidates per character.
pub const N_ITER: usize = 2000;

/// Multiplicative jitter range for each weight: candidate = default * U(LOW, HIGH).
const JITTER_LOW: f32 = 0.4;
const JITTER_HIGH: f32 = 2.5;

/// Per-page, per-character merged data needed to recompute certainty under
/// candidate weights. We keep the raw blob/mass cross-character data so we
/// can recompute the competition factor (which also depends on the candidate
/// floor/slope).
#[derive(Clone, Debug)]
struct MergedPage {
    /// character_key -> PageScore for this character on this page.
    scores: BTreeMap<String, PageScore>,
    /// Cross-character max-largest-blob (used in competition).
    max_largest: u32,
    /// Cross-character max-combined-mass (used in competition).
    max_combined_mass: u32,
    /// Per-character combined mass (hair + 0.7 × secondary), pre-computed.
    combined_mass: BTreeMap<String, u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TunedConfig {
    /// character_key -> tuned scoring weights.
    pub weights: BTreeMap<String, ScoringWeights>,
    /// Diagnostics: AUC achieved per character (default vs tuned).
    pub diagnostics: BTreeMap<String, TuneDiag>,
    /// When this file was written (informational).
    pub timestamp: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TuneDiag {
    pub n_pos: usize,
    pub n_neg: usize,
    pub auc_default: f32,
    pub auc_tuned: f32,
}

pub fn default_tuned_config_path(out_dir: &Path) -> PathBuf {
    out_dir.join("tuned_config.json")
}

/// Load all per-character rankings.json files and merge them by (book, filename).
fn load_merged(out_dir: &Path, characters: &[Character]) -> Result<Vec<MergedPage>> {
    let mut merged: BTreeMap<(String, String), MergedPage> = BTreeMap::new();
    for ch in characters {
        let p = out_dir.join(ch.key).join("rankings.json");
        if !p.exists() {
            anyhow::bail!("missing {}: run `misato rank` first", p.display());
        }
        let f = std::fs::File::open(&p)
            .with_context(|| format!("open {}", p.display()))?;
        let scores: Vec<PageScore> = serde_json::from_reader(f)
            .with_context(|| format!("parse {}", p.display()))?;
        for s in scores {
            let key = (s.book.clone(), s.filename.clone());
            let entry = merged.entry(key).or_insert_with(|| MergedPage {
                scores: BTreeMap::new(),
                max_largest: 0,
                max_combined_mass: 0,
                combined_mass: BTreeMap::new(),
            });
            entry.scores.insert(s.character.clone(), s);
        }
    }
    // Compute per-page cross-character maxes.
    for page in merged.values_mut() {
        let mut max_largest = 0u32;
        let mut max_combined = 0u32;
        for (key, s) in &page.scores {
            let combined =
                s.hair_pixels + (s.secondary_pixels as f32 * 0.7) as u32;
            page.combined_mass.insert(key.clone(), combined);
            if s.largest_blob_area > max_largest {
                max_largest = s.largest_blob_area;
            }
            if combined > max_combined {
                max_combined = combined;
            }
        }
        page.max_largest = max_largest;
        page.max_combined_mass = max_combined;
    }
    Ok(merged.into_values().collect())
}

/// Recompute certainty for a (page, character) under candidate weights.
fn rescore(page: &MergedPage, character: &str, w: &ScoringWeights) -> f32 {
    let s = match page.scores.get(character) {
        Some(s) => s,
        None => return 0.0,
    };
    let largest = s.largest_blob_area;
    let combined = page.combined_mass.get(character).copied().unwrap_or(0);
    let blob_share = if page.max_largest == 0 {
        1.0
    } else {
        (largest as f32) / (page.max_largest as f32)
    };
    let mass_share = if page.max_combined_mass == 0 {
        1.0
    } else {
        (combined as f32) / (page.max_combined_mass as f32)
    };
    let competition =
        (w.competition_floor + w.competition_slope * blob_share.min(mass_share)).min(1.0);
    combine_score(
        s.hair_pixel_ratio,
        s.largest_blob_ratio,
        s.hair_blobs,
        s.secondary_ratio,
        s.coupling_pixels,
        s.width * s.height,
        s.largest_blob_density,
        s.largest_blob_aspect,
        w,
    ) * competition
}

/// AUC via Mann-Whitney U. O((n_pos + n_neg) log (n_pos + n_neg)).
fn auc(pos: &[f32], neg: &[f32]) -> f32 {
    if pos.is_empty() || neg.is_empty() {
        return 0.5;
    }
    // Combine and sort, tracking which class each came from.
    let mut all: Vec<(f32, u8)> =
        pos.iter().map(|&v| (v, 1)).chain(neg.iter().map(|&v| (v, 0))).collect();
    all.sort_by(|a, b| {
        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    // Average ranks (handle ties).
    let n = all.len();
    let mut ranks = vec![0f32; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        while j + 1 < n && (all[j + 1].0 - all[i].0).abs() < 1e-9 {
            j += 1;
        }
        let avg_rank = (i + j) as f32 / 2.0 + 1.0; // 1-indexed
        for k in i..=j {
            ranks[k] = avg_rank;
        }
        i = j + 1;
    }
    let sum_ranks_pos: f32 = (0..n).filter(|&k| all[k].1 == 1).map(|k| ranks[k]).sum();
    let n_pos = pos.len() as f32;
    let n_neg = neg.len() as f32;
    let u = sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0;
    u / (n_pos * n_neg)
}

fn jitter(default: f32, low: f32, high: f32, rng: &mut impl Rng) -> f32 {
    let m = rng.gen_range(low..high);
    (default * m).max(0.0)
}

fn sample_weights(rng: &mut impl Rng) -> ScoringWeights {
    let d = ScoringWeights::default();
    ScoringWeights {
        blob_weight: jitter(d.blob_weight, JITTER_LOW, JITTER_HIGH, rng),
        mass_weight: jitter(d.mass_weight, JITTER_LOW, JITTER_HIGH, rng),
        mult_weight: jitter(d.mult_weight, JITTER_LOW, JITTER_HIGH, rng),
        secondary_weight: jitter(d.secondary_weight, JITTER_LOW, JITTER_HIGH, rng),
        coupling_weight: jitter(d.coupling_weight, JITTER_LOW, JITTER_HIGH, rng),
        // density_floor and aspect_log_div have natural bounds; sample more
        // tightly so we don't blow them out of useful ranges.
        density_floor: rng.gen_range(0.2..0.7),
        density_slope: rng.gen_range(0.4..2.0),
        aspect_log_div: rng.gen_range(0.8..2.5),
        coupling_cap: rng.gen_range(0.05..0.3),
        // Competition is shared across characters but we still sample per-char
        // because each character's optimal demotion behavior may differ.
        competition_floor: rng.gen_range(0.1..0.5),
        competition_slope: rng.gen_range(0.4..1.0),
    }
}

fn evaluate(
    pages: &[MergedPage],
    labels: &LabelStore,
    character: &str,
    w: &ScoringWeights,
) -> (Vec<f32>, Vec<f32>) {
    let mut pos = Vec::new();
    let mut neg = Vec::new();
    for page in pages {
        let s = match page.scores.get(character) {
            Some(s) => s,
            None => continue,
        };
        let key = LabelStore::page_key(&s.book, &s.filename);
        let v = match labels.get(&key, character) {
            Some(v) => v,
            None => continue,
        };
        let score = rescore(page, character, w);
        match v {
            Verdict::Yes => pos.push(score),
            Verdict::No => neg.push(score),
            Verdict::Unsure => {}
        }
    }
    (pos, neg)
}

pub fn run_tune(out_dir: &Path, characters: &[Character]) -> Result<TunedConfig> {
    let labels_path = default_labels_path(out_dir);
    let labels = LabelStore::load(&labels_path)?;
    if labels.entries.is_empty() {
        anyhow::bail!(
            "no labels at {}: run `misato label` first",
            labels_path.display()
        );
    }
    let merged = load_merged(out_dir, characters)?;
    println!("[tune] loaded {} merged pages", merged.len());

    let mut weights_out: BTreeMap<String, ScoringWeights> = BTreeMap::new();
    let mut diag_out: BTreeMap<String, TuneDiag> = BTreeMap::new();

    let default_w = ScoringWeights::default();

    for ch in characters {
        let (pos_def, neg_def) = evaluate(&merged, &labels, ch.key, &default_w);
        let n_pos = pos_def.len();
        let n_neg = neg_def.len();
        let auc_default = auc(&pos_def, &neg_def);

        if n_pos < MIN_POS || n_neg < MIN_NEG {
            println!(
                "[tune] {}: skipping (need >={} Yes and >={} No, have {} / {}). default AUC={:.3}",
                ch.key, MIN_POS, MIN_NEG, n_pos, n_neg, auc_default
            );
            // Keep default weights for this character; record diagnostic.
            weights_out.insert(ch.key.to_string(), default_w.clone());
            diag_out.insert(
                ch.key.to_string(),
                TuneDiag {
                    n_pos,
                    n_neg,
                    auc_default,
                    auc_tuned: auc_default,
                },
            );
            continue;
        }

        // Random search.
        let mut rng = rand::thread_rng();
        let mut best_w = default_w.clone();
        let mut best_auc = auc_default;
        for _ in 0..N_ITER {
            let cand = sample_weights(&mut rng);
            let (p, n) = evaluate(&merged, &labels, ch.key, &cand);
            let a = auc(&p, &n);
            if a > best_auc {
                best_auc = a;
                best_w = cand;
            }
        }

        println!(
            "[tune] {}: AUC default {:.3} → tuned {:.3}  (n_pos={}, n_neg={})",
            ch.key, auc_default, best_auc, n_pos, n_neg
        );
        weights_out.insert(ch.key.to_string(), best_w);
        diag_out.insert(
            ch.key.to_string(),
            TuneDiag {
                n_pos,
                n_neg,
                auc_default,
                auc_tuned: best_auc,
            },
        );
    }

    let cfg = TunedConfig {
        weights: weights_out,
        diagnostics: diag_out,
        timestamp: chrono_now(),
    };
    let out_path = default_tuned_config_path(out_dir);
    let f = std::fs::File::create(&out_path)?;
    serde_json::to_writer_pretty(f, &cfg)?;
    println!("[tune] wrote {}", out_path.display());
    Ok(cfg)
}

/// Tiny RFC-3339-ish "now" without pulling in chrono. UTC seconds since epoch
/// is enough for our diagnostic field.
fn chrono_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let s = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("epoch:{}", s)
}

// TODO(color-target tuning): add a `tune-deep` subcommand that re-runs the
// full pixel pipeline per candidate over a per-character grid of
// (hue_min/max, sat_min, val_min/max, de_max). Requires caching the
// downscaled RGB pixels per page (~6MB × 812 = ~5GB) so we can avoid JPEG
// decode per iteration. Otherwise ~hours, not minutes.
