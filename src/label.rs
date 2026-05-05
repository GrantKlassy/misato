//! Human-supplied training labels: per-page, per-character ground truth.
//!
//! Labels are keyed by `book/filename` (not full path) so they survive mount
//! moves. They live in `output/labels.json` and are written atomically
//! (temp file + rename) on every mutation.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    Yes,
    No,
    Unsure,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LabelStore {
    /// page_key ("book/filename") -> character_key -> verdict.
    pub entries: BTreeMap<String, BTreeMap<String, Verdict>>,
}

impl LabelStore {
    pub fn page_key(book: &str, filename: &str) -> String {
        format!("{}/{}", book, filename)
    }

    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let f = std::fs::File::open(path)
            .with_context(|| format!("open {}", path.display()))?;
        let store: Self = serde_json::from_reader(f)
            .with_context(|| format!("parse {}", path.display()))?;
        Ok(store)
    }

    /// Atomic save: write to a sibling .tmp file, then rename over the target.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp: PathBuf = path.with_extension("json.tmp");
        {
            let f = std::fs::File::create(&tmp)
                .with_context(|| format!("create {}", tmp.display()))?;
            serde_json::to_writer_pretty(f, self)?;
        }
        std::fs::rename(&tmp, path)
            .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
        Ok(())
    }

    pub fn get(&self, page_key: &str, character: &str) -> Option<Verdict> {
        self.entries.get(page_key).and_then(|m| m.get(character)).copied()
    }

    pub fn set(&mut self, page_key: &str, character: &str, v: Verdict) {
        self.entries
            .entry(page_key.to_string())
            .or_default()
            .insert(character.to_string(), v);
    }

    pub fn clear_page_character(&mut self, page_key: &str, character: &str) {
        if let Some(m) = self.entries.get_mut(page_key) {
            m.remove(character);
            if m.is_empty() {
                self.entries.remove(page_key);
            }
        }
    }

    /// Per-character (Yes, No, Unsure) tallies.
    pub fn counts(&self) -> BTreeMap<String, (u32, u32, u32)> {
        let mut out: BTreeMap<String, (u32, u32, u32)> = BTreeMap::new();
        for chars in self.entries.values() {
            for (ch, v) in chars {
                let entry = out.entry(ch.clone()).or_insert((0, 0, 0));
                match v {
                    Verdict::Yes => entry.0 += 1,
                    Verdict::No => entry.1 += 1,
                    Verdict::Unsure => entry.2 += 1,
                }
            }
        }
        out
    }

    /// How many distinct pages have at least one verdict.
    pub fn labeled_page_count(&self) -> usize {
        self.entries.len()
    }
}

pub fn default_labels_path(out_dir: &Path) -> PathBuf {
    out_dir.join("labels.json")
}
