//! Source page enumeration. Walks the artbook directories and returns a
//! list of page records.

use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Clone, Debug)]
pub struct Page {
    pub book: String,
    pub path: PathBuf,
    pub filename: String,
}

/// Scan `roots` (each treated as one artbook) and return all jpg/jpeg files.
pub fn enumerate_pages(roots: &[PathBuf]) -> anyhow::Result<Vec<Page>> {
    let mut pages = Vec::new();
    for root in roots {
        let book_name = root
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        if !root.exists() {
            anyhow::bail!("artbook root does not exist: {}", root.display());
        }
        for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            if !is_image(path) {
                continue;
            }
            let filename = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            pages.push(Page {
                book: book_name.clone(),
                path: path.to_path_buf(),
                filename,
            });
        }
    }
    pages.sort_by(|a, b| (&a.book, &a.filename).cmp(&(&b.book, &b.filename)));
    Ok(pages)
}

fn is_image(path: &Path) -> bool {
    match path.extension().and_then(|s| s.to_str()) {
        Some(ext) => {
            let ext = ext.to_ascii_lowercase();
            matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "tif" | "tiff")
        }
        None => false,
    }
}

/// Default artbook roots.
pub fn default_artbook_roots() -> Vec<PathBuf> {
    vec![
        PathBuf::from(
            "/media/starling/data/media/art/Neon.Genesis.Evangelion.Artbook.-.Die.Sterne[GAINAX]",
        ),
        PathBuf::from("/media/starling/data/media/art/Neon Genesis Evangelion - Der Mond"),
        PathBuf::from(
            "/media/starling/data/media/art/Neon Genesis Evangelion - Groundwork of Evangelion Vol.1",
        ),
        PathBuf::from(
            "/media/starling/data/media/art/Neon Genesis Evangelion - Groundwork of Evangelion Vol.2",
        ),
    ]
}
