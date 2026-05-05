//! `misato label` web app.
//!
//! Serves a single-page HTML labeler on localhost. The user clicks
//! Yes/No/Unsure for each character on each page; verdicts are persisted to
//! `output/labels.json` after every mutation.
//!
//! Page queue ordering: descending by `max(certainty across characters)`. The
//! algorithm's most-confident pages come first so obvious cases get knocked
//! out fast and borderline cases (where tuning matters) get the most
//! attention. Loaded from `output/<char>/rankings.json` (must exist — run
//! `misato rank` first).

use crate::classify::build_mask;
use crate::config::Character;
use crate::label::{LabelStore, Verdict, default_labels_path};
use crate::rank::{MAX_DIM, PageScore};
use anyhow::{Context, Result};
use axum::extract::{Path as AxPath, State};
use axum::http::{StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use image::imageops::FilterType;
use image::{GenericImageView, ImageReader, RgbaImage};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

/// One page in the queue, merged across characters.
#[derive(Clone, Debug, Serialize)]
pub struct QueuePage {
    pub id: usize,
    pub book: String,
    pub filename: String,
    /// Filesystem path. Not sent to the client; used by /image/:id.
    #[serde(skip)]
    pub path: PathBuf,
    pub orig_width: u32,
    pub orig_height: u32,
    /// character_key -> certainty
    pub scores: BTreeMap<String, f32>,
    /// character_key -> full per-character PageScore (for the detail view)
    #[serde(skip)]
    pub full_scores: BTreeMap<String, PageScore>,
    pub max_certainty: f32,
}

#[derive(Clone)]
pub struct AppState {
    pub pages: Arc<Vec<QueuePage>>,
    pub characters: Arc<Vec<Character>>,
    pub labels: Arc<Mutex<LabelStore>>,
    pub labels_path: PathBuf,
}

/// Build the merged page queue from per-character rankings.json files.
pub fn build_queue(
    out_dir: &std::path::Path,
    characters: &[Character],
) -> Result<Vec<QueuePage>> {
    // Gather per-(book, filename) merged scores across characters.
    let mut merged: BTreeMap<(String, String), QueuePage> = BTreeMap::new();
    for ch in characters {
        let json_path = out_dir.join(ch.key).join("rankings.json");
        if !json_path.exists() {
            anyhow::bail!(
                "missing {}: run `misato rank` first",
                json_path.display()
            );
        }
        let f = std::fs::File::open(&json_path)
            .with_context(|| format!("open {}", json_path.display()))?;
        let scores: Vec<PageScore> = serde_json::from_reader(f)
            .with_context(|| format!("parse {}", json_path.display()))?;
        for s in scores {
            let key = (s.book.clone(), s.filename.clone());
            let entry = merged.entry(key).or_insert_with(|| QueuePage {
                id: 0,
                book: s.book.clone(),
                filename: s.filename.clone(),
                path: PathBuf::from(&s.path),
                orig_width: s.orig_width,
                orig_height: s.orig_height,
                scores: BTreeMap::new(),
                full_scores: BTreeMap::new(),
                max_certainty: 0.0,
            });
            entry.scores.insert(ch.key.to_string(), s.certainty);
            if s.certainty > entry.max_certainty {
                entry.max_certainty = s.certainty;
            }
            entry.full_scores.insert(ch.key.to_string(), s);
        }
    }

    let mut pages: Vec<QueuePage> = merged.into_values().collect();
    pages.sort_by(|a, b| {
        b.max_certainty
            .partial_cmp(&a.max_certainty)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| (&a.book, &a.filename).cmp(&(&b.book, &b.filename)))
    });
    for (i, p) in pages.iter_mut().enumerate() {
        p.id = i;
    }
    Ok(pages)
}

#[derive(Serialize)]
struct QueueResponse {
    total: usize,
    characters: Vec<CharacterInfo>,
    pages: Vec<QueueEntry>,
    labels: BTreeMap<String, BTreeMap<String, Verdict>>,
}

#[derive(Serialize)]
struct CharacterInfo {
    key: String,
    display: String,
}

#[derive(Serialize)]
struct QueueEntry {
    id: usize,
    book: String,
    filename: String,
    scores: BTreeMap<String, f32>,
    max_certainty: f32,
}

async fn get_queue(State(st): State<AppState>) -> Json<QueueResponse> {
    let labels = st.labels.lock().await.entries.clone();
    let entries: Vec<QueueEntry> = st
        .pages
        .iter()
        .map(|p| QueueEntry {
            id: p.id,
            book: p.book.clone(),
            filename: p.filename.clone(),
            scores: p.scores.clone(),
            max_certainty: p.max_certainty,
        })
        .collect();
    Json(QueueResponse {
        total: st.pages.len(),
        characters: st
            .characters
            .iter()
            .map(|c| CharacterInfo {
                key: c.key.to_string(),
                display: c.display.to_string(),
            })
            .collect(),
        pages: entries,
        labels,
    })
}

#[derive(Serialize)]
struct PageDetail {
    id: usize,
    book: String,
    filename: String,
    orig_width: u32,
    orig_height: u32,
    scores: BTreeMap<String, PageScore>,
    verdicts: BTreeMap<String, Verdict>,
}

async fn get_page(
    State(st): State<AppState>,
    AxPath(id): AxPath<usize>,
) -> Result<Json<PageDetail>, StatusCode> {
    let page = st.pages.get(id).ok_or(StatusCode::NOT_FOUND)?;
    let labels = st.labels.lock().await;
    let verdicts: BTreeMap<String, Verdict> = labels
        .entries
        .get(&LabelStore::page_key(&page.book, &page.filename))
        .cloned()
        .unwrap_or_default();
    drop(labels);
    Ok(Json(PageDetail {
        id: page.id,
        book: page.book.clone(),
        filename: page.filename.clone(),
        orig_width: page.orig_width,
        orig_height: page.orig_height,
        scores: page.full_scores.clone(),
        verdicts,
    }))
}

async fn get_image(
    State(st): State<AppState>,
    AxPath(id): AxPath<usize>,
) -> Result<Response, StatusCode> {
    let page = st.pages.get(id).ok_or(StatusCode::NOT_FOUND)?;
    let bytes = tokio::fs::read(&page.path)
        .await
        .map_err(|_| StatusCode::NOT_FOUND)?;
    let mime = match page
        .path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
    {
        Some(ext) if ext == "png" => "image/png",
        Some(ext) if ext == "tif" || ext == "tiff" => "image/tiff",
        _ => "image/jpeg",
    };
    Ok((
        [(header::CONTENT_TYPE, mime), (header::CACHE_CONTROL, "public, max-age=3600")],
        bytes,
    )
        .into_response())
}

#[derive(Deserialize)]
struct LabelRequest {
    id: usize,
    character: String,
    /// Some(verdict) to set; None to clear.
    verdict: Option<Verdict>,
}

#[derive(Serialize)]
struct LabelResponse {
    ok: bool,
    counts: BTreeMap<String, (u32, u32, u32)>,
    labeled_pages: usize,
}

async fn post_label(
    State(st): State<AppState>,
    Json(req): Json<LabelRequest>,
) -> Result<Json<LabelResponse>, StatusCode> {
    let page = st.pages.get(req.id).ok_or(StatusCode::NOT_FOUND)?;
    let key = LabelStore::page_key(&page.book, &page.filename);
    let mut labels = st.labels.lock().await;
    match req.verdict {
        Some(v) => labels.set(&key, &req.character, v),
        None => labels.clear_page_character(&key, &req.character),
    }
    labels
        .save(&st.labels_path)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let counts = labels.counts();
    let labeled_pages = labels.labeled_page_count();
    Ok(Json(LabelResponse {
        ok: true,
        counts,
        labeled_pages,
    }))
}

#[derive(Serialize)]
struct StatsResponse {
    counts: BTreeMap<String, (u32, u32, u32)>,
    labeled_pages: usize,
    total_pages: usize,
}

async fn get_stats(State(st): State<AppState>) -> Json<StatsResponse> {
    let labels = st.labels.lock().await;
    Json(StatsResponse {
        counts: labels.counts(),
        labeled_pages: labels.labeled_page_count(),
        total_pages: st.pages.len(),
    })
}

async fn root() -> Html<&'static str> {
    Html(include_str!("../assets/labeler.html"))
}

async fn get_overlay(
    State(st): State<AppState>,
    AxPath(id): AxPath<usize>,
) -> Result<Response, StatusCode> {
    let page = st.pages.get(id).ok_or(StatusCode::NOT_FOUND)?.clone();
    let chars: Vec<Character> = (*st.characters).clone();
    // Image decode + mask compute is CPU-bound; run on the blocking pool.
    let bytes = tokio::task::spawn_blocking(move || render_overlay(&page.path, &chars))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok((
        [
            (header::CONTENT_TYPE, "image/png"),
            (header::CACHE_CONTROL, "public, max-age=600"),
        ],
        bytes,
    )
        .into_response())
}

/// Render a translucent RGBA PNG overlay layering each character's primary
/// hair-color mask in that character's hair RGB. Same dimensions as the
/// downscaled image used for ranking, so the page's aspect ratio matches —
/// the browser scales the overlay to align with the underlying image.
fn render_overlay(path: &std::path::Path, characters: &[Character]) -> Result<Vec<u8>> {
    let img = ImageReader::open(path)?
        .with_guessed_format()?
        .decode()?;
    let (orig_w, orig_h) = img.dimensions();
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

    let mut out = RgbaImage::from_pixel(w, h, image::Rgba([0, 0, 0, 0]));
    // Layer each character's primary mask. If multiple match the same pixel,
    // later characters overwrite — that's fine, the masks rarely overlap.
    for ch in characters {
        let primary = ch.primary();
        let mask = build_mask(pixels, w, h, primary);
        let cr = primary.rgb.0;
        let cg = primary.rgb.1;
        let cb = primary.rgb.2;
        let n = (w as usize) * (h as usize);
        for i in 0..n {
            if mask[i] == 1 {
                let x = (i as u32) % w;
                let y = (i as u32) / w;
                out.put_pixel(x, y, image::Rgba([cr, cg, cb, 170]));
            }
        }
    }

    let mut buf = Vec::with_capacity((w * h) as usize);
    out.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)?;
    Ok(buf)
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/api/queue", get(get_queue))
        .route("/api/page/:id", get(get_page))
        .route("/api/stats", get(get_stats))
        .route("/api/label", post(post_label))
        .route("/image/:id", get(get_image))
        .route("/overlay/:id", get(get_overlay))
        .with_state(state)
}

/// Boot the server (blocking, runs until Ctrl-C).
pub fn serve(
    out_dir: &std::path::Path,
    characters: &[Character],
    bind: &str,
) -> Result<()> {
    println!("[label] building page queue from {}/<char>/rankings.json", out_dir.display());
    let pages = build_queue(out_dir, characters)?;
    println!("[label] {} pages in queue", pages.len());

    let labels_path = default_labels_path(out_dir);
    let labels = LabelStore::load(&labels_path)?;
    println!(
        "[label] loaded {} existing label entries from {}",
        labels.labeled_page_count(),
        labels_path.display()
    );

    let state = AppState {
        pages: Arc::new(pages),
        characters: Arc::new(characters.to_vec()),
        labels: Arc::new(Mutex::new(labels)),
        labels_path,
    };

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async move {
        let app = build_router(state);
        let listener = tokio::net::TcpListener::bind(bind)
            .await
            .with_context(|| format!("bind {}", bind))?;
        let addr = listener.local_addr()?;
        println!("\n[label] listening on http://{}", addr);
        println!("[label] open that URL in your browser; Ctrl-C to stop\n");
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await?;
        Ok::<(), anyhow::Error>(())
    })?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    println!("\n[label] shutting down");
}
