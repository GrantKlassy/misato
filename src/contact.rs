//! Generate a static HTML contact sheet per character so the user can
//! visually review the top-N ranked pages with their certainty scores.
//!
//! Pages are linked relative to the output dir using their `top_pages/`
//! and `extracted/` files (already produced by the extract pipeline). The
//! HTML is a single self-contained file with inline CSS — no external
//! dependencies and no JS.

use crate::config::Character;
use crate::rank::PageScore;
use anyhow::Result;
use std::fmt::Write as _;
use std::path::Path;

pub fn write_contact_sheet(
    out_dir: &Path,
    character: &Character,
    scores: &[PageScore],
    top_n: usize,
) -> Result<()> {
    let dir = out_dir.join(character.key);
    let html_path = dir.join("index.html");

    let mut html = String::new();
    write!(
        &mut html,
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{display} — Top {top_n} hits</title>
<style>
  body {{ background:#111; color:#eee; font:14px/1.4 system-ui,sans-serif;
         margin:0; padding:24px; }}
  h1 {{ margin:0 0 4px; font-weight:600; }}
  .lede {{ color:#aaa; margin-bottom:24px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr));
          gap:18px; }}
  .card {{ background:#1c1c1c; border-radius:6px; overflow:hidden;
          display:flex; flex-direction:column; }}
  .card img {{ width:100%; display:block; background:#000; }}
  .meta {{ padding:8px 10px; }}
  .meta .rank {{ font-weight:700; color:#fff; }}
  .meta .cert {{ float:right; font-variant-numeric:tabular-nums; color:#7fc7ff; }}
  .meta .src {{ color:#888; font-size:12px; word-break:break-all;
                margin-top:4px; }}
  .meta .stats {{ color:#666; font-size:11px; margin-top:4px;
                  font-variant-numeric:tabular-nums; }}
  a {{ color:inherit; text-decoration:none; }}
  a:hover .card {{ outline:1px solid #444; }}
</style>
</head>
<body>
<h1>{display}</h1>
<div class="lede">Top {top_n} pages by certainty score, drawn from {n_total} pages
across the four NGE artbooks. Click an image to open the full-resolution page.</div>
<div class="grid">
"#,
        display = character.display,
        top_n = top_n,
        n_total = scores.len()
    )?;

    let pick: Vec<&PageScore> = scores
        .iter()
        .filter(|s| s.certainty > 0.0)
        .take(top_n)
        .collect();

    for (rank0, s) in pick.iter().enumerate() {
        let rank = rank0 + 1;
        let book_short = sanitize_book(&s.book);
        let stem = std::path::Path::new(&s.path)
            .file_stem()
            .and_then(|x| x.to_str())
            .unwrap_or("page");
        // Match the filename convention used by the extract pipeline.
        let img_rel = format!(
            "top_pages/rank{:03}_score{:.3}_{}_{}.jpg",
            rank, s.certainty, book_short, stem
        );
        write!(
            &mut html,
            r#"  <a href="{img}" target="_blank"><div class="card">
    <img src="{img}" loading="lazy" alt="rank {rank}">
    <div class="meta">
      <span class="rank">#{rank}</span>
      <span class="cert">cert {cert:.2}</span>
      <div class="src">{book} / {file}</div>
      <div class="stats">blobs {blobs} · largest {largest} · density {dens:.2} · aspect {asp:.2} · {ow}×{oh}</div>
    </div>
  </div></a>
"#,
            img = img_rel,
            rank = rank,
            cert = s.certainty,
            book = s.book,
            file = s.filename,
            blobs = s.hair_blobs,
            largest = s.largest_blob_area,
            dens = s.largest_blob_density,
            asp = s.largest_blob_aspect,
            ow = s.orig_width,
            oh = s.orig_height,
        )?;
    }
    html.push_str("</div>\n</body>\n</html>\n");

    std::fs::write(&html_path, html)?;
    Ok(())
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
