//! misato — find Misato (and Rei + Asuka) on Neon Genesis Evangelion artbook
//! pages by hair-class color clustering.
//!
//! Two-stage pipeline per HUMAN.md:
//!   1. `rank` — score every page by certainty that the character appears,
//!              using HSV-band + LAB ΔE matching followed by connected-
//!              component blob analysis and secondary-color coupling.
//!   2. `extract` — for the top-N ranked pages, re-detect at full res and
//!                  emit padded crops around each detected character region,
//!                  plus the full-res page itself.
//!
//! Use `misato all` to run both back-to-back.

mod classify;
mod color;
mod config;
mod connected;
mod contact;
mod extract;
mod io;
mod rank;

use crate::config::{Character, build_characters};
use crate::extract::{ExtractConfig, extract_for_character};
use crate::io::{Page, default_artbook_roots, enumerate_pages};
use crate::rank::{
    PageScore, RankConfig, group_by_character, rank_pages, write_per_character,
};
use anyhow::Result;
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Override artbook source root(s). Repeatable. Defaults to the four
    /// NGE artbook directories under /mnt/smb/media/art.
    #[arg(long = "art-root")]
    art_roots: Vec<PathBuf>,

    /// Output directory.
    #[arg(short, long, default_value = "output")]
    out_dir: PathBuf,

    /// Restrict to a specific character key (misato | rei | asuka).
    /// Repeatable.
    #[arg(long = "character")]
    characters: Vec<String>,

    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Pipeline 1: rank every page by certainty for each character.
    Rank,
    /// Pipeline 2: extract top-N ranked pages' character regions at full res.
    /// Reads `output/<char>/rankings.json` (must exist).
    Extract {
        /// How many top-ranked pages per character to process.
        #[arg(long, default_value_t = 30)]
        top_n: usize,
        /// Skip saving the full-res page alongside the crops.
        #[arg(long)]
        no_full_page: bool,
    },
    /// Run rank then extract back-to-back.
    All {
        #[arg(long, default_value_t = 30)]
        top_n: usize,
        #[arg(long)]
        no_full_page: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let roots = if cli.art_roots.is_empty() {
        default_artbook_roots()
    } else {
        cli.art_roots.clone()
    };

    let mut characters = build_characters();
    if !cli.characters.is_empty() {
        let want: std::collections::HashSet<String> =
            cli.characters.iter().map(|s| s.to_lowercase()).collect();
        characters.retain(|c| want.contains(c.key));
        if characters.is_empty() {
            anyhow::bail!("no characters matched filter: {:?}", cli.characters);
        }
    }

    println!(
        "characters: {}",
        characters
            .iter()
            .map(|c| c.display)
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("art roots:");
    for r in &roots {
        println!("  - {}", r.display());
    }
    println!("output dir: {}", cli.out_dir.display());

    match cli.cmd {
        Command::Rank => {
            do_rank(&roots, &characters, &cli.out_dir)?;
        }
        Command::Extract { top_n, no_full_page } => {
            do_extract(&characters, &cli.out_dir, top_n, !no_full_page)?;
        }
        Command::All { top_n, no_full_page } => {
            do_rank(&roots, &characters, &cli.out_dir)?;
            do_extract(&characters, &cli.out_dir, top_n, !no_full_page)?;
        }
    }
    Ok(())
}

fn do_rank(
    roots: &[PathBuf],
    characters: &[Character],
    out_dir: &PathBuf,
) -> Result<()> {
    println!("\n[1/2] enumerating pages...");
    let pages: Vec<Page> = enumerate_pages(roots)?;
    println!("  found {} pages", pages.len());

    println!(
        "\n[1/2] ranking {} pages × {} characters...",
        pages.len(),
        characters.len()
    );
    let pb = ProgressBar::new(pages.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
        )
        .unwrap()
        .progress_chars("=>-"),
    );

    let _cfg = RankConfig::default();
    let scores: Vec<PageScore> = rank_pages(&pages, characters, Some(&pb));
    pb.finish_with_message("ranking done");

    let grouped = group_by_character(scores);

    println!("\n[1/2] writing rankings to {}", out_dir.display());
    write_per_character(out_dir, &grouped)?;

    // Print top 5 per character to stdout for an at-a-glance check.
    for (key, scores) in &grouped {
        println!("\nTop 5 — {}:", key);
        for s in scores.iter().take(5) {
            println!(
                "  {:.3}  blobs={:>3} largest={:>5} sec={:>5} couple={:>5}  {}",
                s.certainty,
                s.hair_blobs,
                s.largest_blob_area,
                s.secondary_pixels,
                s.coupling_pixels,
                short_path(&s.path)
            );
        }
    }
    Ok(())
}

fn do_extract(
    characters: &[Character],
    out_dir: &PathBuf,
    top_n: usize,
    save_full: bool,
) -> Result<()> {
    println!("\n[2/2] extracting top-{} pages per character...", top_n);
    for ch in characters {
        let json_path = out_dir.join(&ch.key).join("rankings.json");
        if !json_path.exists() {
            eprintln!(
                "  skip {}: no rankings.json (run `rank` first): {}",
                ch.key,
                json_path.display()
            );
            continue;
        }
        let f = std::fs::File::open(&json_path)?;
        let scores: Vec<PageScore> = serde_json::from_reader(f)?;
        let n_to_process = scores
            .iter()
            .filter(|s| s.certainty > 0.0)
            .take(top_n)
            .count();
        let pb = ProgressBar::new(n_to_process as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{prefix:>8} [{bar:30.cyan/blue}] {pos}/{len} {wide_msg}",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        pb.set_prefix(ch.key.to_string());
        let cfg = ExtractConfig {
            top_n,
            also_save_full_page: save_full,
        };
        let crops = extract_for_character(ch, &scores, out_dir, &cfg, Some(&pb))?;
        pb.finish();
        println!(
            "  {}: {} crops written from top {} pages",
            ch.key,
            crops.len(),
            n_to_process
        );

        contact::write_contact_sheet(out_dir, ch, &scores, top_n)?;
        println!(
            "  {}: contact sheet at {}",
            ch.key,
            out_dir.join(&ch.key).join("index.html").display()
        );
    }
    Ok(())
}

fn short_path(p: &str) -> String {
    let path = std::path::Path::new(p);
    let parent = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(p);
    format!("{}/{}", parent, name)
}
