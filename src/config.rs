//! Character + color-target configs. Hand-built from HUMAN.md.

use crate::color::Rgb;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Hair color — the dominant signature we cluster on.
    Primary,
    /// Confirmation evidence (jacket, eyes, plugsuit) — boosts certainty when
    /// it co-occurs with the primary nearby.
    Secondary,
}

#[derive(Clone, Debug)]
pub struct ColorTarget {
    pub label: &'static str,
    pub hex: &'static str,
    pub rgb: Rgb,
    /// Hue band (degrees). If `hue_min > hue_max`, wraparound across 360°/0°.
    pub hue_min: f32,
    pub hue_max: f32,
    pub sat_min: f32,
    pub val_min: f32,
    pub val_max: f32,
    /// ΔE76 max in CIELAB.
    pub de_max: f32,
    pub role: Role,
}

#[derive(Clone, Debug)]
pub struct Character {
    pub key: &'static str,
    pub display: &'static str,
    pub targets: Vec<ColorTarget>,
}

impl Character {
    pub fn primary(&self) -> &ColorTarget {
        self.targets
            .iter()
            .find(|t| t.role == Role::Primary)
            .expect("character must have a primary target")
    }

    pub fn secondaries(&self) -> impl Iterator<Item = &ColorTarget> {
        self.targets.iter().filter(|t| t.role == Role::Secondary)
    }
}

/// Build the three NGE characters with hand-tuned tolerances.
///
/// Notes on tuning:
/// - Misato hair (#5B4B7A) is muted violet, H≈258°. We allow a wide hue band
///   because illustrations shift hair into bluer/redder violets under shading.
/// - Rei hair (#B8D4E3) is pale blue, S≈0.20 — close to grey. We force a
///   tight ΔE in LAB to avoid grabbing skies/whites, and require some sat.
/// - Asuka hair (#C44827) sits at H≈14°. Kept distinct from her plugsuit red
///   by hue band: hair allowed in 5..30°, plugsuit red 340..6°.
pub fn build_characters() -> Vec<Character> {
    vec![
        Character {
            key: "misato",
            display: "Misato Katsuragi",
            targets: vec![
                ColorTarget {
                    // Tightened from initial smoke test: a violet background
                    // washes (Fatal Fury page) were sneaking past sat_min=0.15
                    // and de_max=22. Real Misato hair blobs are reliably more
                    // saturated and closer in LAB.
                    label: "hair_violet",
                    hex: "#5B4B7A",
                    rgb: Rgb(0x5B, 0x4B, 0x7A),
                    hue_min: 245.0,
                    hue_max: 280.0,
                    sat_min: 0.22,
                    val_min: 0.20,
                    val_max: 0.78,
                    de_max: 16.0,
                    role: Role::Primary,
                },
                ColorTarget {
                    label: "jacket_red",
                    hex: "#B91C2C",
                    rgb: Rgb(0xB9, 0x1C, 0x2C),
                    hue_min: 345.0,
                    hue_max: 10.0, // wraps
                    sat_min: 0.55,
                    val_min: 0.30,
                    val_max: 0.85, // Asuka's plugsuit is brighter — exclude.
                    de_max: 18.0,
                    role: Role::Secondary,
                },
            ],
        },
        Character {
            key: "rei",
            display: "Rei Ayanami",
            targets: vec![
                ColorTarget {
                    label: "hair_pale_blue",
                    hex: "#B8D4E3",
                    rgb: Rgb(0xB8, 0xD4, 0xE3),
                    hue_min: 180.0,
                    hue_max: 220.0,
                    sat_min: 0.08,
                    val_min: 0.65,
                    val_max: 0.98,
                    de_max: 12.0,
                    role: Role::Primary,
                },
                ColorTarget {
                    label: "eyes_red",
                    hex: "#D72638",
                    rgb: Rgb(0xD7, 0x26, 0x38),
                    hue_min: 350.0,
                    hue_max: 10.0,
                    sat_min: 0.65,
                    val_min: 0.45,
                    val_max: 0.98,
                    de_max: 18.0,
                    role: Role::Secondary,
                },
                ColorTarget {
                    label: "a10_orange",
                    hex: "#FF6B1A",
                    rgb: Rgb(0xFF, 0x6B, 0x1A),
                    hue_min: 15.0,
                    hue_max: 30.0,
                    sat_min: 0.70,
                    val_min: 0.70,
                    val_max: 1.0,
                    de_max: 18.0,
                    role: Role::Secondary,
                },
            ],
        },
        Character {
            key: "asuka",
            display: "Asuka Langley Soryu",
            targets: vec![
                ColorTarget {
                    label: "hair_auburn",
                    hex: "#C44827",
                    rgb: Rgb(0xC4, 0x48, 0x27),
                    hue_min: 5.0,
                    hue_max: 28.0,
                    sat_min: 0.55,
                    val_min: 0.35,
                    val_max: 0.95,
                    de_max: 22.0,
                    role: Role::Primary,
                },
                ColorTarget {
                    label: "plugsuit_red",
                    hex: "#C8102E",
                    rgb: Rgb(0xC8, 0x10, 0x2E),
                    hue_min: 348.0,
                    hue_max: 4.0, // wraps
                    sat_min: 0.75,
                    val_min: 0.40,
                    val_max: 0.95,
                    de_max: 18.0,
                    role: Role::Secondary,
                },
            ],
        },
    ]
}
