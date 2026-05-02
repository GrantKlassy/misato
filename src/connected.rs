//! 4-connected component labeling on a binary mask, with bounding box and
//! area for each component. Iterative flood-fill with a manual stack; no
//! recursion (page masks can have huge connected regions and recursion
//! would blow the stack).

#[derive(Clone, Debug)]
pub struct Blob {
    pub area: u32,
    pub min_x: u32,
    pub min_y: u32,
    pub max_x: u32,
    pub max_y: u32,
    /// Centroid in pixel coordinates.
    pub cx: f32,
    pub cy: f32,
}

impl Blob {
    pub fn width(&self) -> u32 {
        self.max_x - self.min_x + 1
    }
    pub fn height(&self) -> u32 {
        self.max_y - self.min_y + 1
    }
    /// area / bounding-box area, in (0, 1]. A real hair cluster is fairly
    /// compact (≈ 0.4–0.7); a washed-out background of stray matching
    /// pixels is sparse (< 0.3).
    pub fn density(&self) -> f32 {
        let bbox_area = (self.width() as f32) * (self.height() as f32);
        if bbox_area <= 0.0 {
            0.0
        } else {
            (self.area as f32) / bbox_area
        }
    }
    /// Aspect ratio of the bbox. ~1 is square, ≪1 or ≫1 is elongated.
    pub fn aspect(&self) -> f32 {
        let w = self.width() as f32;
        let h = self.height() as f32;
        if h <= 0.0 { 0.0 } else { w / h }
    }
}

/// Label connected components in a binary mask. Returns blobs whose area is
/// at least `min_area` pixels.
pub fn label(mask: &[u8], width: u32, height: u32, min_area: u32) -> Vec<Blob> {
    let w = width as usize;
    let h = height as usize;
    debug_assert_eq!(mask.len(), w * h);

    // visited flag per pixel; we also use it as "already labeled".
    let mut visited = vec![false; mask.len()];
    let mut blobs = Vec::new();
    let mut stack: Vec<(u32, u32)> = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if visited[idx] || mask[idx] == 0 {
                continue;
            }
            // New component — flood-fill.
            stack.clear();
            stack.push((x as u32, y as u32));
            visited[idx] = true;

            let mut area: u32 = 0;
            let mut min_x = x as u32;
            let mut max_x = x as u32;
            let mut min_y = y as u32;
            let mut max_y = y as u32;
            let mut sum_x: u64 = 0;
            let mut sum_y: u64 = 0;

            while let Some((px, py)) = stack.pop() {
                area += 1;
                if px < min_x {
                    min_x = px;
                }
                if px > max_x {
                    max_x = px;
                }
                if py < min_y {
                    min_y = py;
                }
                if py > max_y {
                    max_y = py;
                }
                sum_x += px as u64;
                sum_y += py as u64;

                // 4-neighbors
                if px > 0 {
                    let n = (py as usize) * w + (px as usize - 1);
                    if !visited[n] && mask[n] != 0 {
                        visited[n] = true;
                        stack.push((px - 1, py));
                    }
                }
                if (px as usize) + 1 < w {
                    let n = (py as usize) * w + (px as usize + 1);
                    if !visited[n] && mask[n] != 0 {
                        visited[n] = true;
                        stack.push((px + 1, py));
                    }
                }
                if py > 0 {
                    let n = (py as usize - 1) * w + (px as usize);
                    if !visited[n] && mask[n] != 0 {
                        visited[n] = true;
                        stack.push((px, py - 1));
                    }
                }
                if (py as usize) + 1 < h {
                    let n = (py as usize + 1) * w + (px as usize);
                    if !visited[n] && mask[n] != 0 {
                        visited[n] = true;
                        stack.push((px, py + 1));
                    }
                }
            }

            if area >= min_area {
                blobs.push(Blob {
                    area,
                    min_x,
                    min_y,
                    max_x,
                    max_y,
                    cx: (sum_x as f32) / (area as f32),
                    cy: (sum_y as f32) / (area as f32),
                });
            }
        }
    }
    // Largest first.
    blobs.sort_by(|a, b| b.area.cmp(&a.area));
    blobs
}

/// Count secondary-color pixels that fall within a circular neighborhood of
/// each primary blob's centroid (radius = 2 × blob diagonal). Returns the
/// total such pixels summed over all primary blobs.
pub fn coupling_score(
    secondary_mask: &[u8],
    width: u32,
    primary_blobs: &[Blob],
) -> u32 {
    if primary_blobs.is_empty() {
        return 0;
    }
    let w = width as usize;
    let mut total: u32 = 0;
    for blob in primary_blobs {
        let diag =
            ((blob.width() as f32).powi(2) + (blob.height() as f32).powi(2)).sqrt();
        let radius = 2.0 * diag;
        let r2 = radius * radius;
        let cx = blob.cx;
        let cy = blob.cy;
        let min_x = ((cx - radius).max(0.0)) as u32;
        let min_y = ((cy - radius).max(0.0)) as u32;
        let max_x = (cx + radius) as u32;
        let max_y = (cy + radius) as u32;
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let idx = (y as usize) * w + (x as usize);
                if idx >= secondary_mask.len() {
                    continue;
                }
                if secondary_mask[idx] == 0 {
                    continue;
                }
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                if dx * dx + dy * dy <= r2 {
                    total += 1;
                }
            }
        }
    }
    total
}
