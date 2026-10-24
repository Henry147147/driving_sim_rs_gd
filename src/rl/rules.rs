//! Explicit convex road cells with directed, shared cross-section gates.
//! Sweeps are clipped to the union of adjacent cells, never to default asphalt.
use serde::{Deserialize, Serialize};
pub type Point = [f64; 2];
fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1]]
}
fn cross(a: Point, b: Point) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
fn dot(a: Point, b: Point) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn dist(a: Point, b: Point) -> f64 {
    let d = sub(a, b);
    dot(d, d).sqrt()
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cell {
    pub polygon: Vec<Point>,
    pub start: Point,
    pub end: Point,
    pub gate: [Point; 2],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Route {
    pub schema: String,
    pub id: String,
    pub closed: bool,
    pub spawn: Point,
    pub rotation: f64,
    pub cells: Vec<Cell>,
}
impl Route {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != "altd_rl_route_v1" || self.cells.len() < 2 || !self.rotation.is_finite() {
            return Err("invalid route schema".into());
        }
        for (i, c) in self.cells.iter().enumerate() {
            if c.polygon.len() < 3
                || c.polygon
                    .iter()
                    .chain([&c.start, &c.end, &c.gate[0], &c.gate[1], &self.spawn])
                    .flatten()
                    .any(|x| !x.is_finite())
                || dist(c.start, c.end) < 1.
            {
                return Err("nonfinite/degenerate route cell".into());
            }
            for j in 0..c.polygon.len() {
                let n = c.polygon.len();
                if cross(
                    sub(c.polygon[(j + 1) % n], c.polygon[j]),
                    sub(c.polygon[(j + 2) % n], c.polygon[(j + 1) % n]),
                ) <= 0.
                {
                    return Err("cells must be strictly convex CCW".into());
                }
            }
            if clip(&c.polygon, c.end, c.end).is_none() || clip(&c.polygon, c.start, c.start).is_none() {
                return Err("route center outside cell".into());
            }
            if (i + 1 < self.cells.len() || self.closed)
                && dist(c.end, self.cells[(i + 1) % self.cells.len()].start) > 1e-5
            {
                return Err("disconnected route centerline".into());
            }
            if dist(
                [(c.gate[0][0] + c.gate[1][0]) / 2., (c.gate[0][1] + c.gate[1][1]) / 2.],
                c.end,
            ) > 1e-5
            {
                return Err("gate midpoint must match the route station".into());
            }
            let is_edge = |polygon: &[Point]| {
                (0..polygon.len()).any(|j| {
                    (dist(polygon[j], c.gate[0]) < 1e-5 && dist(polygon[(j + 1) % polygon.len()], c.gate[1]) < 1e-5)
                        || (dist(polygon[j], c.gate[1]) < 1e-5
                            && dist(polygon[(j + 1) % polygon.len()], c.gate[0]) < 1e-5)
                })
            };
            if !is_edge(&c.polygon)
                || ((i + 1 < self.cells.len() || self.closed)
                    && !is_edge(&self.cells[(i + 1) % self.cells.len()].polygon))
            {
                return Err("gate must span a shared complete corridor edge".into());
            }
            if dist(c.gate[0], c.gate[1]) < 1. || c.gate.iter().any(|p| clip(&c.polygon, *p, *p).is_none()) {
                return Err("gate outside cell".into());
            }
        }
        let c = &self.cells[0];
        let d = sub(c.end, c.start);
        if dot(sub(self.spawn, c.start), d).abs() > 1e-4 {
            return Err("spawn must be on first route station".into());
        }
        Ok(())
    }
    pub fn length(&self, n: usize) -> f64 {
        self.cells.iter().take(n).map(|c| dist(c.start, c.end)).sum()
    }
    fn neighbors(&self, cursor: usize) -> Vec<usize> {
        let n = self.cells.len();
        let c = cursor % n;
        let mut ids = vec![c];
        if c > 0 {
            ids.push(c - 1)
        } else if self.closed {
            ids.push(n - 1)
        }
        if c + 1 < n {
            ids.push(c + 1)
        } else if self.closed {
            ids.push(0)
        }
        ids
    }
    pub fn legal_sweep(&self, cursor: usize, a: Point, b: Point) -> bool {
        let mut intervals: Vec<_> = self
            .neighbors(cursor)
            .iter()
            .filter_map(|&i| clip(&self.cells[i].polygon, a, b))
            .collect();
        intervals.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut end = 0.;
        for (lo, hi) in intervals {
            if lo > end + 1e-8 {
                return false;
            }
            end = end.max(hi);
            if end >= 1. - 1e-8 {
                return true;
            }
        }
        false
    }
}
/// Liang-Barsky clipping against all convex half planes; covers the full sweep.
fn clip(poly: &[Point], a: Point, b: Point) -> Option<(f64, f64)> {
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    let d = sub(b, a);
    for j in 0..poly.len() {
        let e = sub(poly[(j + 1) % poly.len()], poly[j]);
        let x = cross(e, sub(a, poly[j]));
        let y = cross(e, d);
        if y.abs() < 1e-12 {
            if x < -1e-6 {
                return None;
            }
        } else {
            let t = (-1e-6 - x) / y;
            if y > 0. {
                lo = lo.max(t)
            } else {
                hi = hi.min(t)
            }
        }
        if lo > hi {
            return None;
        }
    }
    Some((lo.max(0.), hi.min(1.)))
}
#[derive(Clone, Debug)]
pub struct Tracker {
    pub cursor: usize,
    pub progress: f64,
    pub signed_distance: f64,
    pub cell: usize,
    winding: i64,
}
impl Default for Tracker {
    fn default() -> Self {
        Self {
            cursor: 0,
            progress: 0.,
            signed_distance: 0.,
            cell: 0,
            winding: 0,
        }
    }
}
impl Tracker {
    /// Wheel endpoints and wheel chord sweeps are checked independently of center projection.
    pub fn advance(
        &mut self,
        route: &Route,
        target: usize,
        a: Point,
        b: Point,
        old_wheels: &[Point],
        wheels: &[Point],
    ) -> Result<bool, &'static str> {
        if !route.legal_sweep(self.cell, a, b)
            || old_wheels
                .iter()
                .zip(wheels)
                .any(|(&x, &y)| !route.legal_sweep(self.cell, x, y))
        {
            return Err("off_track");
        }
        let mut hits = Vec::new();
        let movement = sub(b, a);
        for (i, c) in route.cells.iter().enumerate() {
            let edge = sub(c.gate[1], c.gate[0]);
            let den = cross(movement, edge);
            if den.abs() < 1e-10 {
                continue;
            }
            let v = sub(c.gate[0], a);
            let t = cross(v, edge) / den;
            let u = cross(v, movement) / den;
            if t > 1e-9 && t <= 1. + 1e-9 && (0.0..=1.0).contains(&u) && dot(movement, sub(c.end, c.start)) > 0. {
                hits.push((t, i));
            }
        }
        hits.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (_, i) in hits {
            if i == self.cursor && self.cursor < target {
                self.cursor += 1;
            } else if i == route.cells.len() - 1 && self.cursor == 0 {
                // Returning across the seam from behind the start is not a lap.
                continue;
            } else if i >= self.cursor {
                return Err("invalid_route");
            }
        }
        if self.cursor >= target {
            self.progress = 1.;
            self.signed_distance = route.length(target);
            return Ok(true);
        }
        // Only reachable cells. Reverse movement can decrease signed progress.
        let mut best = (f64::INFINITY, 0., self.cell, self.winding);
        for i in route.neighbors(self.cell) {
            let c = &route.cells[i];
            let d = sub(c.end, c.start);
            let l = dot(d, d);
            let t = (dot(sub(b, c.start), d) / l).clamp(0., 1.);
            let p = [c.start[0] + t * d[0], c.start[1] + t * d[1]];
            let distance = dist(p, b);
            let mut s = route.cells.iter().take(i).map(|c| dist(c.start, c.end)).sum::<f64>() + t * l.sqrt();
            let mut winding = self.winding;
            if self.cell == 0 && i == route.cells.len() - 1 && route.closed {
                winding -= 1;
            }
            if self.cell == route.cells.len() - 1 && i == 0 && route.closed {
                winding += 1;
            }
            s += winding as f64 * route.length(route.cells.len());
            if distance < best.0 {
                best = (distance, s, i, winding);
            }
        }
        self.cell = best.2;
        self.winding = best.3;
        self.signed_distance = best.1;
        self.progress = (best.1 / route.length(target)).clamp(0., 1.);
        Ok(false)
    }
}
