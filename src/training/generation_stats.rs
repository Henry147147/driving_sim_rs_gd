//! The spread of each per-car statistic over a generation, which Drive Lab
//! stores for every generation (`Session::generation_stats`). One summary per
//! statistic replaces transferring every car's state and metrics.

use crate::training::evolution::metric_index;
use crate::training::TrainingAgent;
use serde::Serialize;
use std::collections::BTreeMap;

/// Equal slices of `[min, max]` in `Spread::bins`.
pub const HISTOGRAM_BINS: usize = 16;
/// Statistics update every 6 ticks, 10 times a second of driving.
const UPDATE_SECONDS: f64 = 0.1;
/// Score counts 5/384 per unit of track length.
const SCORE_PER_UNIT: f64 = 5.0 / 384.0;
/// A track tile's side in track units.
const TILE: f64 = 768.0;

/// One statistic's spread over the cars that have a value for it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Spread {
    pub n: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    /// The population standard deviation.
    pub sd: f64,
    /// P10, P25, the median, P75 and P90, interpolating between ranks.
    pub q: [f64; 5],
    /// Cars in each of `HISTOGRAM_BINS` equal slices of `[min, max]`; every
    /// car falls in the first slice when all values are equal.
    pub bins: Vec<u32>,
}

/// The value at fraction `p` of ascending `sorted`, interpolating between ranks.
fn quantile(sorted: &[f64], p: f64) -> f64 {
    let i = (sorted.len() - 1) as f64 * p;
    let (lo, hi) = (i.floor() as usize, i.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo as f64)
}

/// The spread of finite `values`, or `None` without any. Sums run in
/// ascending order, as Drive Lab's own summaries of older simulators do.
pub fn spread(mut values: Vec<f64>) -> Option<Spread> {
    values.retain(|v| v.is_finite());
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n == 0 {
        return None;
    }
    let mean = values.iter().fold(0.0, |a, v| a + v) / n as f64;
    let variance = values.iter().fold(0.0, |a, v| a + (v - mean).powi(2)) / n as f64;
    let (min, max) = (values[0], values[n - 1]);
    let mut bins = vec![0; HISTOGRAM_BINS];
    for &v in &values {
        let slice = if max > min {
            (((v - min) / (max - min) * HISTOGRAM_BINS as f64).floor() as usize).min(HISTOGRAM_BINS - 1)
        } else {
            0
        };
        bins[slice] += 1;
    }
    Some(Spread {
        n,
        min,
        max,
        mean,
        sd: variance.sqrt(),
        q: [0.1, 0.25, 0.5, 0.75, 0.9].map(|p| quantile(&values, p)),
        bins,
    })
}

/// Each statistic's spread over `agents`, keyed by Drive Lab's statistic ids.
/// Units are those Drive Lab shows: seconds, percentages, tiles and track
/// lengths of score. A statistic no car has a value for is left out.
pub fn generation_stats(agents: &[TrainingAgent]) -> BTreeMap<&'static str, Spread> {
    let mut values: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    let mut add = |id: &'static str, value: f64| values.entry(id).or_default().push(value);
    let line = metric_index("total_racing_line_efficiency").unwrap();
    for a in agents {
        let s = &a.stats;
        add("score", s.total_score);
        if let Some(lap) = s.best_lap_time {
            add("lap", lap);
        }
        add("laps", s.score.lap_count as f64);
        add("collisions", a.car.collision_count as f64);
        let updates = s.update_count as f64;
        add("time", updates * UPDATE_SECONDS);
        add("novelty", s.network_novelty);
        add("distance", s.total_actual_distance * SCORE_PER_UNIT);
        if s.total_actual_distance > 0.0 {
            add("line", s.metrics()[line].unwrap_or(f64::NAN) * 100.0);
        }
        // Cars that never drove have no averages.
        if s.update_count == 0 {
            continue;
        }
        add("speed", s.total_speed / updates);
        add("steering", s.total_abs_steering / updates * 100.0);
        add("throttle", s.total_throttle_change_frequency / updates);
        add("momentum", s.total_momentum_change_frequency / updates);
        add("wall", s.total_distance_from_wall / updates / TILE);
        add("centre", s.total_distance_from_center / updates / TILE);
        // These grow by the update period while the car drifts or brakes.
        add("drifting", s.total_drifting_amount / updates / UPDATE_SECONDS * 100.0);
        add("braking", s.total_braking_frequency / updates / UPDATE_SECONDS * 100.0);
        // Net useful boost (`total_boost_net`) in tanks per second of driving.
        add(
            "boost",
            (s.total_boost_useful - s.total_boost_wasted) / (updates * UPDATE_SECONDS),
        );
    }
    values
        .into_iter()
        .filter_map(|(id, list)| spread(list).map(|s| (id, s)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spreads_hold_exact_quantiles_and_every_value_in_a_slice() {
        let s = spread(vec![4.0, 1.0, 3.0, 2.0, 5.0, f64::NAN]).unwrap();
        assert_eq!((s.n, s.min, s.max, s.mean), (5, 1.0, 5.0, 3.0));
        assert_eq!(s.q, [1.4, 2.0, 3.0, 4.0, 4.6]);
        assert!((s.sd - 2f64.sqrt()).abs() < 1e-12);
        assert_eq!(s.bins.len(), HISTOGRAM_BINS);
        assert_eq!(s.bins.iter().sum::<u32>(), 5);
        assert_eq!((s.bins[0], s.bins[HISTOGRAM_BINS - 1]), (1, 1));
        assert_eq!(spread(vec![f64::NAN]), None);
        assert_eq!(spread(vec![2.0, 2.0]).unwrap().bins[0], 2);
    }
}
