use serde::Serialize;
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Reward {
    pub progress: f64,
    pub time: f64,
    pub failure: f64,
    pub strict: f64,
    pub total: f64,
}
pub fn compute(before: f64, after: f64, ticks: u64, elapsed: u64, deadline: u64, failed: bool, strict: bool) -> Reward {
    let progress = after - before;
    let time = -(ticks as f64) / (deadline as f64);
    let failure = if failed {
        -((deadline.saturating_sub(elapsed) as f64) / (deadline as f64) + 1.)
    } else {
        0.
    };
    let strict = if failed && strict { -after } else { 0. };
    Reward {
        progress,
        time,
        failure,
        strict,
        total: progress + time + failure + strict,
    }
}
