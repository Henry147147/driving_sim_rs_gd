use crate::physics::car::Controls;
pub const NAMES: [&str; 5] = ["Steering", "Acceleration", "Handbrake", "Brake", "Boost"];
pub const LOW: [f64; 5] = [-1., -1., 0., 0., 0.];
pub const HIGH: [f64; 5] = [1.; 5];
pub fn validate(a: &[f64; 5]) -> Result<Controls, String> {
    if a.iter()
        .enumerate()
        .any(|(i, x)| !x.is_finite() || *x < LOW[i] || *x > HIGH[i])
    {
        return Err("RL action outside finite schema bounds".into());
    }
    Ok(Controls {
        steering: a[0],
        acceleration: a[1],
        handbrake: a[2],
        brake: a[3],
        boost: a[4],
    })
}
pub fn pack_gpu(a: [f64; 5]) -> [f64; 5] {
    [a[1], a[0], a[3], a[2], a[4]]
}
pub fn unpack_gpu(a: [f64; 5]) -> [f64; 5] {
    [a[1], a[0], a[3], a[2], a[4]]
}
