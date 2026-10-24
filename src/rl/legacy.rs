use super::{action, env::RlVecEnv};
use crate::{
    nn::network::{ForwardScratch, Network},
    physics::car::SensorScratch,
    training::SensorLayout,
};
pub struct LegacyDriver {
    network: Network,
    layout: SensorLayout,
    outputs: Vec<String>,
    sensors: SensorScratch,
    inputs: Vec<f64>,
    scratch: ForwardScratch,
}
impl LegacyDriver {
    pub fn new(
        network: &serde_json::Value,
        metadata: &serde_json::Value,
        model: &serde_json::Value,
        math: crate::math::profile::MathProfile,
    ) -> Result<Self, String> {
        let net = Network::try_from_game_export(network)?;
        let layout = SensorLayout::try_from_exports(metadata, model, math)?;
        let outputs: Vec<String> = metadata["outputs"]
            .as_array()
            .ok_or("missing outputs")?
            .iter()
            .map(|x| x.as_str().map(str::to_owned).ok_or("invalid output"))
            .collect::<Result<_, _>>()?;
        if net.shape[0] != layout.names.len()
            || *net.shape.last().unwrap() != outputs.len()
            || outputs.iter().any(|x| !action::NAMES.contains(&x.as_str()))
        {
            return Err("unsupported legacy network layout".into());
        }
        Ok(Self {
            network: net,
            layout,
            outputs,
            sensors: SensorScratch::default(),
            inputs: Vec::new(),
            scratch: ForwardScratch::default(),
        })
    }
    pub fn actions(&mut self, env: &RlVecEnv) -> Vec<[f64; 5]> {
        (0..env.config.num_envs)
            .map(|id| {
                self.layout
                    .read_into(&env.world, env.car(id), &mut self.sensors, &mut self.inputs);
                let values = self
                    .network
                    .forward_into(&self.inputs, &mut self.scratch, env.world.math);
                let mut a = [0.; 5];
                for (name, &v) in self.outputs.iter().zip(values) {
                    let j = action::NAMES.iter().position(|x| *x == name).unwrap();
                    a[j] = v.clamp(action::LOW[j], action::HIGH[j]);
                }
                a
            })
            .collect()
    }
}
