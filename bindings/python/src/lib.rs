use altd_sim::{rl::{RlVecEnv,EnvConfig},training::SensorLayout,track::world::World};
use pyo3::{prelude::*,exceptions::PyValueError,types::PyBytes};
fn err(e:impl ToString)->PyErr {PyValueError::new_err(e.to_string())}
fn bytes<'py>(py:Python<'py>,rows:Vec<Vec<f32>>)->Bound<'py,PyBytes> {
    let v:Vec<u8>=rows.into_iter().flatten().flat_map(f32::to_le_bytes).collect();PyBytes::new(py,&v)
}
#[pyclass(unsendable)]
struct NativeEnv { env:RlVecEnv }
#[pymethods]
impl NativeEnv {
    #[new]
    fn new(scene_json:&str,layout_json:&str,config_json:&str)->PyResult<Self> {
        let c:EnvConfig=serde_json::from_str(config_json).map_err(err)?;
        c.validate().map_err(err)?;
        let scene=serde_json::from_str(scene_json).map_err(err)?;
        let layout=SensorLayout::from_ordered(&serde_json::from_str(layout_json).map_err(err)?).map_err(err)?;
        let world=World::from_scene_with(&scene,c.math_profile);
        Ok(Self{env:RlVecEnv::new(world,layout,c).map_err(err)?})
    }
    fn reset_options<'py>(&mut self,py:Python<'py>,ids:Vec<usize>,options_json:&str)->PyResult<Bound<'py,PyBytes>> {
        let value:serde_json::Value=serde_json::from_str(options_json).map_err(err)?;
        let seeds:Option<Vec<u64>>=serde_json::from_value(value["seeds"].clone()).map_err(err)?;
        let spawns:Option<Vec<[f64;3]>>=serde_json::from_value(value["spawn_specs"].clone()).map_err(err)?;
        Ok(bytes(py,self.env.reset_with(&ids,seeds.as_deref(),spawns.as_deref()).map_err(err)?))
    }
    fn legacy_actions<'py>(&self,py:Python<'py>,network_json:&str,metadata_json:&str,model_json:&str)->PyResult<Bound<'py,PyBytes>> {
        let mut driver=altd_sim::rl::legacy::LegacyDriver::new(&serde_json::from_str(network_json).map_err(err)?,&serde_json::from_str(metadata_json).map_err(err)?,&serde_json::from_str(model_json).map_err(err)?,self.env.world.math).map_err(err)?;
        let values=driver.actions(&self.env).into_iter().flatten().flat_map(f64::to_le_bytes).collect::<Vec<_>>();Ok(PyBytes::new(py,&values))
    }
    fn enable_gpu(&mut self)->PyResult<()> {self.env.enable_gpu().map_err(err)}
    fn schema(&self)->String {self.env.schema().to_string()}
    fn observe<'py>(&mut self,py:Python<'py>,ids:Vec<usize>)->PyResult<Bound<'py,PyBytes>> {Ok(bytes(py,self.env.observe(&ids).map_err(err)?))}
    fn reset<'py>(&mut self,py:Python<'py>,ids:Vec<usize>)->PyResult<Bound<'py,PyBytes>> {Ok(bytes(py,self.env.reset(&ids).map_err(err)?))}
    fn step<'py>(&mut self,py:Python<'py>,actions:&[u8])->PyResult<(Bound<'py,PyBytes>,Bound<'py,PyBytes>,String)> {
        let n=self.env.config.num_envs;
        if actions.len()!=n*5*8 {return Err(err("actions must be contiguous little-endian float64 [N,5]"));}
        let a:Vec<[f64;5]>=actions.chunks_exact(40).map(|row|std::array::from_fn(|j|f64::from_le_bytes(row[j*8..j*8+8].try_into().unwrap()))).collect();
        let b=self.env.step(&a).map_err(err)?;
        let meta=serde_json::json!({"schema":b.schema,"rewards":b.rewards,"terminated":b.terminated,"truncated":b.truncated,"info":b.info}).to_string();
        Ok((bytes(py,b.observations),bytes(py,b.next_observations),meta))
    }
}
#[pymodule]
fn _native(m:&Bound<'_,PyModule>)->PyResult<()> {m.add_class::<NativeEnv>()?;Ok(())}
