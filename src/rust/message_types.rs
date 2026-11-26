#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "python", pyclass)]
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct MessageServer {
    pub purpose: MessageType,
}

impl MessageServer {
    pub fn new(purpose: MessageType) -> Self {
        MessageServer { purpose }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl MessageServer {
    #[new]
    fn py_new(purpose: MessageType) -> Self {
        Self::new(purpose)
    }
}

#[cfg_attr(feature = "python", pyclass)]
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub enum MessageType {
    Initialise(Entity),
    JobSendPath(String),
    StatisticsSend(Statistics),
    RequestingNet(),
    NewNetworkPath(String),
    IdentityConfirmation((Entity, usize)),
    JobSendData(Vec<DataFileType>),
    NewNetworkData(Vec<u8>),
    TBLink((String, String)),
    CreateTB(),
    TBLinkRequest(),
    TestResult(GateTesterResult),
    EvaluationRequest(ExternalPacket), // use Vec<f32> to handle raw input data
}

#[cfg_attr(feature = "python", pyclass)]
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct GateTesterResult {
    pub elo: (f64, f64, f64),
    pub accept_new_net: bool,
}

#[cfg_attr(feature = "python", pyclass)]
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct ExternalPacket {
    pub data: Vec<f32>,
    pub datagen_id: usize,
    pub mcts_id: usize,
}

#[cfg_attr(feature = "python", pyclass)]
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub enum DataFileType {
    OffFile(Vec<u8>),
    MetaDataFile(Vec<u8>),
    BinFile(Vec<u8>),
}

#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Eq)]
pub enum Entity {
    RustDataGen,
    PythonTraining,
    TBHost,
    GUIMonitor,
    GateTesterRunner,
}

#[cfg_attr(feature = "python", pyclass)]
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub enum Statistics {
    NodesPerSecond(usize),
    EvalsPerSecond(usize),
}

#[cfg(feature = "python")]
#[pymodule]
#[pyo3(name = "tzrust")]
fn tzrust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MessageServer>()?;
    m.add_class::<ExternalPacket>()?;
    m.add_class::<MessageType>()?;
    m.add_class::<DataFileType>()?;
    m.add_class::<Statistics>()?;
    m.add_class::<Entity>()?;
    Ok(())
}
