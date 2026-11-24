use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[pyclass]
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct MessageServer {
    pub purpose: MessageType,
}

#[pymethods]
impl MessageServer {
    #[new]
    pub fn new(purpose: MessageType) -> Self {
        MessageServer { purpose }
    }
}

#[pyclass]
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
    /// Server -> every client after it identifies: the id of the current run. A client
    /// that started under another run exits so the launcher restarts it on this one.
    RunInfo(String),
    /// `server reset` -> the running server: a new run was created (its id). The server
    /// forgets its nets and forwards this to every client, which then restart.
    ResetRun(String),
}

#[pyclass]
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct GateTesterResult {
    pub elo: (f64, f64, f64),
    pub accept_new_net: bool,
}

#[pyclass]
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct ExternalPacket {
    pub data: Vec<f32>,
    pub datagen_id: usize,
    pub mcts_id: usize,
}

#[pyclass]
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub enum DataFileType {
    OffFile(Vec<u8>),
    MetaDataFile(Vec<u8>),
    BinFile(Vec<u8>),
}

#[pyclass(eq, eq_int)]
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Eq)]
pub enum Entity {
    RustDataGen,
    PythonTraining,
    TBHost,
    GUIMonitor,
    GateTesterRunner,
}

#[pyclass]
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub enum Statistics {
    NodesPerSecond(usize),
    EvalsPerSecond(usize),
}

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
