use crate::{
    debug_print,
    decoder::{eval_state, eval_state_on_device, INPUT_LEN},
    mcts_trainer::Net,
    selfplay::CollectorMessage,
    utils::TimeStampDebugger,
};

use crossbeam::thread;

use flume::{Receiver, RecvError, RecvTimeoutError, Selector, Sender, TrySendError};

use std::{
    cmp::min,
    collections::VecDeque,
    process,
    time::{Duration, Instant},
};
use tch::{Device, Kind, Tensor};

/// How long the request handler waits for one more request before it flushes a partial batch.
/// Defaults to 1 µs; `TZ_BATCH_TIMEOUT_US` overrides it, so larger batches can be A/B tested
/// without a rebuild. Read once per process.
pub fn partial_batch_timeout() -> Duration {
    static TIMEOUT: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *TIMEOUT.get_or_init(|| {
        let micros = match std::env::var("TZ_BATCH_TIMEOUT_US") {
            Ok(value) => value.trim().parse::<u64>().unwrap_or_else(|_| {
                panic!("TZ_BATCH_TIMEOUT_US must be a whole number of microseconds, got {value:?}")
            }),
            Err(_) => 1,
        };
        Duration::from_micros(micros)
    })
}
pub const TEMP_NETWORK_PREFIX: &str = "tz_temp_net_";

pub struct Packet {
    /// One position's network input, `INPUT_LEN` floats (see `decoder::board_input`).
    pub job: Vec<f32>,
    pub resender: Sender<ReturnMessage>,
    pub id: String,
}

pub struct ExecutorPacket {
    pub job: VecDeque<Vec<f32>>,
    pub resenders: VecDeque<Sender<ReturnMessage>>,
    pub id: VecDeque<String>,
}

pub struct ReturnPacket {
    /// (value head row, policy head row) for this position, as raw network outputs.
    pub packet: (Vec<f32>, Vec<f32>),
    pub id: String,
}

pub enum Message {
    NewNetwork(Result<String, RecvError>),
    JobTensor(Result<Packet, RecvError>), // (converted) tensor from mcts search that needs NN evaluation
    JobTensorExecutor(Result<ExecutorPacket, RecvError>), // (converted) tensor from mcts search that needs NN evaluation

    StopServer, // end the executor process
}

/// Where an executor thread's time goes, summed over one reporting window. Printed every
/// `STATS_INTERVAL` so a throughput problem can be located without a profiler: a large
/// `waiting` share means the generators cannot keep the executor fed, large `fill`/`unpack`
/// shares mean CPU-side batch handling is starving the GPU.
struct ExecutorStats {
    window_start: Instant,
    batches: usize,
    positions: usize,
    min_batch: usize,
    max_batch: usize,
    waiting: Duration,
    cat: Duration,
    eval: Duration,
    unpack: Duration,
    /// Part of `unpack` spent slicing each position's outputs out of the batch tensors.
    slice: Duration,
    /// Part of `unpack` spent sending results back (waking each generator's search task).
    send: Duration,
}

const STATS_INTERVAL: Duration = Duration::from_secs(30);

impl ExecutorStats {
    fn new() -> Self {
        ExecutorStats {
            window_start: Instant::now(),
            batches: 0,
            positions: 0,
            min_batch: usize::MAX,
            max_batch: 0,
            waiting: Duration::ZERO,
            cat: Duration::ZERO,
            eval: Duration::ZERO,
            unpack: Duration::ZERO,
            slice: Duration::ZERO,
            send: Duration::ZERO,
        }
    }

    fn record_batch(
        &mut self,
        size: usize,
        cat: Duration,
        eval: Duration,
        unpack: Duration,
        slice: Duration,
        send: Duration,
    ) {
        self.batches += 1;
        self.positions += size;
        self.min_batch = self.min_batch.min(size);
        self.max_batch = self.max_batch.max(size);
        self.cat += cat;
        self.eval += eval;
        self.unpack += unpack;
        self.slice += slice;
        self.send += send;
    }

    fn report_if_due(&mut self, thread_name: &str) {
        let elapsed = self.window_start.elapsed();
        if elapsed < STATS_INTERVAL {
            return;
        }
        if self.batches > 0 {
            let secs = elapsed.as_secs_f64();
            let share = |d: Duration| 100.0 * d.as_secs_f64() / secs;
            println!(
                "[{thread_name}] {:.0}s: {} batches, mean batch {:.0} (min {}, max {}), {:.0} evals/s; \
                 time: waiting {:.0}%, fill {:.0}%, forward+copy {:.0}%, unpack {:.0}% \
                 (slice {:.0}%, send {:.0}%)",
                secs,
                self.batches,
                self.positions as f64 / self.batches as f64,
                self.min_batch,
                self.max_batch,
                self.positions as f64 / secs,
                share(self.waiting),
                share(self.cat),
                share(self.eval),
                share(self.unpack),
                share(self.slice),
                share(self.send),
            );
        }
        *self = ExecutorStats::new();
    }
}

pub enum ReturnMessage {
    ReturnMessage(Result<ReturnPacket, RecvError>),
}

fn handle_new_graph(
    network: &mut Option<Net>,
    graph: Option<String>,
    _thread_name: &str,
    id: usize,
) {
    debug_print!("{}: loading new net", _thread_name);

    if let Some(network) = network.take() {
        drop(network)
    }

    *network = graph.map(|graph| Net::new_with_device_id(&graph[..], id));
}

fn handle_requests(
    handler_send: Sender<ExecutorPacket>,
    tensor_receiver: Receiver<Packet>,
    max_batch_size: usize,
) {
    let mut input_vec: VecDeque<Vec<f32>> = VecDeque::new();
    let mut output_senders: VecDeque<Sender<ReturnMessage>> = VecDeque::new();
    let mut id_vec: VecDeque<String> = VecDeque::new();
    // maximum time a partial batch may wait before being flushed to the GPU. Self-play
    // datagen fills its large batch well within this window, so this only triggers for
    // finite/low-concurrency workloads (e.g. the GateTester match tail), preventing the
    // partial-batch deadlock where the executor waits forever for a batch that never fills.
    let flush_batch = |input_vec: &mut VecDeque<Vec<f32>>,
                       output_senders: &mut VecDeque<Sender<ReturnMessage>>,
                       id_vec: &mut VecDeque<String>| {
        if input_vec.is_empty() {
            return true;
        }
        let pack = ExecutorPacket {
            job: std::mem::take(input_vec),
            resenders: std::mem::take(output_senders),
            id: std::mem::take(id_vec),
        };
        handler_send.send(pack).is_ok()
    };

    loop {
        match tensor_receiver.recv_timeout(partial_batch_timeout()) {
            Ok(job) => {
                input_vec.push_back(job.job);
                output_senders.push_back(job.resender);
                id_vec.push_back(job.id);
                debug_print!("Executor: received new request!");
                if input_vec.len() >= max_batch_size
                    && !flush_batch(&mut input_vec, &mut output_senders, &mut id_vec)
                {
                    break;
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                if !flush_batch(&mut input_vec, &mut output_senders, &mut id_vec) {
                    break;
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                let _ = flush_batch(&mut input_vec, &mut output_senders, &mut id_vec);
                break;
            }
        }
    }
}

/// Host-side buffers reused for every batch, so evaluating a batch allocates no large host memory.
/// This follows Lc0's CUDA backend, which allocates its input and output buffers once with
/// `cudaHostAlloc`. On CUDA the buffers are pinned, so the copies to and from the GPU go straight
/// to and from them.
struct BatchBuffers {
    capacity: usize,
    /// `[capacity, INPUT_LEN]`
    input: Tensor,
    /// `[capacity, value width]` and `[capacity, policy width]`, created on the first batch, when
    /// the widths of the network's outputs are known.
    outputs: Option<(Tensor, Tensor)>,
    device: Device,
}

impl BatchBuffers {
    fn new(capacity: usize, device: Device) -> Self {
        BatchBuffers {
            capacity,
            input: Self::host_buffer(capacity, INPUT_LEN, device),
            outputs: None,
            device,
        }
    }

    fn host_buffer(rows: usize, width: usize, device: Device) -> Tensor {
        let t = Tensor::zeros([rows as i64, width as i64], (Kind::Float, Device::Cpu));
        match device {
            Device::Cuda(_) => t.pin_memory(device),
            _ => t,
        }
    }

    /// The whole buffer as a flat slice.
    ///
    /// Safety: `t` must be one of this struct's buffers (contiguous `f32` on the CPU), and no
    /// other reference into it may be alive.
    unsafe fn as_mut_slice(t: &Tensor) -> &mut [f32] {
        std::slice::from_raw_parts_mut(t.data_ptr() as *mut f32, t.numel())
    }

    /// Copies the inputs into the start of the input buffer and returns a view of the filled rows.
    fn fill_input(&mut self, inputs: &[Vec<f32>]) -> Tensor {
        assert!(inputs.len() <= self.capacity, "batch larger than buffers");
        let buf = unsafe { Self::as_mut_slice(&self.input) };
        for (row, input) in buf.chunks_exact_mut(INPUT_LEN).zip(inputs) {
            assert_eq!(input.len(), INPUT_LEN, "network input has the wrong length");
            row.copy_from_slice(input);
        }
        self.input.narrow(0, 0, inputs.len() as i64)
    }

    /// Copies the network outputs for `n` positions into the output buffers and returns them as
    /// flat row-major slices, with the width of each row.
    fn read_outputs(
        &mut self,
        value: &Tensor,
        policy: &Tensor,
        n: usize,
    ) -> (&[f32], usize, &[f32], usize) {
        let value = value.reshape([n as i64, -1]);
        let policy = policy.reshape([n as i64, -1]);
        let (vw, pw) = (value.size()[1] as usize, policy.size()[1] as usize);
        let (capacity, device) = (self.capacity, self.device);
        let fits = matches!(&self.outputs, Some((v, p)) if v.size()[1] as usize == vw && p.size()[1] as usize == pw);
        if !fits {
            self.outputs = Some((
                Self::host_buffer(capacity, vw, device),
                Self::host_buffer(capacity, pw, device),
            ));
        }
        let (v_buf, p_buf) = self.outputs.as_ref().unwrap();
        // copy_ converts the dtype as well, so a half-precision network also lands as f32
        v_buf.narrow(0, 0, n as i64).copy_(&value);
        p_buf.narrow(0, 0, n as i64).copy_(&policy);
        let v = unsafe { &Self::as_mut_slice(v_buf)[..n * vw] };
        let p = unsafe { &Self::as_mut_slice(p_buf)[..n * pw] };
        (v, vw, p, pw)
    }
}

pub fn executor_main(
    net_receiver: Receiver<String>,
    tensor_receiver: Receiver<Packet>,
    max_batch_size: usize,
    evals_per_sec_sender: Option<Sender<CollectorMessage>>,
    executor_ready_sender: Option<Sender<bool>>,
    executor_id: usize,
) {
    let mut graph_disconnected = false;
    let mut network: Option<Net> = None;
    let thread_name = std::thread::current()
        .name()
        .unwrap_or("unnamed-executor")
        .to_owned();

    let mut waiting_for_batch_debugger = TimeStampDebugger::create_debug();
    let mut packing_time_debugger = TimeStampDebugger::create_debug();
    let mut evaluation_time_taken_debugger = TimeStampDebugger::create_debug();

    let (handler_send, handler_recv) = flume::bounded::<ExecutorPacket>(1);
    let mut stats = ExecutorStats::new();
    let mut buffers: Option<BatchBuffers> = None;

    thread::scope(|s| {
        let _ = s
            .builder()
            .name(format!("executor-wait-{}", executor_id).to_string())
            .spawn(move |_| {
                handle_requests(handler_send, tensor_receiver, max_batch_size);
            })
            .unwrap();

        loop {
            let mut selector = Selector::new();
            // begin_event_with_color("waiting_for_job", CL_ORANGE);
            let _sw = Instant::now();
            assert!(network.is_some() || !graph_disconnected);

            if !graph_disconnected {
                selector = selector.recv(&net_receiver, Message::NewNetwork);
                waiting_for_batch_debugger.record("waiting_for_batch", &thread_name);
            }

            if let Some(_) = network {
                selector = selector.recv(&handler_recv, Message::JobTensorExecutor);
            }

            let wait_start = Instant::now();
            let message = selector.wait();
            stats.waiting += wait_start.elapsed();

            match message {
                Message::StopServer => break,
                Message::NewNetwork(Ok(graph)) => {
                    handle_new_graph(&mut network, Some(graph), &thread_name, executor_id);
                    match executor_ready_sender {
                        Some(ref sender) => sender.send(true).unwrap(),
                        None => {}
                    }
                }
                Message::JobTensor(_) => {}
                Message::JobTensorExecutor(job) => {
                    let job = job.expect("JobTensor should be available");
                    let mut input_vec = job.job;
                    let mut id_vec = job.id;
                    let mut output_senders = job.resenders;

                    let network = network.as_mut().expect("Network should be available");
                    let batch_size = input_vec.len();
                    let buffers = match &mut buffers {
                        Some(b) if b.device == network.device => b,
                        slot => slot.insert(BatchBuffers::new(max_batch_size, network.device)),
                    };
                    let cat_start = Instant::now();
                    let input_tensors = buffers.fill_input(input_vec.make_contiguous());
                    let cat_time = cat_start.elapsed();

                    evaluation_time_taken_debugger.reset();
                    let eval_start = Instant::now();
                    let (board_eval, policy) =
                        eval_state_on_device(input_tensors, network).expect("Evaluation failed");
                    let (board_eval, value_width, policy, policy_width) =
                        buffers.read_outputs(&board_eval, &policy, batch_size);
                    let eval_time = eval_start.elapsed();
                    evaluation_time_taken_debugger.record("evaluation_time_taken", &thread_name);
                    evaluation_time_taken_debugger.reset();
                    if let Some(ref sender) = evals_per_sec_sender {
                        // Statistics are best-effort. The collector shares this channel with
                        // finished games and may be busy writing a file to a slow (network) drive;
                        // a blocking send here would stall the GPU until it catches up.
                        match sender.try_send(CollectorMessage::ExecutorStatistics(batch_size)) {
                            Ok(()) | Err(TrySendError::Full(_)) => {}
                            Err(TrySendError::Disconnected(_)) => {
                                panic!("collector disconnected")
                            }
                        }
                    }

                    let unpack_start = Instant::now();

                    packing_time_debugger.reset();
                    let mut slice_time = Duration::ZERO;
                    let mut send_time = Duration::ZERO;
                    for i in 0..batch_size {
                        let sender: Sender<ReturnMessage> = output_senders
                            .pop_front()
                            .expect("There should be a sender for each job");
                        let id = id_vec
                            .pop_front()
                            .expect("There should be an ID for each job");
                        let slice_start = Instant::now();
                        let result = (
                            board_eval[i * value_width..(i + 1) * value_width].to_vec(),
                            policy[i * policy_width..(i + 1) * policy_width].to_vec(),
                        );
                        let return_pack = ReturnPacket { packet: result, id };
                        let send_start = Instant::now();
                        slice_time += send_start - slice_start;
                        let sent = sender.send(ReturnMessage::ReturnMessage(Ok(return_pack)));
                        send_time += send_start.elapsed();
                        if sent.is_err() {
                            debug_print!(
                                "Executor: search request was cancelled before the result arrived"
                            );
                        }
                    }

                    packing_time_debugger.record("packing_time", &thread_name);
                    stats.record_batch(
                        batch_size,
                        cat_time,
                        eval_time,
                        unpack_start.elapsed(),
                        slice_time,
                        send_time,
                    );
                    stats.report_if_due(&thread_name);

                    drop(input_vec.drain(0..batch_size));
                    waiting_for_batch_debugger.reset();
                    packing_time_debugger.reset();
                }
                Message::NewNetwork(Err(RecvError::Disconnected)) => {
                    graph_disconnected = true;
                    if network.is_none() {
                        break;
                    }
                }
            }
        }

        process::exit(0)
    })
    .unwrap();
}

pub fn executor_static(
    net_path: String,
    tensor_receiver: Receiver<Packet>, // receive tensors from mcts
    ctrl_receiver: Receiver<Message>,  // receive control messages
    num_threads: usize,
) {
    let max_batch_size = min(1024, num_threads);
    let mut network: Option<Net> = None;
    let thread_name = std::thread::current()
        .name()
        .unwrap_or("unnamed-executor")
        .to_owned();

    let (handler_send, handler_recv) = flume::bounded::<ExecutorPacket>(1);

    handle_new_graph(&mut network, Some(net_path), thread_name.as_str(), 0); // TODO maybe pass an option to use either/both GPUs? and not hardcode static GPU to 1 GPU only?

    let _ = thread::scope(|s| {
        s.builder()
            .name(format!("executor-wait-{}", 0).to_string()) // TODO placeholder
            .spawn(move |_| {
                handle_requests(handler_send, tensor_receiver, max_batch_size);
            })
            .unwrap();

        loop {
            let mut selector = Selector::new();

            // register all receivers in the selector
            selector = selector.recv(&handler_recv, Message::JobTensorExecutor);
            selector = selector.recv(&ctrl_receiver, |_| Message::StopServer);
            let message = selector.wait();
            match message {
                Message::StopServer => {
                    break;
                }
                Message::NewNetwork(_) => {
                    unreachable!(); // handle new network message if needed
                }
                Message::JobTensor(_) => {}
                Message::JobTensorExecutor(job) => {
                    let job = job.expect("JobTensor should be available");
                    let network = network.as_mut().expect("Network should be available");
                    let mut input_vec = job.job;
                    let mut id_vec = job.id;
                    let mut output_senders = job.resenders;

                    let batch_size = min(max_batch_size, input_vec.len());
                    let flat: Vec<f32> = input_vec.range(..batch_size).flatten().copied().collect();
                    let (board_eval, policy) =
                        eval_state(Tensor::from_slice(&flat), network).expect("Evaluation failed");
                    let board_eval: Vec<f32> =
                        Vec::try_from(board_eval.to_kind(Kind::Float).reshape([-1])).unwrap();
                    let policy: Vec<f32> =
                        Vec::try_from(policy.to_kind(Kind::Float).reshape([-1])).unwrap();
                    let value_width = board_eval.len() / batch_size;
                    let policy_width = policy.len() / batch_size;
                    for i in 0..batch_size {
                        let sender = output_senders
                            .pop_front()
                            .expect("There should be a sender for each job");
                        let id = id_vec
                            .pop_front()
                            .expect("There should be an ID for each job");
                        let result = (
                            board_eval[i * value_width..(i + 1) * value_width].to_vec(),
                            policy[i * policy_width..(i + 1) * policy_width].to_vec(),
                        );
                        let return_pack = ReturnPacket { packet: result, id };
                        if sender
                            .send(ReturnMessage::ReturnMessage(Ok(return_pack)))
                            .is_err()
                        {
                            eprintln!("Executor: result receiver dropped; discarding result.");
                        }
                    }
                    drop(input_vec.drain(0..batch_size));
                }
            }
        }
        // return the senders to avoid them being dropped and disconnected
    });
}
