//! `microllm studio`: a training page on this machine only (std only).
//!
//! It serves viewer/studio.html and checks a run's settings as they are typed with the
//! same rules as the command line (`crate::parse`), plus the ones training would only
//! discover later (data too short, --chars too few steps, no GPU driver). It reads this
//! machine's cores, memory and free disk without changing anything, and on request runs
//! `microllm probe` in a separate, short-lived process to time a few real training steps.
//! Start launches `microllm train ... --serve PORT` as a child process at low priority;
//! Stop ends it. One run at a time. When a run finishes, `microllm landscape` follows it
//! (also at low priority), so the viewer's valley panel has its data.
//!
//! Binds 127.0.0.1 only. Requests must name this server in Host (so other web pages cannot
//! reach it through DNS tricks), and changes (POST) must carry the X-Microllm-Studio header,
//! which a page from another origin cannot send without a preflight this server never grants.

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::autograd::Graph;
use crate::json::{self, Json};
use crate::model::{forward, forward_eval, Bigram, Mlp, Model, Transformer};
use crate::optim::{Kind, Optimizer};
use crate::rng::Rng;
use crate::serve::{content_type, head_too_long, percent_decode, resolve, respond};
use crate::sysinfo;
use crate::tensor::{Tensor, F};
use crate::trace::{array, object, string};
use crate::train::{self, Config, ModelKind};

/// The longest a probe may take before it is stopped.
const PROBE_LIMIT: Duration = Duration::from_secs(120);
const MAX_BODY: usize = 64 * 1024;
/// Train options the page may set. `serve` is the studio's to choose.
const OPTIONS: &[&str] = &[
    "model", "steps", "examples", "context", "emb", "hidden", "heads", "layers", "variant", "optimizer", "muon-lr", "wsd", "val-frac", "lr", "lr-min", "warmup", "wd", "seed", "eval-every",
    "eval-batches", "sample-len", "snap-every", "snap-first", "detail", "throttle-ms", "run", "data", "threads", "device", "vocab", "chars",
];

pub struct Options {
    pub port: u16,
    pub open: bool,
    /// Try later port pairs when `port` or `port + 1` is taken (not when --port was given).
    pub find_port: bool,
}

struct Job {
    child: Child,
    run: String,
    port: u16,
    log: PathBuf,
    started: Instant,
    stopped: bool,
    threads: usize,
    valley: Valley,
}

/// The loss landscape computed after a run finishes (`microllm landscape`), for the valley panel.
enum Valley {
    Waiting,
    Computing(Child),
    Done,
    Failed,
    /// The run did not finish, or the valley was stopped.
    None,
}

struct Studio {
    port: u16,
    roots: Vec<(String, PathBuf)>,
    job: Mutex<Option<Job>>,
    probing: Mutex<()>,
    data: Mutex<HashMap<String, DataInfo>>,
}

#[derive(Clone)]
struct DataInfo {
    bytes: u64,
    modified: Option<std::time::SystemTime>,
    chars: usize,
    /// Distinct characters: the vocabulary of a character-level run.
    alphabet: usize,
}

pub fn run(o: &Options) -> Result<(), String> {
    let dir = project_root().ok_or("cannot find the microllm folder (the one holding viewer/ and data/); start microllm from there")?;
    std::env::set_current_dir(&dir).map_err(|e| format!("entering {}: {e}", dir.display()))?;
    fs::create_dir_all("runs").map_err(|e| format!("creating runs/: {e}"))?;
    let (listener, port) = bind_pair(o.port, if o.find_port { 20 } else { 1 })?;
    let root = |d: &str| fs::canonicalize(d).map(|p| (d.to_string(), p)).map_err(|e| format!("{d}/: {e}"));
    let studio = Arc::new(Studio { port, roots: vec![root("viewer")?, root("runs")?], job: Mutex::new(None), probing: Mutex::new(()), data: Mutex::new(HashMap::new()) });
    let url = format!("http://127.0.0.1:{port}/");
    println!("microllm studio: {url}  (this machine only)");
    println!("Keep this window open while you use the page. Closing it, or Ctrl+C, stops the studio and any training it started.");
    if o.open {
        open_browser(&url);
    }
    // Follow runs even when no page is asking, so the valley starts as soon as training ends.
    let ticker = studio.clone();
    thread::spawn(move || loop {
        thread::sleep(Duration::from_secs(1));
        advance(&ticker);
    });
    for conn in listener.incoming().flatten() {
        let studio = studio.clone();
        thread::spawn(move || {
            let _ = handle(conn, &studio);
        });
    }
    Ok(())
}

/// The folder holding viewer/: the one microllm was started from, else the program's own folder
/// or one above it (a downloaded bundle keeps the program beside viewer/; `cargo run` keeps it
/// in target/release/).
fn project_root() -> Option<PathBuf> {
    let ok = |d: &Path| d.join("viewer").join("studio.html").is_file();
    let here = std::env::current_dir().ok().filter(|d| ok(d));
    here.or_else(|| std::env::current_exe().ok()?.ancestors().skip(1).find(|d| ok(d)).map(Path::to_path_buf))
}

/// The first free pair of ports from `first`: the studio's, and the next one for a run's live view.
fn bind_pair(first: u16, tries: u16) -> Result<(TcpListener, u16), String> {
    for i in 0..tries {
        let Some(p) = first.checked_add(2 * i).filter(|&p| p < u16::MAX) else { break };
        if let Ok(l) = TcpListener::bind(("127.0.0.1", p)) {
            if TcpListener::bind(("127.0.0.1", p + 1)).is_ok() {
                return Ok((l, p));
            }
        }
    }
    Err(if tries == 1 {
        format!("port {first} or {} is in use (is another studio running? try another --port)", first + 1)
    } else {
        format!("no free pair of ports between {first} and {} (are other studios running?)", first.saturating_add(2 * tries - 1))
    })
}

fn open_browser(url: &str) {
    let r = if cfg!(windows) {
        Command::new("cmd").args(["/C", "start", "", url]).spawn()
    } else if cfg!(target_os = "macos") {
        Command::new("open").arg(url).spawn()
    } else {
        Command::new("xdg-open").arg(url).spawn()
    };
    if r.is_err() {
        println!("open {url} in your browser");
    }
}

struct Request {
    method: String,
    path: String,
    query: String,
    headers: HashMap<String, String>,
    body: String,
}

fn read_request(s: &mut TcpStream) -> std::io::Result<Option<Request>> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let head_end = loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i;
        }
        let n = s.read(&mut chunk)?;
        if n == 0 {
            return Ok(None);
        }
        buf.extend_from_slice(&chunk[..n]);
        if head_too_long(&buf) {
            return Ok(None);
        }
    };
    let Ok(head) = std::str::from_utf8(&buf[..head_end]) else { return Ok(None) };
    let mut lines = head.lines();
    let mut first = lines.next().unwrap_or("").split(' ');
    let (method, target) = (first.next().unwrap_or("").to_string(), first.next().unwrap_or(""));
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let mut headers: HashMap<String, String> = HashMap::new();
    for (k, v) in lines.filter_map(|l| l.split_once(':')) {
        // A repeated header could mean different things to different readers: refuse it.
        if headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string()).is_some() {
            return Ok(None);
        }
    }
    let len: usize = headers.get("content-length").and_then(|v| v.parse().ok()).unwrap_or(0);
    if len > MAX_BODY {
        return Ok(None);
    }
    let mut body = buf[head_end + 4..].to_vec();
    while body.len() < len {
        let n = s.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(len);
    Ok(Some(Request { method, path: path.to_string(), query: query.to_string(), headers, body: String::from_utf8_lossy(&body).into_owned() }))
}

fn handle(mut s: TcpStream, st: &Studio) -> std::io::Result<()> {
    s.set_read_timeout(Some(Duration::from_secs(5)))?;
    s.set_write_timeout(Some(Duration::from_secs(10)))?;
    let Some(req) = read_request(&mut s)? else {
        return respond(&mut s, 400, "text/plain", b"bad request");
    };
    let here = [format!("127.0.0.1:{}", st.port), format!("localhost:{}", st.port)];
    if !req.headers.get("host").is_some_and(|h| here.iter().any(|x| x.eq_ignore_ascii_case(h))) {
        return respond(&mut s, 403, "text/plain", b"wrong Host");
    }
    // Every /api request must come from the studio page itself: an Origin, when sent, is this
    // studio, and the custom header is one another site's page cannot add without asking first.
    let origin_ok = req.headers.get("origin").is_none_or(|o| here.iter().any(|h| o.eq_ignore_ascii_case(&format!("http://{h}"))));
    let from_page = origin_ok && req.headers.get("x-microllm-studio").map(String::as_str) == Some("1");
    if req.path.starts_with("/api/") && !from_page {
        return respond(&mut s, 403, "text/plain", b"the studio API answers only the studio page");
    }
    let json = |s: &mut TcpStream, code: u16, body: String| respond(s, code, "application/json", body.as_bytes());
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/") => {
            let msg = "HTTP/1.1 302 Found\r\nLocation: /viewer/studio.html\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            s.write_all(msg.as_bytes())
        }
        ("GET", "/api/system") => json(&mut s, 200, system_json(st)),
        ("GET", "/api/check") => {
            let opts = form(&req.query);
            json(&mut s, 200, check(st, &opts).json())
        }
        ("GET", "/api/status") => json(&mut s, 200, status_json(st)),
        ("POST", path) => {
            if !path.starts_with("/api/") {
                return respond(&mut s, 405, "text/plain", b"only the studio API takes POST");
            }
            let Ok(body) = json::parse(if req.body.is_empty() { "{}" } else { &req.body }) else {
                return respond(&mut s, 400, "text/plain", b"body must be JSON");
            };
            let opts: Vec<(String, String)> = match body.get("opts") {
                Some(Json::Obj(kv)) => kv.iter().map(|(k, v)| (k.clone(), v.as_str().map(str::to_string).or_else(|| v.as_f64().map(|x| format!("{x}"))).unwrap_or_default())).collect(),
                _ => Vec::new(),
            };
            match path {
                "/api/probe" => {
                    let (code, body) = probe_request(st, &opts);
                    json(&mut s, code, body)
                }
                "/api/train" => {
                    let replace = body.get("replace").and_then(Json::as_bool).unwrap_or(false);
                    let (code, body) = start_training(st, &opts, replace);
                    json(&mut s, code, body)
                }
                "/api/stop" => {
                    let mut job = st.job.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(j) = job.as_mut() {
                        j.stopped |= phase(j) == "running";
                        end(j);
                    }
                    drop(job);
                    json(&mut s, 200, status_json(st))
                }
                _ => respond(&mut s, 404, "text/plain", b"not found"),
            }
        }
        ("GET", path) => match resolve(path, &st.roots) {
            Some(file) => match fs::read(&file) {
                Ok(bytes) => respond(&mut s, 200, content_type(&file), &bytes),
                Err(_) => respond(&mut s, 404, "text/plain", b"not found"),
            },
            None => respond(&mut s, 404, "text/plain", b"not found"),
        },
        _ => respond(&mut s, 405, "text/plain", b"GET or POST"),
    }
}

/// `a=1&b=x` → pairs, '+' read as a space.
fn form(q: &str) -> Vec<(String, String)> {
    q.split('&')
        .filter(|kv| !kv.is_empty())
        .filter_map(|kv| {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            Some((percent_decode(&k.replace('+', " "))?, percent_decode(&v.replace('+', " "))?))
        })
        .collect()
}

/// The command-line arguments for these options; blanks mean "use the default".
fn to_args(opts: &[(String, String)]) -> Result<Vec<String>, (String, String)> {
    let mut args = Vec::new();
    for (k, v) in opts {
        let v = v.trim();
        if v.is_empty() {
            continue;
        }
        if !OPTIONS.contains(&k.as_str()) {
            return Err((k.clone(), format!("--{k} cannot be set from the studio")));
        }
        args.push(format!("--{k}"));
        args.push(v.to_string());
    }
    Ok(args)
}

fn datasets() -> Vec<(String, u64)> {
    // Only files that really sit in data/: a link to a file elsewhere is not offered.
    let Ok(root) = fs::canonicalize("data") else { return Vec::new() };
    let mut list: Vec<(String, u64)> = fs::read_dir("data")
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "txt") && e.path().is_file())
        .filter(|e| fs::canonicalize(e.path()).is_ok_and(|p| p.parent() == Some(root.as_path())))
        .map(|e| (format!("data/{}", e.file_name().to_string_lossy()), e.metadata().map_or(0, |m| m.len())))
        .collect();
    list.sort();
    list
}

fn data_info(st: &Studio, path: &str) -> Result<DataInfo, String> {
    let meta = fs::metadata(path).map_err(|e| format!("{path}: {e}"))?;
    let modified = meta.modified().ok();
    if let Some(d) = st.data.lock().unwrap_or_else(|e| e.into_inner()).get(path) {
        if d.bytes == meta.len() && d.modified == modified {
            return Ok(d.clone());
        }
    }
    let d = DataInfo { bytes: meta.len(), modified, ..scan(path)? };
    st.data.lock().unwrap_or_else(|e| e.into_inner()).insert(path.to_string(), d.clone());
    Ok(d)
}

fn scan(path: &str) -> Result<DataInfo, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e} (the text must be UTF-8)"))?;
    let mut seen = std::collections::HashSet::new();
    let mut chars = 0;
    for c in text.chars() {
        seen.insert(c);
        chars += 1;
    }
    Ok(DataInfo { bytes: text.len() as u64, modified: None, chars, alphabet: seen.len() })
}

/// Parameters, counted from the shapes `Bigram::new`, `Mlp::new` and `Transformer::with_variant` create.
pub fn param_count(cfg: &Config, v: usize) -> usize {
    let (t, d) = (cfg.context, cfg.emb);
    match cfg.model {
        ModelKind::Bigram => v * v,
        ModelKind::Mlp => v * d + t * d * cfg.hidden + cfg.hidden + cfg.hidden * v + v,
        ModelKind::Transformer => {
            let qk = if cfg.variant.qk_norm { 2 * (d / cfg.heads) } else { 0 };
            v * d + t * d + cfg.layers * (12 * d * d + 13 * d + qk) + 2 * d + d * v + v
        }
    }
}

/// Numbers in one step's graph (values, plus gradients for everything that has one), counted
/// from the nodes `logits_from` and `forward` create. Matches what `train::graph_numbers` measures
/// before its logits, targets and optimizer terms.
pub fn node_numbers(cfg: &Config, v: usize) -> usize {
    let n = cfg.examples;
    let p = param_count(cfg, v);
    let values = match cfg.model {
        ModelKind::Bigram => n * v + 1,
        ModelKind::Mlp => 2 * n * cfg.context * cfg.emb + 3 * n * cfg.hidden + 2 * n * v + 1,
        ModelKind::Transformer => {
            let (t, d) = (cfg.context, cfg.emb);
            let r = n * t;
            let qk = if cfg.variant.qk_norm { 4 * r * d } else { 0 };
            3 * r * d + cfg.layers * (35 * r * d + qk + 3 * n * cfg.heads * t * t) + 3 * r * d + 2 * r * v + 1
        }
    };
    2 * values + 2 * p
}

fn logits_numbers(cfg: &Config, v: usize) -> usize {
    cfg.examples * if cfg.model == ModelKind::Transformer { cfg.context } else { 1 } * v
}

/// How many of `0..steps` satisfy `train::is_snapshot_step`, without looping over every step.
fn snapshot_count(cfg: &Config) -> usize {
    let (s, e) = (cfg.steps, cfg.snap_every);
    let a = cfg.snap_first.min(s);
    let multiples = s.div_ceil(e) - a.div_ceil(e);
    let last = usize::from(s - 1 >= a && (s - 1) % e != 0);
    a + multiples + last
}

fn eval_count(cfg: &Config) -> usize {
    cfg.steps.div_ceil(cfg.eval_every) + usize::from((cfg.steps - 1) % cfg.eval_every != 0)
}

struct Check {
    error: Option<(String, String)>,
    warnings: Vec<(String, String)>,
    fields: Vec<(&'static str, String)>,
    config: Option<Config>,
}

impl Check {
    fn json(&self) -> String {
        let pair = |(f, m): &(String, String)| object(&[("field", string(f)), ("message", string(m))]);
        let mut fields = vec![
            ("ok", self.error.is_none().to_string()),
            ("error", self.error.as_ref().map_or("null".into(), pair)),
            ("warnings", array(&self.warnings.iter().map(pair).collect::<Vec<_>>())),
        ];
        fields.extend(self.fields.iter().cloned());
        object(&fields)
    }
}

/// The option an error message is about: the first `--name` in it, or a known lead word.
fn field_of(msg: &str) -> String {
    for (lead, f) in [("run name", "run"), ("unknown model", "model"), ("unknown optimizer", "optimizer"), ("unknown detail", "detail"), ("unknown device", "device"), ("unknown variant", "variant")] {
        if msg.starts_with(lead) {
            return f.into();
        }
    }
    msg.split("--").nth(1).map(|r| r.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '-').collect()).unwrap_or_default()
}

fn check(st: &Studio, opts: &[(String, String)]) -> Check {
    let mut c = Check { error: None, warnings: Vec::new(), fields: Vec::new(), config: None };
    let args = match to_args(opts) {
        Ok(a) => a,
        Err(e) => {
            c.error = Some(e);
            return c;
        }
    };
    let cfg = match crate::parse(&args) {
        Ok(cfg) => cfg,
        Err(e) => {
            let msg = e.lines().next().unwrap_or("").to_string();
            c.error = Some((field_of(&msg), msg));
            return c;
        }
    };
    c.fields.push(("resolved", resolved_json(&cfg)));
    let fail = |c: &mut Check, f: &str, m: String| c.error = Some((f.to_string(), m));
    if !datasets().iter().any(|(p, _)| *p == cfg.data) {
        fail(&mut c, "data", format!("{} is not a .txt file in data/", cfg.data));
        return c;
    }
    let info = match data_info(st, &cfg.data) {
        Ok(i) => i,
        Err(e) => {
            fail(&mut c, "data", e);
            return c;
        }
    };
    let bpe = cfg.vocab > info.alphabet;
    let v = if bpe { cfg.vocab } else { info.alphabet };
    let tf = cfg.model == ModelKind::Transformer;
    // Training checks these only after loading the data; here they are known now.
    let ctx = match cfg.model {
        ModelKind::Bigram => 1,
        _ => cfg.context,
    };
    let val_chars = (info.chars as F * cfg.val_frac) as usize;
    if !bpe && (val_chars <= ctx || info.chars - val_chars <= ctx) {
        fail(&mut c, "context", format!("{} has {} characters; with --val-frac {} the validation split ({val_chars}) must be longer than the context ({ctx})", cfg.data, info.chars, cfg.val_frac));
        return c;
    }
    let mut cfg = cfg;
    if cfg.chars > 0 {
        // As train.rs: steps = chars / (characters of targets per step). Characters per token is 1
        // without BPE; with BPE it is only known after the tokenizer is learned, so 1 gives the most steps.
        let per_step = cfg.examples * if tf { cfg.context } else { 1 };
        let steps = cfg.chars.div_ceil(per_step);
        if steps <= cfg.warmup {
            let m = format!("--chars {} gives {steps} steps, not more than --warmup ({})", cfg.chars, cfg.warmup);
            if bpe {
                c.warnings.push(("chars".into(), format!("{m} with one character per token; BPE tokens hold more, so even fewer")));
            } else {
                fail(&mut c, "chars", m);
                return c;
            }
        }
        cfg.steps = steps.max(1);
        if cfg.warmup >= cfg.steps {
            cfg.warmup = cfg.steps - 1;
        }
    }
    if cfg.gpu && !sysinfo::system(Path::new(".")).nvidia_driver {
        fail(&mut c, "device", "no NVIDIA driver found on this machine; use --device cpu".into());
        return c;
    }
    let sys = sysinfo::system(Path::new("."));
    if cfg.threads >= sys.cores && sys.cores > 1 {
        c.warnings.push(("threads".into(), format!("{} threads uses every core ({}); the rest of the system may stall. {} leaves room", cfg.threads, sys.cores, sys.cores.saturating_sub(2).max(1))));
    }
    if bpe && info.bytes > 20_000_000 {
        c.warnings.push(("vocab".into(), format!("learning a {}-token BPE tokenizer on {:.0} MB of text takes a while before the first step", cfg.vocab, info.bytes as F / 1e6)));
    }
    if Path::new("runs").join(&cfg.run).exists() {
        c.warnings.push(("run".into(), format!("runs/{} already exists; starting replaces it", cfg.run)));
    }

    let p = param_count(&cfg, v);
    let nodes = node_numbers(&cfg, v);
    let adam = cfg.optimizer != Kind::Sgd;
    let optim = p * if adam { 5 } else { 1 };
    let (logits, targets) = (logits_numbers(&cfg, v), cfg.examples * if tf { cfg.context } else { 1 });
    let full = nodes + logits + targets + optim;
    let snaps = snapshot_count(&cfg);
    let detail = cfg.detail.unwrap_or(if (full * snaps) as F * train::BYTES_PER_NUMBER > train::AUTO_FULL_LIMIT_BYTES { crate::trace::Detail::Summary } else { crate::trace::Detail::Full });
    let per_snap = match detail {
        crate::trace::Detail::Full => full,
        // Summary keeps one example's rows of each per-example node, and all of the parameters.
        crate::trace::Detail::Summary => (nodes - 2 * p + logits + targets) / cfg.examples + 2 * p + optim,
    };
    let snap_bytes = per_snap as F * train::BYTES_PER_NUMBER;
    // model.js: every parameter as base64 f64 (8 bytes → 32/3 characters).
    let model_bytes = p as F * 32.0 / 3.0;
    let disk = snap_bytes * snaps as F + model_bytes;
    // The text, and one 8-byte token id per character (fewer with BPE).
    let data_memory = info.bytes as F + 8.0 * info.chars as F / if bpe { 2.5 } else { 1.0 };
    if let Some(free) = sys.disk_free {
        if disk > 0.9 * free as F {
            fail(&mut c, "snap-every", format!("snapshots would need about {}, and the disk has {} free; raise --snap-every or use --detail summary", gb(disk), gb(free as F)));
            return c;
        }
        if disk > 0.5 * free as F {
            c.warnings.push(("snap-every".into(), format!("snapshots would take about {} of the {} free on this disk", gb(disk), gb(free as F))));
        }
    }
    let num = |x: usize| x.to_string();
    c.fields.extend([
        ("vocab", num(v)),
        ("vocab_exact", (!bpe).to_string()),
        ("data_chars", num(info.chars)),
        ("data_bytes", info.bytes.to_string()),
        ("params", num(p)),
        ("steps", num(cfg.steps)),
        ("steps_exact", (cfg.chars == 0 || !bpe).to_string()),
        ("snapshots", num(snaps)),
        ("evals", num(eval_count(&cfg))),
        ("graph_numbers", num(full)),
        ("detail", string(detail.name())),
        ("snapshot_bytes", format!("{snap_bytes:.0}")),
        ("disk_bytes", format!("{disk:.0}")),
        ("data_memory_bytes", format!("{data_memory:.0}")),
    ]);
    c.config = Some(cfg);
    c
}

fn gb(b: F) -> String {
    if b >= 1e9 { format!("{:.1} GB", b / 1e9) } else { format!("{:.0} MB", b / 1e6) }
}

/// Every setting as the run will use it, keyed by option name.
fn resolved_json(c: &Config) -> String {
    let f = |x: F| format!("{x}");
    let variant: Vec<&str> = [(c.variant.relu2, "relu2"), (c.variant.qk_norm, "qk-norm"), (c.variant.zero_init, "zero-init")].iter().filter(|(on, _)| *on).map(|(_, n)| *n).collect();
    let opt = match c.optimizer {
        Kind::Sgd => "sgd",
        Kind::AdamW => "adamw",
        Kind::Muon => "muon",
    };
    let model = match c.model {
        ModelKind::Bigram => "bigram",
        ModelKind::Mlp => "mlp",
        ModelKind::Transformer => "transformer",
    };
    object(&[
        ("model", string(model)),
        ("data", string(&c.data)),
        ("run", string(&c.run)),
        ("val-frac", f(c.val_frac)),
        ("steps", c.steps.to_string()),
        ("examples", c.examples.to_string()),
        ("context", c.context.to_string()),
        ("emb", c.emb.to_string()),
        ("hidden", c.hidden.to_string()),
        ("heads", c.heads.to_string()),
        ("layers", c.layers.to_string()),
        ("variant", string(&if variant.is_empty() { "none".to_string() } else { variant.join(",") })),
        ("optimizer", string(opt)),
        ("lr", f(c.lr)),
        ("lr-min", f(c.lr_min)),
        ("warmup", c.warmup.to_string()),
        ("wsd", f(c.wsd)),
        ("muon-lr", f(c.muon_lr)),
        ("wd", f(c.weight_decay)),
        ("seed", c.seed.to_string()),
        ("eval-every", c.eval_every.to_string()),
        ("eval-batches", c.eval_batches.to_string()),
        ("sample-len", c.sample_len.to_string()),
        ("snap-every", c.snap_every.to_string()),
        ("snap-first", c.snap_first.to_string()),
        ("detail", string(c.detail.map_or("auto", |d| d.name()))),
        ("throttle-ms", c.throttle_ms.to_string()),
        ("threads", c.threads.to_string()),
        ("device", string(if c.gpu { "gpu" } else { "cpu" })),
        ("vocab", c.vocab.to_string()),
        ("chars", c.chars.to_string()),
    ])
}

fn opt_u64(x: Option<u64>) -> String {
    x.map_or("null".into(), |v| v.to_string())
}

fn system_json(st: &Studio) -> String {
    let s = sysinfo::system(Path::new("."));
    let data: Vec<String> = datasets().iter().map(|(p, b)| object(&[("path", string(p)), ("bytes", b.to_string())])).collect();
    let mut runs: Vec<String> = fs::read_dir("runs").into_iter().flatten().flatten().filter(|e| e.path().join("meta.js").exists()).map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    runs.sort();
    object(&[
        ("os", string(s.os)),
        ("arch", string(s.arch)),
        ("cores", s.cores.to_string()),
        ("ram_total", opt_u64(s.ram_total)),
        ("ram_available", opt_u64(s.ram_available)),
        ("disk_free", opt_u64(s.disk_free)),
        ("nvidia_driver", s.nvidia_driver.to_string()),
        ("gpu_kernels", string(crate::cuda::ptx_build())),
        ("datasets", array(&data)),
        ("runs", array(&runs.iter().map(|r| string(r)).collect::<Vec<_>>())),
        ("train_port", (st.port + 1).to_string()),
    ])
}

fn tail(path: &Path, lines: usize) -> String {
    let text = fs::read(path).map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default();
    let all: Vec<&str> = text.lines().collect();
    all[all.len().saturating_sub(lines)..].join("\n")
}

/// Where a job is. With --serve, train keeps serving after it ends, so a live process can
/// already be finished or failed; its log says which.
fn phase(j: &mut Job) -> &'static str {
    let log = fs::read_to_string(&j.log).unwrap_or_default();
    if log.contains("training finished; still serving") {
        return "finished";
    }
    if log.contains("so the failure can be inspected") {
        return "failed";
    }
    match (j.child.try_wait().ok().flatten(), j.stopped) {
        (None, _) => "running",
        (Some(_), true) => "stopped",
        (Some(e), false) if e.success() => "finished",
        (Some(_), false) => "failed",
    }
}

/// Kill the job's processes that are still alive: training (or only serving its viewer) and the valley.
fn end(j: &mut Job) {
    if j.child.try_wait().ok().flatten().is_none() {
        let _ = j.child.kill();
        let _ = j.child.wait();
    }
    if let Valley::Computing(c) = &mut j.valley {
        let _ = c.kill();
        let _ = c.wait();
        j.valley = Valley::None;
    }
}

/// Start the valley once a run has finished, and note when it is done.
fn advance(st: &Studio) {
    let mut job = st.job.lock().unwrap_or_else(|e| e.into_inner());
    let Some(j) = job.as_mut() else { return };
    if matches!(j.valley, Valley::Waiting) {
        j.valley = match phase(j) {
            "running" => Valley::Waiting,
            "finished" if Path::new("runs").join(&j.run).join("model.js").exists() => start_valley(j),
            _ => Valley::None,
        };
    } else if let Valley::Computing(c) = &mut j.valley {
        if let Ok(Some(s)) = c.try_wait() {
            j.valley = if s.success() { Valley::Done } else { Valley::Failed };
        }
    }
}

/// `microllm landscape --run NAME` at low priority, its output appended to the run's console log.
fn start_valley(j: &Job) -> Valley {
    let args = ["--run".to_string(), j.run.clone(), "--threads".into(), j.threads.to_string()];
    let Ok(out) = fs::OpenOptions::new().append(true).open(&j.log) else { return Valley::Failed };
    let (Ok(mut cmd), Ok(err)) = (low_priority("landscape", &args), out.try_clone()) else { return Valley::Failed };
    match cmd.stdin(Stdio::null()).stdout(Stdio::from(out)).stderr(Stdio::from(err)).spawn() {
        Ok(c) => Valley::Computing(c),
        Err(_) => Valley::Failed,
    }
}

fn status_json(st: &Studio) -> String {
    let mut job = st.job.lock().unwrap_or_else(|e| e.into_inner());
    let Some(j) = job.as_mut() else {
        return object(&[("state", string("idle"))]);
    };
    let state = phase(j);
    let alive = j.child.try_wait().ok().flatten().is_none();
    let exit = j.child.try_wait().ok().flatten();
    // "live view:" is printed once near the start, so look at the whole log, not the tail.
    let serving = alive && fs::read_to_string(&j.log).unwrap_or_default().contains("live view:");
    let log = tail(&j.log, 40);
    object(&[
        ("state", string(state)),
        ("run", string(&j.run)),
        ("port", j.port.to_string()),
        ("seconds", format!("{:.0}", j.started.elapsed().as_secs_f64())),
        ("exit_code", exit.and_then(|e| e.code()).map_or("null".into(), |c| c.to_string())),
        ("serving", serving.to_string()),
        ("valley", string(match j.valley {
            Valley::Waiting => "waiting",
            Valley::Computing(_) => "computing",
            Valley::Done => "done",
            Valley::Failed => "failed",
            Valley::None => "none",
        })),
        ("log", string(&log)),
    ])
}

/// `microllm <sub> args…` at below-normal priority, so the desktop stays responsive.
fn low_priority(sub: &str, args: &[String]) -> Result<Command, String> {
    let exe = std::env::current_exe().map_err(|e| format!("finding microllm itself: {e}"))?;
    #[cfg(windows)]
    let cmd = {
        use std::os::windows::process::CommandExt;
        let mut c = Command::new(exe);
        c.creation_flags(0x0000_4000); // BELOW_NORMAL_PRIORITY_CLASS
        c.arg(sub).args(args);
        c
    };
    #[cfg(not(windows))]
    let cmd = {
        let mut c = Command::new("nice");
        c.args(["-n", "10"]).arg(exe).arg(sub).args(args);
        c
    };
    Ok(cmd)
}

fn busy(st: &Studio) -> bool {
    st.job.lock().unwrap_or_else(|e| e.into_inner()).as_mut().is_some_and(|j| phase(j) == "running" || matches!(j.valley, Valley::Computing(_)))
}

fn error_json(field: &str, msg: &str) -> String {
    object(&[("ok", "false".into()), ("error", object(&[("field", string(field)), ("message", string(msg))]))])
}

fn probe_request(st: &Studio, opts: &[(String, String)]) -> (u16, String) {
    let c = check(st, opts);
    if c.config.is_none() {
        return (400, c.json());
    }
    if busy(st) {
        return (409, error_json("", "training (or its valley) is running; a timing now would be wrong and would slow it down"));
    }
    let Ok(_guard) = st.probing.try_lock() else {
        return (409, error_json("", "a timing is already running"));
    };
    let args = match to_args(opts) {
        Ok(a) => a,
        Err((f, m)) => return (400, error_json(&f, &m)),
    };
    let mut cmd = match low_priority("probe", &args) {
        Ok(c) => c,
        Err(e) => return (500, error_json("", &e)),
    };
    let mut child = match cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(e) => return (500, error_json("", &format!("starting the probe: {e}"))),
    };
    let (mut out, mut err) = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
    let reader = thread::spawn(move || {
        let (mut o, mut e) = (String::new(), String::new());
        let _ = out.read_to_string(&mut o);
        let _ = err.read_to_string(&mut e);
        (o, e)
    });
    let t0 = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) if t0.elapsed() > PROBE_LIMIT => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            Ok(None) => thread::sleep(Duration::from_millis(50)),
            Err(_) => break None,
        }
    };
    let (o, e) = reader.join().unwrap_or_default();
    match status {
        Some(s) if s.success() => (200, o.trim().to_string()),
        Some(_) => {
            let msg = e.lines().last().unwrap_or("the probe failed").trim_start_matches("error: ");
            (200, error_json(&field_of(msg), msg))
        }
        None => (200, error_json("", &format!("one training step took longer than {} s; this setting is too big for this machine", PROBE_LIMIT.as_secs()))),
    }
}

fn start_training(st: &Studio, opts: &[(String, String)], replace: bool) -> (u16, String) {
    let c = check(st, opts);
    let Some(cfg) = c.config else { return (400, c.json()) };
    let mut job = st.job.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(j) = job.as_mut() {
        if phase(j) == "running" {
            return (409, error_json("", "a run is already training; stop it first"));
        }
        if matches!(j.valley, Valley::Computing(_)) {
            return (409, error_json("", &format!("the valley of runs/{} is still being computed (about a minute); wait, or press Stop", j.run)));
        }
        // A finished run may still be serving its viewer on the training port.
        end(j);
    }
    if st.probing.try_lock().is_err() {
        return (409, error_json("", "a timing is running; try again in a moment"));
    }
    let dir = Path::new("runs").join(&cfg.run);
    // Any existing folder counts, finished or not: replacing deletes its files.
    if dir.exists() && !replace {
        return (409, error_json("run", &format!("runs/{} already exists; tick \"replace the existing run of this name\" or choose another name", cfg.run)));
    }
    let port = st.port + 1;
    if TcpListener::bind(("127.0.0.1", port)).is_err() {
        return (409, error_json("", &format!("port {port} is in use (an earlier --serve run?); stop it first")));
    }
    let mut args = match to_args(opts) {
        Ok(a) => a,
        Err((f, m)) => return (400, error_json(&f, &m)),
    };
    args.extend(["--serve".to_string(), port.to_string()]);
    let log = dir.join("console.log");
    let files = fs::create_dir_all(&dir).and_then(|_| fs::File::create(&log)).and_then(|f| Ok((f.try_clone()?, f)));
    let (out, err) = match files {
        Ok(f) => f,
        Err(e) => return (500, error_json("", &format!("creating {}: {e}", log.display()))),
    };
    let mut cmd = match low_priority("train", &args) {
        Ok(c) => c,
        Err(e) => return (500, error_json("", &e)),
    };
    match cmd.stdin(Stdio::null()).stdout(Stdio::from(out)).stderr(Stdio::from(err)).spawn() {
        Ok(child) => {
            *job = Some(Job { child, run: cfg.run.clone(), port, log, started: Instant::now(), stopped: false, threads: cfg.threads, valley: Valley::Waiting });
            drop(job);
            (200, status_json(st))
        }
        Err(e) => (500, error_json("", &format!("starting training: {e}"))),
    }
}

/// `microllm probe [train options]`: build the model these options describe and time a few
/// real training steps on random tokens, then print one JSON line. The studio runs it as a
/// separate process, so a setting too big for this machine cannot take the studio down.
pub fn probe(args: &[String]) -> Result<(), String> {
    let cfg = crate::parse(args)?;
    crate::par::set_threads(cfg.threads);
    let info = scan(&cfg.data)?;
    let v = if cfg.vocab > info.alphabet { cfg.vocab } else { info.alphabet };
    let mut init_rng = Rng::new(cfg.seed ^ 0x1A17);
    let mut model: Box<dyn Model> = match cfg.model {
        ModelKind::Bigram => Box::new(Bigram::new(v)),
        ModelKind::Mlp => Box::new(Mlp::new(v, cfg.context, cfg.emb, cfg.hidden, &mut init_rng)),
        ModelKind::Transformer => Box::new(Transformer::with_variant(v, cfg.context, cfg.emb, cfg.heads, cfg.layers, cfg.variant, &mut init_rng)),
    };
    let mut opt = match cfg.optimizer {
        Kind::Sgd => Optimizer::sgd(cfg.lr),
        Kind::AdamW => Optimizer::adamw(cfg.lr, cfg.weight_decay),
        Kind::Muon => Optimizer::muon(cfg.lr, cfg.muon_lr, cfg.weight_decay, model.params()),
    };
    let adam = opt.kind != Kind::Sgd;
    let gpu_name = if cfg.gpu { Some(crate::cuda::enable().map_err(|e| format!("the GPU could not be opened: {e}"))?) } else { None };
    let resident = cfg.gpu.then(|| crate::gpu::Resident::new(model.params(), &opt.muon));
    let (ctx, per) = (model.context(), model.targets_per_example());
    let mut rng = Rng::new(1);
    let mut batch = || {
        let x: Vec<usize> = (0..cfg.examples * ctx).map(|_| rng.below(v)).collect();
        let y: Vec<usize> = (0..cfg.examples * per).map(|_| rng.below(v)).collect();
        (x, y)
    };
    let mut times = Vec::new();
    let mut numbers = 0;
    for i in 0..3 {
        let (x, y) = batch();
        let t = Instant::now();
        if let Some(res) = &resident {
            let mut gg = crate::gpu::Graph::with_params(res);
            let f = forward(model.as_ref(), &mut gg, &x, &y);
            gg.backward(f.loss);
            let _ = gg.item(f.loss);
            res.step(&mut opt);
            if i == 0 {
                times.push(t.elapsed().as_secs_f64());
                // A snapshot step copies the whole graph to the host, as this does.
                let g = gg.to_host(|_, _| true, true);
                numbers = train::graph_numbers(model.as_ref(), &g, &f, y.len(), adam);
                continue;
            }
        } else {
            let mut g = Graph::new();
            let f = forward(model.as_ref(), &mut g, &x, &y);
            g.backward(f.loss);
            let grads: Vec<Tensor> = f.params.iter().map(|p| g.grad(*p).clone()).collect();
            opt.step(model.params_mut(), &grads);
            if i == 0 {
                numbers = train::graph_numbers(model.as_ref(), &g, &f, y.len(), adam);
            }
        }
        times.push(t.elapsed().as_secs_f64());
        if times[0] > 10.0 {
            break;
        }
    }
    // An evaluation runs forward passes only (on the GPU when it is on).
    let (x, y) = batch();
    let t = Instant::now();
    let _ = forward_eval(model.as_ref(), &x, &y, false);
    let forward_secs = t.elapsed().as_secs_f64();
    let later = &times[1.min(times.len() - 1)..];
    let step = later.iter().sum::<F>() / later.len() as F;
    println!(
        "{}",
        object(&[
            ("ok", "true".into()),
            ("vocab", v.to_string()),
            ("params", model.params().count().to_string()),
            ("first_step_seconds", format!("{:.6}", times[0])),
            ("step_seconds", format!("{step:.6}")),
            ("forward_seconds", format!("{forward_secs:.6}")),
            ("graph_numbers", numbers.to_string()),
            ("peak_bytes", opt_u64(sysinfo::peak_memory())),
            ("device", string(if cfg.gpu { "gpu" } else { "cpu" })),
            ("gpu_name", gpu_name.map_or("null".into(), string)),
            ("threads", cfg.threads.to_string()),
        ])
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(args: &[&str]) -> Config {
        crate::parse(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap()
    }

    /// The counts from shapes agree with real models and real graphs.
    #[test]
    fn counts_match_real_models() {
        let cases = [
            cfg(&["--model", "bigram", "--examples", "5"]),
            cfg(&["--model", "mlp", "--examples", "3", "--context", "4", "--emb", "5", "--hidden", "7"]),
            cfg(&["--model", "transformer", "--examples", "2", "--context", "6", "--emb", "8", "--heads", "2", "--layers", "2"]),
            cfg(&["--model", "transformer", "--examples", "2", "--context", "5", "--emb", "12", "--heads", "3", "--layers", "1", "--variant", "qk-norm,relu2", "--optimizer", "muon"]),
        ];
        let v = 11;
        for c in &cases {
            let mut rng = Rng::new(3);
            let model: Box<dyn Model> = match c.model {
                ModelKind::Bigram => Box::new(Bigram::new(v)),
                ModelKind::Mlp => Box::new(Mlp::new(v, c.context, c.emb, c.hidden, &mut rng)),
                ModelKind::Transformer => Box::new(Transformer::with_variant(v, c.context, c.emb, c.heads, c.layers, c.variant, &mut rng)),
            };
            assert_eq!(param_count(c, v), model.params().count());
            let (ctx, per) = (model.context(), model.targets_per_example());
            let x: Vec<usize> = (0..c.examples * ctx).map(|i| i % v).collect();
            let y: Vec<usize> = (0..c.examples * per).map(|i| (i * 7) % v).collect();
            let mut g = Graph::new();
            let f = forward(model.as_ref(), &mut g, &x, &y);
            g.backward(f.loss);
            let adam = c.optimizer != Kind::Sgd;
            let p = param_count(c, v);
            let expect = node_numbers(c, v) + logits_numbers(c, v) + y.len() + p * if adam { 5 } else { 1 };
            assert_eq!(expect, train::graph_numbers(model.as_ref(), &g, &f, y.len(), adam));
        }
    }

    #[test]
    fn counts_agree_with_the_training_schedule() {
        for (steps, first, every) in [(1, 10, 500), (10, 10, 3), (11, 10, 3), (5000, 10, 500), (5001, 0, 500), (7, 0, 1), (2294, 10, 1000)] {
            let mut c = cfg(&["--model", "bigram"]);
            (c.steps, c.snap_first, c.snap_every, c.eval_every) = (steps, first, every, every);
            assert_eq!(snapshot_count(&c), (0..steps).filter(|&s| train::is_snapshot_step(s, &c)).count(), "{steps} {first} {every}");
            assert_eq!(eval_count(&c), (0..steps).filter(|&s| s % every == 0 || s + 1 == steps).count());
        }
    }

    #[test]
    fn errors_name_their_field() {
        assert_eq!(field_of("--warmup (100) must be less than --steps (50)"), "warmup");
        assert_eq!(field_of("run name \"a b\" may only use letters"), "run");
        assert_eq!(field_of("--emb 50 must split evenly into --heads 4"), "emb");
        assert_eq!(form("model=mlp&run=a%2Db+c&x"), vec![("model".into(), "mlp".into()), ("run".into(), "a-b c".into()), ("x".into(), String::new())]);
        assert!(to_args(&[("serve".into(), "9".into())]).is_err());
    }
}
