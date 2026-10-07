//! microllm: a from-scratch, fully traced character-level language model.
//!
//!   microllm                        the same as `microllm studio` (so a double-click opens it)
//!   microllm gradcheck              verify every op's gradient against finite differences
//!   microllm train [options]        train a model and write runs/<run>/
//!   microllm bench                  time one transformer-sized training step (Phase 3 sizing)
//!   microllm landscape --run NAME   the loss over the plane of a finished run's path (runs/NAME/landscape.js)
//!                                   [--grid N] [--tokens N] [--threads N]
//!   microllm studio [--port P] [--no-open]  a training page at http://127.0.0.1:P/ (default 8790, or the next free pair)
//!   microllm probe [train options]  time a few training steps of that model (the studio runs this)
//!
//! Train options (defaults depend on --model):
//!   --model bigram|mlp|transformer  --steps N  --examples N  --context K  --emb D  --hidden H
//!   --variant none|relu2,qk-norm,zero-init (transformer changes, any subset)
//!   --wsd FRAC (warm-up–stable–decay, the last FRAC of steps decaying; 0 = cosine)  --muon-lr X
//!   --heads H  --layers L  --optimizer sgd|adamw|muon  --lr X  --lr-min X  --warmup N  --wd X  --seed S
//!   --eval-every N  --eval-batches N  --sample-len N  --snap-every N  --snap-first N
//!   --detail auto|full|summary  --serve PORT  --throttle-ms MS  --run NAME  --data PATH  --val-frac X  --threads N  --device cpu|gpu
//!   --vocab N (BPE tokenizer with N tokens; 0 = characters)  --chars N (train on about N characters: sets --steps)

mod autograd;
mod bench;
mod bpe;
mod cuda;
mod data;
mod gpu;
mod gradcheck;
mod json;
mod landscape;
mod metrics;
mod model;
mod optim;
mod par;
mod params;
mod rng;
mod score;
mod serve;
mod sha256;
mod studio;
mod sysinfo;
mod tensor;
mod trace;
mod train;

use std::collections::HashMap;

use optim::Kind;
use trace::Detail;
use train::{Config, ModelKind};

const USAGE: &str = "usage: microllm gradcheck | microllm bench | microllm studio [--port P] [--no-open] | microllm landscape --run NAME | microllm train [--model bigram|mlp|transformer] [--steps N] [--examples N] [--context K] \
[--emb D] [--hidden H] [--heads H] [--layers L] [--variant LIST] [--optimizer sgd|adamw|muon] [--muon-lr X] [--wsd FRAC] [--val-frac X] [--lr X] [--lr-min X] [--warmup N] [--wd X] [--seed S] [--eval-every N] [--eval-batches N] \
[--sample-len N] [--snap-every N] [--snap-first N] [--detail auto|full|summary] [--serve [PORT]] [--throttle-ms MS] [--run NAME] [--data PATH] [--threads N] [--device cpu|gpu] [--vocab N] [--chars N]";

fn parse(args: &[String]) -> Result<Config, String> {
    let mut opts: HashMap<String, String> = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].strip_prefix("--").ok_or(format!("unexpected argument: {}\n{USAGE}", args[i]))?;
        // --serve may be given without a port.
        let value = match args.get(i + 1) {
            Some(v) if !v.starts_with("--") => {
                i += 1;
                v.clone()
            }
            _ if key == "serve" => "8787".to_string(),
            _ => return Err(format!("missing value for --{key}")),
        };
        if opts.insert(key.to_string(), value).is_some() {
            return Err(format!("--{key} given twice"));
        }
        i += 1;
    }
    fn get<T: std::str::FromStr>(opts: &mut HashMap<String, String>, key: &str, default: T) -> Result<T, String> {
        match opts.remove(key) {
            Some(v) => v.parse().map_err(|_| format!("bad value for --{key}: {v}")),
            None => Ok(default),
        }
    }
    let model = match get(&mut opts, "model", "bigram".to_string())?.as_str() {
        "bigram" => ModelKind::Bigram,
        "mlp" => ModelKind::Mlp,
        "transformer" => ModelKind::Transformer,
        other => return Err(format!("unknown model {other}; use bigram, mlp or transformer")),
    };
    let mlp = model == ModelKind::Mlp;
    let tf = model == ModelKind::Transformer;
    // Defaults per model: (run name, steps, examples, context, emb, snap_every)
    let (run, steps, examples, context, emb, snap_every) = match model {
        ModelKind::Bigram => ("bigram", 3000, 64, 8, 16, 100),
        ModelKind::Mlp => ("mlp", 10000, 32, 8, 16, 500),
        ModelKind::Transformer => ("transformer", 5000, 32, 32, 48, 500),
    };
    let optimizer = match get(&mut opts, "optimizer", if mlp || tf { "adamw" } else { "sgd" }.to_string())?.as_str() {
        "sgd" => Kind::Sgd,
        "adamw" => Kind::AdamW,
        "muon" => Kind::Muon,
        other => return Err(format!("unknown optimizer {other}; use sgd, adamw or muon")),
    };
    let detail = match get(&mut opts, "detail", "auto".to_string())?.as_str() {
        "auto" => None,
        "full" => Some(Detail::Full),
        "summary" => Some(Detail::Summary),
        other => return Err(format!("unknown detail {other}; use auto, full or summary")),
    };
    let serve = match opts.remove("serve") {
        Some(p) => Some(p.parse::<u16>().map_err(|_| format!("bad port for --serve: {p}"))?),
        None => None,
    };
    let lr = get(&mut opts, "lr", if optimizer == Kind::Sgd { 10.0 } else { 3e-3 })?;
    let cfg = Config {
        data: get(&mut opts, "data", "data/tinyshakespeare.txt".to_string())?,
        val_frac: get(&mut opts, "val-frac", 0.1)?,
        run: get(&mut opts, "run", run.to_string())?,
        model,
        steps: get(&mut opts, "steps", steps)?,
        examples: get(&mut opts, "examples", examples)?,
        context: get(&mut opts, "context", context)?,
        emb: get(&mut opts, "emb", emb)?,
        hidden: get(&mut opts, "hidden", 128)?,
        heads: get(&mut opts, "heads", 4)?,
        layers: get(&mut opts, "layers", 2)?,
        variant: model::Variant::parse(&get(&mut opts, "variant", "none".to_string())?)?,
        optimizer,
        lr,
        // The transformer warms up and then decays to a tenth; the others keep lr constant.
        lr_min: get(&mut opts, "lr-min", if tf { lr / 10.0 } else { lr })?,
        warmup: get(&mut opts, "warmup", if tf { 100 } else { 0 })?,
        wsd: get(&mut opts, "wsd", 0.0)?,
        muon_lr: get(&mut opts, "muon-lr", lr)?,
        weight_decay: get(&mut opts, "wd", if tf { 0.1 } else { 0.01 })?,
        seed: get(&mut opts, "seed", 42)?,
        eval_every: get(&mut opts, "eval-every", 100)?,
        eval_batches: get(&mut opts, "eval-batches", 20)?,
        sample_len: get(&mut opts, "sample-len", 200)?,
        snap_every: get(&mut opts, "snap-every", snap_every)?,
        snap_first: get(&mut opts, "snap-first", 10)?,
        detail,
        serve,
        throttle_ms: get(&mut opts, "throttle-ms", 0)?,
        threads: get(&mut opts, "threads", par::available())?,
        gpu: match get(&mut opts, "device", "cpu".to_string())?.as_str() {
            "cpu" => false,
            "gpu" => true,
            other => return Err(format!("unknown device {other}; use cpu or gpu")),
        },
        vocab: get(&mut opts, "vocab", 0)?,
        chars: get(&mut opts, "chars", 0)?,
    };
    if let Some(k) = opts.keys().next() {
        return Err(format!("unknown option --{k}\n{USAGE}"));
    }
    if cfg.run.is_empty() || !cfg.run.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
        return Err(format!("run name {:?} may only use letters, digits, '_' and '-'", cfg.run));
    }
    for (name, v) in [("steps", cfg.steps), ("examples", cfg.examples), ("eval-every", cfg.eval_every), ("eval-batches", cfg.eval_batches), ("snap-every", cfg.snap_every), ("threads", cfg.threads)] {
        if v == 0 {
            return Err(format!("--{name} must be at least 1"));
        }
    }
    if !(cfg.lr.is_finite() && cfg.lr > 0.0) {
        return Err(format!("--lr must be a finite number above 0, got {}", cfg.lr));
    }
    if cfg.warmup >= cfg.steps {
        return Err(format!("--warmup ({}) must be less than --steps ({})", cfg.warmup, cfg.steps));
    }
    if !(cfg.lr_min.is_finite() && cfg.lr_min > 0.0 && cfg.lr_min <= cfg.lr) {
        return Err(format!("--lr-min must be above 0 and at most --lr ({}), got {}", cfg.lr, cfg.lr_min));
    }
    if !(cfg.val_frac > 0.0 && cfg.val_frac < 0.5) {
        return Err(format!("--val-frac must be above 0 and below 0.5, got {}", cfg.val_frac));
    }
    if !(cfg.wsd.is_finite() && (0.0..=1.0).contains(&cfg.wsd)) {
        return Err(format!("--wsd must be between 0 (cosine) and 1, got {}", cfg.wsd));
    }
    if cfg.optimizer == Kind::Muon && !tf {
        return Err("--optimizer muon is for the transformer's block matrices".into());
    }
    if !(cfg.weight_decay.is_finite() && cfg.weight_decay >= 0.0) {
        return Err(format!("--wd must be a finite number at least 0, got {}", cfg.weight_decay));
    }
    if mlp && (cfg.context == 0 || cfg.emb == 0 || cfg.hidden == 0) {
        return Err("--context, --emb and --hidden must be at least 1".into());
    }
    if tf {
        if cfg.context == 0 || cfg.emb == 0 || cfg.heads == 0 || cfg.layers == 0 {
            return Err("--context, --emb, --heads and --layers must be at least 1".into());
        }
        if cfg.emb % cfg.heads != 0 {
            return Err(format!("--emb {} must split evenly into --heads {}", cfg.emb, cfg.heads));
        }
    }
    Ok(cfg)
}

fn parse_landscape(args: &[String]) -> Result<landscape::Options, String> {
    let usage = "usage: microllm landscape --run NAME [--grid N] [--tokens N] [--threads N]";
    let mut opts: HashMap<String, String> = HashMap::new();
    for pair in args.chunks(2) {
        let key = pair[0].strip_prefix("--").ok_or(format!("unexpected argument: {}
{usage}", pair[0]))?;
        let value = pair.get(1).ok_or(format!("missing value for --{key}"))?;
        opts.insert(key.to_string(), value.clone());
    }
    let num = |opts: &mut HashMap<String, String>, key: &str, default: usize| -> Result<usize, String> {
        match opts.remove(key) {
            Some(v) => v.parse().ok().filter(|&n| n > 0).ok_or(format!("bad value for --{key}: {v}")),
            None => Ok(default),
        }
    };
    let run = opts.remove("run").ok_or(format!("--run is required
{usage}"))?;
    if run.is_empty() || !run.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
        return Err(format!("run name {run:?} may only use letters, digits, '_' and '-'"));
    }
    let grid = num(&mut opts, "grid", 21)?;
    let tokens = num(&mut opts, "tokens", 4096)?;
    par::set_threads(num(&mut opts, "threads", par::available())?);
    if let Some(k) = opts.keys().next() {
        return Err(format!("unknown option --{k}
{usage}"));
    }
    if grid < 2 {
        return Err("--grid must be at least 2".into());
    }
    Ok(landscape::Options { run, grid, tokens })
}

fn parse_studio(args: &[String]) -> Result<studio::Options, String> {
    let mut o = studio::Options { port: 8790, open: true, find_port: true };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--no-open" => o.open = false,
            "--port" => {
                let p = args.get(i + 1).ok_or("missing value for --port")?;
                o.port = p.parse().ok().filter(|&p: &u16| p > 0 && p < u16::MAX).ok_or(format!("bad port for --port: {p}"))?;
                o.find_port = false;
                i += 1;
            }
            other => return Err(format!("unexpected argument: {other}
usage: microllm studio [--port P] [--no-open]")),
        }
        i += 1;
    }
    Ok(o)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        None => parse_studio(&[]).and_then(|o| studio::run(&o)),
        Some("gradcheck") => {
            let reports = gradcheck::run_all();
            for l in train::gradcheck_lines(&reports) {
                println!("{l}");
            }
            if reports.iter().all(|r| r.passed()) { Ok(()) } else { Err("gradient check failed".into()) }
        }
        Some("train") => parse(&args[1..]).and_then(|cfg| train::train(&cfg)),
        Some("bench") => {
            let size = |n, t, d, heads, layers| bench::Size { n, t, d, heads, layers };
            bench::run(&[size(32, 32, 48, 4, 2), size(32, 32, 64, 4, 2), size(32, 32, 64, 4, 4), size(32, 64, 64, 4, 4), size(64, 64, 96, 6, 4)])
        }
        Some("landscape") => parse_landscape(&args[1..]).and_then(|o| landscape::run(&o)),
        Some("studio") => parse_studio(&args[1..]).and_then(|o| studio::run(&o)),
        Some("probe") => studio::probe(&args[1..]),
        Some("profile") => {
            // microllm profile [threads]: the scaled transformer's real training step, by op
            let threads = args.get(1).and_then(|t| t.parse().ok()).unwrap_or_else(par::available);
            par::set_threads(threads);
            bench::profile(32, bench::Size { n: 32, t: 64, d: 128, heads: 4, layers: 4 }, 3)
        }
        _ => Err(USAGE.into()),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        if args.is_empty() {
            // Started by a double-click: keep the window open long enough to read why.
            eprintln!("press Enter to close");
            let _ = std::io::stdin().read_line(&mut String::new());
        }
        std::process::exit(1);
    }
}
