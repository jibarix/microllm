# Security

microllm runs on your own computer, starts processes and writes files, so this page says
exactly what it does and how to report a problem.

## Reporting a vulnerability

Please report privately: on this repository's **Security** tab, choose **Report a
vulnerability**. Do not open a public issue for a security problem. You should get a reply
within a week. Fixes go into the latest version only.

## What microllm does on your machine

- **Local web servers.** The studio listens on `127.0.0.1` (port 8790, or the next free even
  port), and a training run's live view on the port after it. Nothing listens on other network
  interfaces, and nothing is sent anywhere: there is no telemetry and no download.
- **Processes.** The studio starts microllm itself as a child process to train, at low
  priority (`nice` on macOS and Linux, below-normal priority on Windows). It opens your browser
  at its own address. On macOS it reads memory figures with `sysctl` and `vm_stat`, and on
  macOS and Linux free disk space with `df`.
- **Files.** It writes only under `runs/<name>/`, and refuses a run folder that is a link
  (symlink or junction) or that resolves outside `runs/`. In the studio, text comes only from
  `.txt` files in `data/`, and starting a run with an existing name replaces that run's files
  only after you tick "replace". On the command line, `microllm train --data PATH` reads
  whatever file you name, and an existing run of the same name is replaced without asking.
- **Run records.** A run's `train.log` and `meta.js` record the machine it trained on: the
  processor thread count, and for GPU runs the GPU model and CUDA version. Check them
  before you publish a run of your own.
- **Build.** `build.rs` runs `rustc -V`, and `nvcc` if the CUDA toolkit is installed, to
  compile the GPU kernels in `src/kernels.cu`. On Windows, if `cl.exe` is not on `PATH`, it
  looks through the Visual Studio folders under `Program Files` for one to give `nvcc`.
- **Load.** Training can keep every processor core, and the GPU if chosen, busy for as long as
  the run lasts. The page estimates memory, disk and time before it starts.

## How the servers are protected

- Both servers accept a request only when its `Host` header names `127.0.0.1` or `localhost`
  with their own port, which blocks DNS-rebinding attacks from other web pages.
- The studio answers its API (`/api/...`, reads and changes alike) only from its own page: an
  `Origin`, when sent, must be the studio itself, and the request must carry the
  `X-Microllm-Studio` header, which a page from another site cannot send without the browser
  asking this server first; the server never grants that. POST is accepted only on the API.
- Every response carries `Cross-Origin-Resource-Policy: same-origin`,
  `X-Frame-Options: SAMEORIGIN` and `Content-Security-Policy: frame-ancestors 'self'`, so
  other sites can neither load the files nor frame the pages, and `Cache-Control: no-store`
  with `X-Content-Type-Options: nosniff`.
- The studio refuses a request that repeats any header, and the live view one with two `Host`s.
- Files are served only from inside `viewer/` and `runs/`; the path is decoded once,
  `..`, backslashes, drive letters and colons are rejected, and the resolved path must still
  sit inside that folder.
- A run's name may use only letters, digits, `_` and `-`, and its text must be a `.txt` file
  already in `data/`. Settings are passed to the child process as separate arguments, never
  through a shell.
- Request bodies are capped at 64 kB and request heads at exactly 16 kB.

## Supply chain

microllm uses only the Rust standard library: `Cargo.lock` lists no third-party crates. The
repository runs no GitHub Actions. `data/prepare_tinystories.py` uses only Python's standard
library; run it with `python -I` so it cannot import files from the folder it reads.
