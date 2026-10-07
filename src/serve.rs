//! A tiny read-only HTTP server for watching training live (std only).
//!
//! Binds 127.0.0.1 only. Serves files under viewer/ and runs/, plus GET /live,
//! which returns everything recorded since the client's cursor. One thread per
//! connection; the training loop never waits on a client.

use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::metrics::Metrics;
use crate::trace::{ints, object, string};

pub struct LiveState {
    pub metrics: Metrics,
    /// Snapshot steps whose files are completely written (published after the rename).
    pub snapshots: Vec<usize>,
    pub status: &'static str,
    pub last_completed_step: Option<usize>,
}

pub type Shared = Arc<Mutex<LiveState>>;

pub fn lock(s: &Shared) -> std::sync::MutexGuard<'_, LiveState> {
    s.lock().unwrap_or_else(|e| e.into_inner())
}

pub(crate) const MAX_HEADER: usize = 16 * 1024;

/// Identifies this training process. It is written to meta.js and returned by /live, so a
/// viewer can tell that both came from the same run; when they differ the run was restarted
/// (possibly a different run under the same name) and the viewer starts over, even if the
/// new run has already caught up with its cursor.
pub fn instance() -> &'static str {
    static ID: OnceLock<String> = OnceLock::new();
    ID.get_or_init(|| {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        format!("{}-{nanos}", std::process::id())
    })
}

pub fn start(port: u16, state: Shared, run: &str) -> Result<(), String> {
    let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|e| format!("binding 127.0.0.1:{port}: {e}"))?;
    let root = |d: &str| fs::canonicalize(d).map(|p| (d.to_string(), p)).map_err(|e| format!("{d}/: {e}"));
    let roots = Arc::new(vec![root("viewer")?, root("runs")?]);
    let run = run.to_string();
    thread::spawn(move || {
        for conn in listener.incoming().flatten() {
            let (state, roots, run) = (state.clone(), roots.clone(), run.clone());
            thread::spawn(move || {
                let _ = handle(conn, &state, &roots, &run, port);
            });
        }
    });
    Ok(())
}

fn handle(mut s: TcpStream, state: &Shared, roots: &[(String, PathBuf)], run: &str, port: u16) -> std::io::Result<()> {
    s.set_read_timeout(Some(Duration::from_secs(5)))?;
    s.set_write_timeout(Some(Duration::from_secs(10)))?;
    // Read only the request head; we never wait for the client to close.
    let mut buf = Vec::new();
    let mut chunk = [0u8; 2048];
    let head_end = loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i;
        }
        let n = s.read(&mut chunk)?;
        if n == 0 {
            return Ok(());
        }
        buf.extend_from_slice(&chunk[..n]);
        if head_too_long(&buf) {
            return respond(&mut s, 431, "text/plain", b"request header too large");
        }
    };
    let Ok(head) = std::str::from_utf8(&buf[..head_end]) else {
        return respond(&mut s, 400, "text/plain", b"bad request");
    };
    if !host_ok(head, port) {
        return respond(&mut s, 403, "text/plain", b"wrong Host");
    }
    let mut parts = head.lines().next().unwrap_or("").split(' ');
    let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    if method != "GET" {
        return respond(&mut s, 405, "text/plain", b"only GET");
    }
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    match path {
        "/" => {
            let loc = format!("/viewer/index.html?run={run}&live=1");
            let msg = format!("HTTP/1.1 302 Found\r\nLocation: {loc}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            s.write_all(msg.as_bytes())
        }
        "/live" => match live_json(state, query, run) {
            Some(body) => respond(&mut s, 200, "application/json", body.as_bytes()),
            None => respond(&mut s, 400, "text/plain", b"cursor must be three whole numbers: steps,evals,snapshots"),
        },
        _ => match resolve(path, roots) {
            Some(file) => match fs::read(&file) {
                Ok(bytes) => respond(&mut s, 200, content_type(&file), &bytes),
                Err(_) => respond(&mut s, 404, "text/plain", b"not found"),
            },
            None => respond(&mut s, 404, "text/plain", b"not found"),
        },
    }
}

/// Parse `cursor=steps,evals,snapshots`: exactly three whole numbers, or None.
pub fn parse_cursor(query: &str) -> Option<(usize, usize, usize)> {
    let c = query.split('&').find_map(|kv| kv.strip_prefix("cursor="))?;
    let v: Vec<usize> = c.split(',').map(|x| x.parse().ok()).collect::<Option<_>>()?;
    match v[..] {
        [a, b, c] => Some((a, b, c)),
        _ => None,
    }
}

/// Everything after the client's cursor, plus the next cursor. None if the cursor is malformed.
fn live_json(state: &Shared, query: &str, run: &str) -> Option<String> {
    let (cs, ce, cn) = parse_cursor(query)?;
    // Copy the delta under the lock; encode after releasing it.
    let (delta, snaps, status, last, fs, fe, fn_) = {
        let st = lock(state);
        let (fs, fe, fn_) = (cs.min(st.metrics.steps()), ce.min(st.metrics.evals()), cn.min(st.snapshots.len()));
        (st.metrics.slice(fs, fe), st.snapshots[fn_..].to_vec(), st.status, st.last_completed_step, fs, fe, fn_)
    };
    let next = format!("{},{},{}", fs + delta.steps(), fe + delta.evals(), fn_ + snaps.len());
    Some(object(&[
        ("instance", string(instance())),
        ("run", string(run)),
        ("status", string(status)),
        ("last_completed_step", last.map_or("null".to_string(), |s| s.to_string())),
        ("cursor", string(&next)),
        ("snapshots", ints(&snaps)),
        ("metrics", delta.json(fs, fe)),
    ]))
}

pub(crate) fn respond(s: &mut TcpStream, code: u16, ctype: &str, body: &[u8]) -> std::io::Result<()> {
    let reason = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        403 => "Forbidden",
        405 => "Method Not Allowed",
        409 => "Conflict",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        _ => "Error",
    };
    let head = format!(
        "HTTP/1.1 {code} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n\
         Cross-Origin-Resource-Policy: same-origin\r\nX-Frame-Options: SAMEORIGIN\r\n\
         Content-Security-Policy: frame-ancestors 'self'\r\nConnection: close\r\n\r\n",
        body.len()
    );
    s.write_all(head.as_bytes())?;
    s.write_all(body)
}

pub(crate) fn content_type(p: &std::path::Path) -> &'static str {
    match p.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("json") => "application/json",
        Some("css") => "text/css; charset=utf-8",
        Some("log") | Some("txt") => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

pub(crate) fn percent_decode(s: &str) -> Option<String> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' {
            let hex = std::str::from_utf8(b.get(i + 1..i + 3)?).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// Map a URL path to a file inside one of the roots, or None.
/// Decode once; reject NUL, backslashes, drive/stream colons, empty, `.` and `..`
/// components and in-progress `.tmp` files; then canonicalize and require the result to sit under that root.
pub fn resolve(raw: &str, roots: &[(String, PathBuf)]) -> Option<PathBuf> {
    let decoded = percent_decode(raw)?;
    let rest = decoded.strip_prefix('/')?;
    if decoded.contains('\0') || decoded.contains('\\') {
        return None;
    }
    let parts: Vec<&str> = rest.split('/').collect();
    if parts.iter().any(|p| p.is_empty() || *p == "." || *p == ".." || p.contains(':') || p.to_ascii_lowercase().ends_with(".tmp")) {
        return None;
    }
    let (first, tail) = parts.split_first()?;
    let (_, root) = roots.iter().find(|(name, _)| name == first)?;
    let mut p = root.clone();
    for c in tail {
        p.push(c);
    }
    let canon = fs::canonicalize(&p).ok()?;
    (canon.starts_with(root) && canon.is_file()).then_some(canon)
}

/// True once the head has grown past MAX_HEADER without ending: a head that ends within the
/// limit passes even when body bytes arrived in the same read.
pub(crate) fn head_too_long(buf: &[u8]) -> bool {
    match buf.windows(4).position(|w| w == b"\r\n\r\n") {
        Some(i) => i + 4 > MAX_HEADER,
        None => buf.len() > MAX_HEADER,
    }
}

/// The Host header must name this server, so other web pages cannot reach it through DNS
/// rebinding (a hostile name that resolves to 127.0.0.1). Exactly one Host header is allowed.
fn host_ok(head: &str, port: u16) -> bool {
    let here = [format!("127.0.0.1:{port}"), format!("localhost:{port}")];
    let hosts: Vec<&str> = head
        .lines()
        .skip(1)
        .filter_map(|l| l.split_once(':'))
        .filter(|(k, _)| k.trim().eq_ignore_ascii_case("host"))
        .map(|(_, v)| v.trim())
        .collect();
    matches!(hosts[..], [h] if here.iter().any(|x| x.eq_ignore_ascii_case(h)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roots() -> (PathBuf, Vec<(String, PathBuf)>) {
        let base = std::env::temp_dir().join(format!("microllm-serve-test-{}", std::process::id()));
        fs::create_dir_all(base.join("viewer")).unwrap();
        fs::create_dir_all(base.join("runs")).unwrap();
        fs::write(base.join("viewer").join("index.html"), "ok").unwrap();
        fs::write(base.join("secret.txt"), "no").unwrap();
        let canon = |d: &str| (d.to_string(), fs::canonicalize(base.join(d)).unwrap());
        let r = vec![canon("viewer"), canon("runs")];
        (base, r)
    }

    #[test]
    fn serves_files_inside_roots_only() {
        let (base, r) = roots();
        assert!(resolve("/viewer/index.html", &r).is_some());
        assert!(resolve("/viewer/ind%65x.html", &r).is_some(), "plain percent-encoding is fine");
        for bad in [
            "/viewer/../secret.txt",
            "/viewer/%2e%2e/secret.txt",
            "/viewer/%2E%2E%2Fsecret.txt",
            "/viewer/..%2fsecret.txt",
            "/viewer/..%5csecret.txt",
            "/viewer\\..\\secret.txt",
            "/viewer/%00",
            "/viewer/%zz",
            "/viewer/%2",
            "/C:/Windows/win.ini",
            "/viewer/C:%5cWindows",
            "/secret.txt",
            "/viewer//index.html",
            "/viewer/./index.html",
            "/viewer",
            "/runs/",
            "viewer/index.html",
            "/viewer/index.html.tmp",
            "/viewer/x.TMP",
        ] {
            assert!(resolve(bad, &r).is_none(), "{bad} should be rejected");
        }
        fs::remove_dir_all(base).ok();
    }

    #[test]
    fn host_must_name_this_server() {
        let req = |h: &str| format!("GET / HTTP/1.1
{h}
Accept: */*");
        assert!(host_ok(&req("Host: 127.0.0.1:8791"), 8791));
        assert!(host_ok(&req("host: localhost:8791"), 8791));
        assert!(!host_ok(&req("Host: evil.example:8791"), 8791));
        assert!(!host_ok(&req("Host: 127.0.0.1:8792"), 8791));
        assert!(!host_ok(&req("Host: 127.0.0.1.:8791"), 8791));
        assert!(!host_ok(&req("Host: [::1]:8791"), 8791));
        assert!(!host_ok(&req("Host: 127.0.0.1:8791\r\nHost: evil.example:8791"), 8791));
        assert!(!host_ok(&req("Host: 127.0.0.1:8791\r\nHost: 127.0.0.1:8791"), 8791));
        assert!(!host_ok("GET / HTTP/1.1
Accept: */*", 8791));
    }

    #[test]
    fn head_limit_is_exact() {
        let head = |n: usize| [vec![b'a'; n - 4], b"\r\n\r\n".to_vec()].concat();
        assert!(!head_too_long(&head(MAX_HEADER)));
        assert!(head_too_long(&head(MAX_HEADER + 1)));
        // Body bytes read together with the head do not count against it.
        assert!(!head_too_long(&[head(MAX_HEADER), vec![b'b'; 4096]].concat()));
        assert!(!head_too_long(&vec![b'a'; MAX_HEADER]));
        assert!(head_too_long(&vec![b'a'; MAX_HEADER + 1]));
    }

    #[test]
    fn cursor_must_be_three_whole_numbers() {
        assert_eq!(parse_cursor("cursor=0,0,0"), Some((0, 0, 0)));
        assert_eq!(parse_cursor("x=1&cursor=12,3,4"), Some((12, 3, 4)));
        for bad in ["", "cursor=", "cursor=1,2", "cursor=1,2,3,4", "cursor=1,x,3", "cursor=1,,3", "cursor=-1,2,3", "cursor=1.5,2,3", "other=1,2,3"] {
            assert_eq!(parse_cursor(bad), None, "{bad} should be rejected");
        }
    }
}
