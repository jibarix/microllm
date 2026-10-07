//! Read-only facts about this machine for the studio's recommendations (std only).
//!
//! Nothing here changes the system: it reads memory and disk counters, and on macOS
//! runs `sysctl`, `vm_stat` and `df`. Every answer may be None when the platform does
//! not say; callers then fall back to conservative choices.

use std::path::Path;

pub struct System {
    pub os: &'static str,
    pub arch: &'static str,
    pub cores: usize,
    pub ram_total: Option<u64>,
    /// Memory that can be used without swapping out other programs.
    pub ram_available: Option<u64>,
    /// Free space on the drive holding `path` (the project, where runs/ is written).
    pub disk_free: Option<u64>,
    /// Whether an NVIDIA driver library is present. Opening the GPU is left to the probe.
    pub nvidia_driver: bool,
}

pub fn system(path: &Path) -> System {
    let (ram_total, ram_available) = memory();
    System {
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        cores: std::thread::available_parallelism().map_or(1, |n| n.get()),
        ram_total,
        ram_available,
        disk_free: disk_free(path),
        nvidia_driver: nvidia_driver(),
    }
}

fn nvidia_driver() -> bool {
    let candidates: &[&str] = if cfg!(windows) {
        &["C:\\Windows\\System32\\nvcuda.dll"]
    } else if cfg!(target_os = "linux") {
        &["/usr/lib/x86_64-linux-gnu/libcuda.so.1", "/usr/lib64/libcuda.so.1", "/usr/lib/libcuda.so.1", "/usr/lib/wsl/lib/libcuda.so.1"]
    } else {
        &[]
    };
    candidates.iter().any(|p| Path::new(p).exists())
}

#[cfg(windows)]
mod win {
    #[repr(C)]
    pub struct MemoryStatusEx {
        pub length: u32,
        pub memory_load: u32,
        pub total_phys: u64,
        pub avail_phys: u64,
        pub total_page_file: u64,
        pub avail_page_file: u64,
        pub total_virtual: u64,
        pub avail_virtual: u64,
        pub avail_extended_virtual: u64,
    }

    #[repr(C)]
    pub struct ProcessMemoryCounters {
        pub cb: u32,
        pub page_fault_count: u32,
        pub peak_working_set_size: usize,
        pub working_set_size: usize,
        pub quota_peak_paged_pool_usage: usize,
        pub quota_paged_pool_usage: usize,
        pub quota_peak_non_paged_pool_usage: usize,
        pub quota_non_paged_pool_usage: usize,
        pub pagefile_usage: usize,
        pub peak_pagefile_usage: usize,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
        pub fn GetDiskFreeSpaceExW(dir: *const u16, free_to_caller: *mut u64, total: *mut u64, total_free: *mut u64) -> i32;
        pub fn GetCurrentProcess() -> isize;
        pub fn K32GetProcessMemoryInfo(process: isize, counters: *mut ProcessMemoryCounters, cb: u32) -> i32;
    }
}

#[cfg(windows)]
fn memory() -> (Option<u64>, Option<u64>) {
    let mut m = win::MemoryStatusEx {
        length: std::mem::size_of::<win::MemoryStatusEx>() as u32,
        memory_load: 0,
        total_phys: 0,
        avail_phys: 0,
        total_page_file: 0,
        avail_page_file: 0,
        total_virtual: 0,
        avail_virtual: 0,
        avail_extended_virtual: 0,
    };
    // SAFETY: m is a correctly sized MEMORYSTATUSEX with its length set, as the call requires.
    if unsafe { win::GlobalMemoryStatusEx(&mut m) } != 0 {
        (Some(m.total_phys), Some(m.avail_phys))
    } else {
        (None, None)
    }
}

#[cfg(windows)]
fn disk_free(path: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    let abs = std::fs::canonicalize(path).ok()?;
    let wide: Vec<u16> = abs.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    let (mut free, mut total, mut total_free) = (0u64, 0u64, 0u64);
    // SAFETY: wide is NUL-terminated and the three out-pointers are valid u64s.
    let ok = unsafe { win::GetDiskFreeSpaceExW(wide.as_ptr(), &mut free, &mut total, &mut total_free) };
    (ok != 0).then_some(free)
}

/// The most memory this process has held at once, in bytes.
#[cfg(windows)]
pub fn peak_memory() -> Option<u64> {
    let mut c: win::ProcessMemoryCounters = unsafe { std::mem::zeroed() };
    c.cb = std::mem::size_of::<win::ProcessMemoryCounters>() as u32;
    // SAFETY: c is a correctly sized PROCESS_MEMORY_COUNTERS; the handle is the pseudo-handle for this process.
    let ok = unsafe { win::K32GetProcessMemoryInfo(win::GetCurrentProcess(), &mut c, c.cb) };
    (ok != 0).then_some(c.peak_working_set_size as u64)
}

#[cfg(target_os = "linux")]
fn memory() -> (Option<u64>, Option<u64>) {
    let text = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let field = |name: &str| text.lines().find_map(|l| l.strip_prefix(name)).and_then(|r| r.trim().trim_end_matches("kB").trim().parse::<u64>().ok()).map(|kb| kb * 1024);
    (field("MemTotal:"), field("MemAvailable:"))
}

#[cfg(target_os = "linux")]
pub fn peak_memory() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    text.lines().find_map(|l| l.strip_prefix("VmHWM:")).and_then(|r| r.trim().trim_end_matches("kB").trim().parse::<u64>().ok()).map(|kb| kb * 1024)
}

#[cfg(target_os = "macos")]
fn memory() -> (Option<u64>, Option<u64>) {
    let run = |cmd: &str, args: &[&str]| std::process::Command::new(cmd).args(args).output().ok().map(|o| String::from_utf8_lossy(&o.stdout).into_owned());
    let total = run("sysctl", &["-n", "hw.memsize"]).and_then(|s| s.trim().parse().ok());
    // vm_stat: "Mach Virtual Memory Statistics: (page size of 16384 bytes)", then "Pages free: 123." lines.
    let available = run("vm_stat", &[]).and_then(|s| {
        let page: u64 = s.split("page size of ").nth(1)?.split_whitespace().next()?.parse().ok()?;
        let pages = |name: &str| s.lines().find_map(|l| l.strip_prefix(name)).and_then(|r| r.trim().trim_end_matches('.').parse::<u64>().ok()).unwrap_or(0);
        Some((pages("Pages free:") + pages("Pages inactive:") + pages("Pages speculative:")) * page)
    });
    (total, available)
}

#[cfg(target_os = "macos")]
pub fn peak_memory() -> Option<u64> {
    // struct rusage: two timevals (16 bytes each), then ru_maxrss, in bytes on macOS.
    unsafe extern "C" {
        fn getrusage(who: i32, usage: *mut [i64; 18]) -> i32;
    }
    let mut r = [0i64; 18];
    // SAFETY: r is larger than struct rusage (144 bytes); RUSAGE_SELF = 0.
    (unsafe { getrusage(0, &mut r) } == 0).then_some(r[4] as u64)
}

#[cfg(unix)]
fn disk_free(path: &Path) -> Option<u64> {
    // POSIX df: one header line, then "filesystem 1024-blocks used available capacity mount".
    let out = std::process::Command::new("df").arg("-Pk").arg(path).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let kb: u64 = text.lines().nth(1)?.split_whitespace().nth(3)?.parse().ok()?;
    Some(kb * 1024)
}

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
fn memory() -> (Option<u64>, Option<u64>) {
    (None, None)
}

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
pub fn peak_memory() -> Option<u64> {
    None
}

#[cfg(not(any(windows, unix)))]
fn disk_free(_: &Path) -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_this_machine() {
        let s = system(Path::new("."));
        assert!(s.cores >= 1);
        if cfg!(any(windows, target_os = "linux")) {
            let (total, avail) = (s.ram_total.unwrap(), s.ram_available.unwrap());
            assert!(total > 0 && avail <= total);
            assert!(s.disk_free.is_some());
            assert!(peak_memory().unwrap() > 0);
        }
    }
}
