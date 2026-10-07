//! Records the compiler version and target so every training log can say exactly
//! which toolchain built the binary that produced it.
//!
//! Also compiles src/kernels.cu to PTX with nvcc (the CUDA toolkit's compiler) when it is
//! installed. The PTX is embedded in the binary and compiled for the GPU by the driver at
//! run time. Without nvcc the PTX is empty and `--device gpu` reports that it is unavailable.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = Command::new(&rustc)
        .arg("-V")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=MICROLLM_RUSTC={version}");
    println!("cargo:rustc-env=MICROLLM_TARGET={}", std::env::var("TARGET").unwrap_or_else(|_| "unknown".into()));
    println!("cargo:rerun-if-changed=build.rs");

    println!("cargo:rerun-if-changed=src/kernels.cu");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("kernels.ptx");
    let (ptx, how) = match compile_ptx(&out) {
        Ok(how) => (std::fs::read_to_string(&out).unwrap_or_default(), how),
        Err(e) => {
            println!("cargo:warning=GPU kernels not built ({e}); --device gpu will be unavailable");
            (String::new(), format!("not built: {e}"))
        }
    };
    std::fs::write(&out, ptx).unwrap();
    println!("cargo:rustc-env=MICROLLM_PTX_BUILD={how}");
}

fn compile_ptx(out: &Path) -> Result<String, String> {
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".into());
    let version = Command::new(&nvcc).arg("--version").output().map_err(|_| "nvcc not found".to_string())?;
    let version = String::from_utf8_lossy(&version.stdout).lines().filter(|l| l.contains("release")).collect::<String>();
    let mut cmd = Command::new(&nvcc);
    cmd.args(["-ptx", "-arch=compute_70", "-o"]).arg(out).arg("src/kernels.cu");
    // On Windows nvcc needs MSVC's cl.exe as host compiler; find it if it is not on PATH.
    if cfg!(windows) && Command::new("cl.exe").output().is_err() {
        cmd.arg("-ccbin").arg(find_cl().ok_or("cl.exe (Visual Studio C++ tools) not found")?);
    }
    let o = cmd.output().map_err(|e| format!("running nvcc: {e}"))?;
    if !o.status.success() {
        return Err(format!("nvcc failed: {}", String::from_utf8_lossy(&o.stderr).trim()));
    }
    Ok(version.trim().to_string())
}

fn find_cl() -> Option<PathBuf> {
    let mut found = Vec::new();
    for root in ["C:/Program Files/Microsoft Visual Studio", "C:/Program Files (x86)/Microsoft Visual Studio"] {
        for year in std::fs::read_dir(root).into_iter().flatten().flatten() {
            for edition in std::fs::read_dir(year.path()).into_iter().flatten().flatten() {
                for msvc in std::fs::read_dir(edition.path().join("VC/Tools/MSVC")).into_iter().flatten().flatten() {
                    let dir = msvc.path().join("bin/Hostx64/x64");
                    if dir.join("cl.exe").exists() {
                        found.push(dir);
                    }
                }
            }
        }
    }
    found.sort();
    found.pop()
}
