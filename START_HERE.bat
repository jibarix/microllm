@echo off
rem Double-click to open microllm in your browser. Keep this window open while you use it.
cd /d "%~dp0"
if exist "microllm.exe" (
  "microllm.exe"
  goto :eof
)
if exist "target\release\microllm.exe" (
  "target\release\microllm.exe"
  goto :eof
)
where cargo >nul 2>nul
if errorlevel 1 (
  echo microllm is not built yet, and Rust is not installed.
  echo Install Rust from https://rustup.rs , then double-click START_HERE.bat again.
  echo The first start builds microllm, which takes about a minute.
  pause
  goto :eof
)
echo Building microllm (about a minute, the first time only)...
cargo run --release
if errorlevel 1 pause
