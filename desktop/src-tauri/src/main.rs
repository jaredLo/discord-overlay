#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use std::fs::{OpenOptions, create_dir_all};
use std::io::Write;
use std::path::PathBuf;

static LOG_ENABLED: AtomicBool = AtomicBool::new(true);

fn log_file_path() -> PathBuf {
  let mut p = PathBuf::from("../logs");
  if create_dir_all(&p).is_ok() { p.push("tauri-client.log"); return p; }
  PathBuf::from("tauri-client.log")
}

fn ts_ms() -> u128 {
  use std::time::{SystemTime, UNIX_EPOCH};
  SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}

fn log_line(msg: &str) {
  if !LOG_ENABLED.load(Ordering::Relaxed) { return; }
  let line = format!("[{}] {}\n", ts_ms(), msg);
  eprintln!("{}", line.trim_end());
  if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(log_file_path()) {
    let _ = f.write_all(line.as_bytes());
    let _ = f.flush();
  }
}

fn log_duration(label: &str, start: Instant) {
  let elapsed = start.elapsed();
  let ms = (elapsed.as_secs() as u128) * 1000 + (elapsed.subsec_millis() as u128);
  log_line(&format!("{} took {}ms", label, ms));
}

#[tauri::command]
fn client_log(msg: String) -> Result<(), String> { log_line(&format!("frontend: {}", msg)); Ok(()) }

#[tauri::command]
fn export_vocabs(csv: String, file_name: Option<String>) -> Result<String, String> {
  // Write into the same logs dir used for client logs
  let mut dir = log_file_path();
  if dir.pop() { /* now at logs/ */ } else { return Err("failed to resolve logs dir".into()); }
  if let Err(e) = std::fs::create_dir_all(&dir) { return Err(format!("create logs dir failed: {}", e)); }
  let file = match file_name {
    Some(name) if !name.trim().is_empty() => dir.join(name),
    _ => {
      let ts = ts_ms();
      dir.join(format!("vocab_report-{}.csv", ts))
    }
  };
  match std::fs::write(&file, csv.as_bytes()) {
    Ok(_) => Ok(file.display().to_string()),
    Err(e) => Err(format!("write failed: {}", e)),
  }
}

#[tauri::command]
fn export_raw_transcript(text: String, file_name: Option<String>) -> Result<String, String> {
  // Save plain text raw transcript to logs directory
  let mut dir = log_file_path();
  if dir.pop() { /* now at logs/ */ } else { return Err("failed to resolve logs dir".into()); }
  if let Err(e) = std::fs::create_dir_all(&dir) { return Err(format!("create logs dir failed: {}", e)); }
  let file = match file_name {
    Some(name) if !name.trim().is_empty() => dir.join(name),
    _ => {
      let ts = ts_ms();
      dir.join(format!("raw_transcript-{}.txt", ts))
    }
  };
  match std::fs::write(&file, text.as_bytes()) {
    Ok(_) => Ok(file.display().to_string()),
    Err(e) => Err(format!("write failed: {}", e)),
  }
}

fn main() {
  let enabled = std::env::var("TAURI_CLIENT_LOG").ok().map(|v| v.to_lowercase());
  if let Some(v) = enabled { if v=="0"||v=="false"||v=="off" { LOG_ENABLED.store(false, Ordering::Relaxed); } }
  log_line("tauri main: builder start");
  let t_build = Instant::now();
  tauri::Builder::default()
    .invoke_handler(tauri::generate_handler![client_log, export_vocabs, export_raw_transcript])
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
  log_duration("tauri main: builder+run", t_build);
}
