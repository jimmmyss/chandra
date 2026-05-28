#!/usr/bin/env python3
"""Bin-pack PDFs across N vLLM shards balanced by page count, run chandra
on each shard in parallel, and show per-shard progress bars.

Usage:
    client.py <input_dir> <output_dir> [num_servers] [extra chandra args...]

Inputs are PDFs (page count via pypdfium2) and images (counted as 1 page).
Each shard is given a temp directory of symlinks. Shards are populated via
Longest-Processing-Time greedy: sort files by page count descending and
assign each to the shard with the lowest running total, which gives a tight
upper bound (4/3 of optimal) on the worst-case shard load.

While chandra runs we parse its stdout (`[i/N] Processing: foo.pdf`,
`Loaded P page(s)`, `Processing pages A-B...`, `Saved: ... (P page(s))`)
to drive a per-shard progress bar. Each shard's full raw output is tee'd
to `<output_dir>/.shard_<i>.log` so nothing is lost.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import pypdfium2 as pdfium
except ImportError:
    print("ERROR: pypdfium2 is required. Install with: pip install pypdfium2",
          file=sys.stderr)
    sys.exit(1)


SUPPORTED_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff", ".bmp"}

RE_PROCESSING_FILE = re.compile(r"\[(\d+)/(\d+)\] Processing: (.+?)\s*$")
RE_LOADED_PAGES = re.compile(r"Loaded\s+(\d+)\s+page")
RE_PROCESSING_BATCH = re.compile(r"Processing pages\s+(\d+)-(\d+)")
RE_SAVED = re.compile(r"Saved:\s+(\S+)\s+\((\d+)\s+page")
RE_VLLM_ERROR = re.compile(r"Error during VLLM generation|vllm error", re.IGNORECASE)


# ---------- server health ----------

def server_ready(url: str, timeout: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(url + "/models")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


def wait_for_servers(urls: List[str], wait_seconds: int = 600,
                     poll_interval: float = 5.0) -> List[bool]:
    """Wait up to wait_seconds for each server to respond on /v1/models.
    Returns a bool list parallel to urls."""
    deadline = time.time() + wait_seconds
    states = [False] * len(urls)
    last_print = 0.0
    while time.time() < deadline and not all(states):
        for i, url in enumerate(urls):
            if not states[i]:
                states[i] = server_ready(url)
        now = time.time()
        if now - last_print >= 10 or all(states):
            ready = sum(states)
            print(f"  waiting for servers... {ready}/{len(urls)} ready "
                  f"(timeout in {int(deadline - now)}s)", flush=True)
            last_print = now
        if all(states):
            break
        time.sleep(poll_interval)
    return states


# ---------- discovery & page counting ----------

def discover_inputs(input_dir: Path) -> List[Path]:
    seen: set[Path] = set()
    for entry in input_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() in SUPPORTED_EXTS:
            seen.add(entry.resolve())
    return sorted(seen)


def count_pages(path: Path) -> int:
    if path.suffix.lower() != ".pdf":
        return 1
    try:
        doc = pdfium.PdfDocument(str(path))
        try:
            return len(doc)
        finally:
            doc.close()
    except Exception:
        return 1


def count_pages_parallel(paths: List[Path], workers: int = 8) -> List[Tuple[Path, int]]:
    out: List[Optional[int]] = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for idx, n in enumerate(ex.map(count_pages, paths)):
            out[idx] = n
    return list(zip(paths, [n or 1 for n in out]))


# ---------- LPT bin packing ----------

def bin_pack(items: List[Tuple[Path, int]], n_shards: int) -> List[List[Tuple[Path, int]]]:
    shards: List[List[Tuple[Path, int]]] = [[] for _ in range(n_shards)]
    totals = [0] * n_shards
    for path, pages in sorted(items, key=lambda x: -x[1]):
        i = min(range(n_shards), key=lambda k: (totals[k], k))
        shards[i].append((path, pages))
        totals[i] += pages
    return shards


# ---------- shard state ----------

@dataclass
class ShardState:
    shard_id: int
    total_pages: int
    total_files: int
    pages_done: int = 0
    files_done: int = 0
    errors: int = 0
    current_file: str = ""
    status: str = "queued"
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    _in_flight_batch: int = 0  # pages in the batch currently executing
    lock: threading.Lock = field(default_factory=threading.Lock)


def reader_thread(proc: subprocess.Popen, state: ShardState, log_path: Path) -> None:
    """Tee chandra stdout to log file and parse it for progress updates."""
    assert proc.stdout is not None
    with log_path.open("w", buffering=1) as log_f:
        for line in proc.stdout:
            log_f.write(line)
            with state.lock:
                if state.started_at is None:
                    state.started_at = time.time()
                state.status = "running"

                m = RE_PROCESSING_FILE.search(line)
                if m:
                    state.current_file = m.group(3)
                    continue

                m = RE_PROCESSING_BATCH.search(line)
                if m:
                    a, b = int(m.group(1)), int(m.group(2))
                    # Previous batch (if any) just finished — chandra is synchronous.
                    state.pages_done += state._in_flight_batch
                    state._in_flight_batch = b - a + 1
                    continue

                m = RE_SAVED.search(line)
                if m:
                    # The last batch of this file just finished.
                    state.pages_done += state._in_flight_batch
                    state._in_flight_batch = 0
                    state.files_done += 1
                    continue

                if RE_VLLM_ERROR.search(line):
                    state.errors += 1
                    continue
    proc.wait()
    with state.lock:
        state.pages_done += state._in_flight_batch
        state._in_flight_batch = 0
        state.exit_code = proc.returncode
        state.finished_at = time.time()
        state.status = "done" if proc.returncode == 0 else "failed"


# ---------- progress rendering ----------

def fmt_dur(seconds: float) -> str:
    if seconds is None or seconds != seconds or seconds == float("inf"):
        return "  ?  "
    s = int(seconds)
    if s < 60:
        return f"{s:>3}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def render_line(s: ShardState, now: float, bar_w: int = 30) -> str:
    with s.lock:
        done = min(s.pages_done, s.total_pages)
        pct = (done / s.total_pages * 100.0) if s.total_pages else 100.0
        filled = int(pct / 100 * bar_w)
        bar = "#" * filled + "-" * (bar_w - filled)
        if s.started_at and done > 0 and s.finished_at is None:
            elapsed = now - s.started_at
            rate = done / elapsed if elapsed > 0 else 0
            eta = (s.total_pages - done) / rate if rate > 0 else float("inf")
        elif s.finished_at:
            eta = 0.0
        else:
            eta = float("inf")
        marker = {"queued": ".", "running": ">", "done": "v", "failed": "X"}.get(s.status, "?")
        cur = s.current_file[:30] if s.current_file else ""
        err = f"err {s.errors:>4}" if s.errors else "         "
        return (
            f"shard {s.shard_id} {marker} |{bar}| "
            f"{done:>6}/{s.total_pages:<6} ({pct:5.1f}%) "
            f"files {s.files_done:>4}/{s.total_files:<4} "
            f"{err} "
            f"ETA {fmt_dur(eta):>7} "
            f"{cur}"
        )


class ProgressRenderer(threading.Thread):
    def __init__(self, states: List[ShardState], stop_event: threading.Event,
                 isatty: bool):
        super().__init__(daemon=True)
        self.states = states
        self.stop_event = stop_event
        self.isatty = isatty
        self.interval = 0.5 if isatty else 15.0
        self._first_render = True

    def run(self) -> None:
        n = len(self.states)
        while not self.stop_event.is_set():
            self._render(n)
            self.stop_event.wait(self.interval)
        self._render(n, final=True)

    def _render(self, n: int, final: bool = False) -> None:
        now = time.time()
        lines = [render_line(s, now) for s in self.states]
        total_pages = sum(s.total_pages for s in self.states)
        with_lock_done = 0
        for s in self.states:
            with s.lock:
                with_lock_done += min(s.pages_done, s.total_pages)
        overall_pct = (with_lock_done / total_pages * 100.0) if total_pages else 100.0
        footer = (
            f"TOTAL    |{'#' * int(overall_pct / 100 * 30)}"
            f"{'-' * (30 - int(overall_pct / 100 * 30))}| "
            f"{with_lock_done}/{total_pages} ({overall_pct:5.1f}%)"
        )

        if self.isatty:
            if not self._first_render:
                # move cursor up over the previously-printed block
                sys.stdout.write(f"\x1b[{n + 1}A")
            self._first_render = False
            for ln in lines:
                sys.stdout.write("\x1b[2K" + ln + "\n")
            sys.stdout.write("\x1b[2K" + footer + "\n")
            sys.stdout.flush()
        else:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{ts}] progress")
            for ln in lines:
                print("  " + ln)
            print("  " + footer)
            sys.stdout.flush()


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sharded chandra runner with page-balanced shards and progress bars",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("num_servers", nargs="?", type=int, default=None,
                    help="number of shards / vLLM servers (default: # of GPUs)")
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="extra args forwarded to chandra")
    args = ap.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    extra_args: List[str] = [a for a in args.extra if a != "--"]

    if not input_dir.is_dir():
        print(f"ERROR: input dir not found: {input_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    n_shards = args.num_servers
    if n_shards is None:
        try:
            n_shards = int(subprocess.check_output(
                ["nvidia-smi", "-L"], text=True).strip().count("\n") + 1)
        except Exception:
            n_shards = 1
    if n_shards < 1:
        n_shards = 1

    base_port = int(os.environ.get("BASE_PORT", "8000"))
    max_workers = os.environ.get("MAX_WORKERS_PER_SHARD", "16")
    max_retries = os.environ.get("MAX_RETRIES", "6")
    batch_size = os.environ.get("BATCH_SIZE", "28")
    skip_existing = os.environ.get("SKIP_EXISTING", "1") == "1"

    # --- discover files
    files = discover_inputs(input_dir)
    if not files:
        print(f"No supported files in {input_dir}")
        return 1

    if skip_existing:
        kept: List[Path] = []
        skipped = 0
        for f in files:
            stem = f.stem
            if (output_dir / stem / f"{stem}.md").exists():
                skipped += 1
            else:
                kept.append(f)
        print(f"Resume mode: {skipped} already done, {len(kept)} remaining "
              f"(of {len(files)}).")
        files = kept
        if not files:
            print("Nothing to do. (Set SKIP_EXISTING=0 to force reprocess.)")
            return 0
    else:
        print(f"Processing {len(files)} files (SKIP_EXISTING=0).")

    # --- count pages
    print(f"Counting pages for {len(files)} files...", flush=True)
    t0 = time.time()
    items = count_pages_parallel(files, workers=min(16, max(1, len(files))))
    total_pages = sum(p for _, p in items)
    print(f"  done in {time.time() - t0:.1f}s. total pages = {total_pages:,}")

    # --- bin pack
    if n_shards > len(items):
        n_shards = len(items)
    shards = bin_pack(items, n_shards)
    print(f"Sharding across {n_shards} server(s):")
    for i, sh in enumerate(shards):
        tot = sum(p for _, p in sh)
        max_p = max((p for _, p in sh), default=0)
        print(f"  shard {i}: {len(sh):>4} files, "
              f"{tot:>6} pages (largest file: {max_p} pages)")

    # --- pre-flight server health check
    urls = [f"http://localhost:{base_port + i}/v1" for i in range(n_shards)]
    wait_seconds = int(os.environ.get("SERVER_WAIT_SECONDS", "600"))
    if wait_seconds > 0:
        print(f"\nChecking server health on {n_shards} port(s)...", flush=True)
        ready = wait_for_servers(urls, wait_seconds=wait_seconds)
        not_ready = [i for i, ok in enumerate(ready) if not ok]
        if not_ready:
            print(f"\nERROR: server(s) not ready after {wait_seconds}s: "
                  f"{not_ready}", file=sys.stderr)
            print("Check: sudo docker ps --filter name=chandra-vllm-", file=sys.stderr)
            print(f"       sudo docker logs chandra-vllm-{not_ready[0]} --tail 50",
                  file=sys.stderr)
            print("Aborting. Start servers first with: ./server.sh", file=sys.stderr)
            return 3
        print(f"  all {n_shards} server(s) ready.")
    else:
        print("\nSkipping server health check (SERVER_WAIT_SECONDS=0).")

    # --- create symlink directories
    workdir = Path(tempfile.mkdtemp(prefix="chandra-shard-"))
    shard_dirs: List[Path] = []
    for i, sh in enumerate(shards):
        d = workdir / f"shard_{i}"
        d.mkdir()
        shard_dirs.append(d)
        for src, _ in sh:
            link = d / src.name
            try:
                link.symlink_to(src)
            except FileExistsError:
                link.unlink()
                link.symlink_to(src)

    # --- launch chandra subprocesses
    states = [
        ShardState(
            shard_id=i,
            total_pages=sum(p for _, p in shards[i]),
            total_files=len(shards[i]),
        )
        for i in range(n_shards)
    ]
    procs: List[subprocess.Popen] = []
    readers: List[threading.Thread] = []

    def cleanup() -> None:
        for p in procs:
            if p.poll() is None:
                try:
                    p.send_signal(signal.SIGTERM)
                except Exception:
                    pass
        deadline = time.time() + 10
        for p in procs:
            try:
                p.wait(timeout=max(0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()
        shutil.rmtree(workdir, ignore_errors=True)

    def on_signal(signum, frame):  # type: ignore[no-untyped-def]
        print(f"\nGot signal {signum}, terminating shards...", flush=True)
        cleanup()
        sys.exit(130)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    print()
    print("Launching shards:")
    for i in range(n_shards):
        port = base_port + i
        log_path = output_dir / f".shard_{i}.log"
        cmd = [
            "chandra",
            str(shard_dirs[i]),
            str(output_dir),
            "--method", "vllm",
            "--max-workers", max_workers,
            "--max-retries", max_retries,
            "--batch-size", batch_size,
            *extra_args,
        ]
        env = os.environ.copy()
        env["VLLM_API_BASE"] = f"http://localhost:{port}/v1"
        print(f"  shard {i} -> http://localhost:{port}/v1   "
              f"({states[i].total_files} files, "
              f"{states[i].total_pages} pages)   log: {log_path}")
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        procs.append(proc)
        t = threading.Thread(
            target=reader_thread, args=(proc, states[i], log_path), daemon=True,
        )
        t.start()
        readers.append(t)

    # --- progress
    isatty = sys.stdout.isatty()
    print()
    if isatty:
        print("Progress (updates in place; full per-shard output in .shard_*.log):")
    else:
        print("Progress (logged periodically; full per-shard output in .shard_*.log):")
    stop = threading.Event()
    renderer = ProgressRenderer(states, stop, isatty=isatty)
    renderer.start()

    # --- wait for completion
    try:
        for t in readers:
            t.join()
    finally:
        stop.set()
        renderer.join(timeout=5)
        shutil.rmtree(workdir, ignore_errors=True)

    # --- final report
    print()
    failed = [s for s in states if s.exit_code != 0]
    total_done = sum(s.files_done for s in states)
    total_errors = sum(s.errors for s in states)
    for s in states:
        elapsed = (s.finished_at or 0) - (s.started_at or 0) if s.started_at else 0
        rate = s.pages_done / elapsed if elapsed > 0 else 0
        print(f"  shard {s.shard_id}: status={s.status} "
              f"files={s.files_done}/{s.total_files} "
              f"pages={s.pages_done}/{s.total_pages} "
              f"errors={s.errors} "
              f"elapsed={fmt_dur(elapsed)} "
              f"rate={rate:.2f} pg/s "
              f"exit={s.exit_code}")
    print(f"\nProcessed {total_done} files across {n_shards} shards "
          f"(vLLM errors observed: {total_errors}).")

    # --- scan for empty / all-zero-token outputs (chandra's silent-fail mode)
    empty_outputs: List[Tuple[str, int]] = []
    for shard in shards:
        for src, _ in shard:
            meta = output_dir / src.stem / f"{src.stem}_metadata.json"
            if not meta.exists():
                continue
            try:
                with meta.open() as f:
                    m = json.load(f)
                if m.get("total_token_count", 0) == 0:
                    empty_outputs.append((src.stem, m.get("num_pages", 0)))
            except Exception:
                pass
    if empty_outputs:
        report = output_dir / ".empty_outputs.txt"
        with report.open("w") as f:
            for stem, pages in empty_outputs:
                f.write(f"{stem}\t{pages}\n")
        print(f"\nWARNING: {len(empty_outputs)} file(s) produced 0-token output "
              f"(chandra exhausted retries and saved empty markdown).")
        print(f"         See: {report}")
        print(f"         Sample: {', '.join(s for s, _ in empty_outputs[:5])}"
              f"{'...' if len(empty_outputs) > 5 else ''}")
        print(f"         To redo these, delete their dirs and rerun:")
        print(f"           xargs -I{{}} rm -rf {output_dir}/{{}} < {report}")
        # Non-zero exit so wrappers/CI can catch this
        if not failed:
            return 4

    if failed:
        print(f"WARNING: {len(failed)} shard(s) reported errors. "
              f"See {output_dir}/.shard_*.log")
        return 1
    print(f"Done. Output in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
