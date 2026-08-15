"""Stand the whole system up locally, run the simulated stream through it, monitor, report.

This is the local equivalent of `docker compose up`, and it is the path every number in the
README was measured on. It starts the real API in a subprocess, waits for its health check,
posts the simulated stream over HTTP, drains the monitoring windows, exports the history and
prints a summary.

The store is reset at the start unless `--keep` is passed, so a rerun on a fixed seed produces
the same numbers rather than appending to the last run and quietly changing them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.monitor import MonitorRunner  # noqa: E402
from src.store import Store  # noqa: E402
from src.stream import StreamRunner  # noqa: E402
from src.train import train  # noqa: E402


def ensure_model(config, retrain: bool) -> None:
    model_path = config.path(config.service.model_path)
    if retrain or not model_path.exists():
        print("fitting the scorecard", flush=True)
        performance = train(config)
        print(json.dumps(performance, indent=2), flush=True)
    else:
        print(f"using the existing artifact at {model_path}", flush=True)


def start_api(host: str, port: int) -> subprocess.Popen:
    print(f"starting the API on {host}:{port}", flush=True)
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api:app", "--host", host, "--port", str(port),
         "--log-level", "warning"],
        cwd=str(PROJECT_ROOT),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--total", type=int, default=None, help="override the stream length")
    parser.add_argument("--retrain", action="store_true", help="refit even if an artifact exists")
    parser.add_argument("--keep", action="store_true", help="append to the existing store")
    args = parser.parse_args()

    config = load_config()
    if args.total is not None:
        config.stream.total_requests = args.total

    ensure_model(config, args.retrain)

    store = Store(config.path(config.service.db_path))
    if not args.keep:
        print("resetting the monitoring store", flush=True)
        store.reset()

    api = start_api(args.host, args.port)
    started = time.perf_counter()
    try:
        runner = StreamRunner(config, f"http://{args.host}:{args.port}")
        runner.wait_for_api(timeout=90.0)

        print(f"posting {config.stream.total_requests} simulated requests", flush=True)
        manifest = runner.run()
        print(
            f"stream done: {manifest['requests_accepted']} accepted, "
            f"{manifest['requests_rejected']} rejected, "
            f"{manifest['requests_per_second']} requests/second",
            flush=True,
        )

        print("running the monitor over every complete window", flush=True)
        monitor = MonitorRunner(config)
        drained = monitor.drain()
        exported = monitor.export_reports()
    finally:
        api.terminate()
        try:
            api.wait(timeout=15)
        except subprocess.TimeoutExpired:
            api.kill()
        print("API stopped", flush=True)

    elapsed = time.perf_counter() - started
    latencies = sorted(store.fetch_latencies())
    summary = {
        "requests_scored": store.count_scores(),
        "requests_rejected": manifest["requests_rejected"],
        "windows_processed": drained["windows_processed"],
        "alerts_fired": drained["alerts_fired"],
        "metric_rows": exported["metric_rows"],
        "wall_clock_seconds": round(elapsed, 1),
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "p50": round(latencies[len(latencies) // 2], 2) if latencies else None,
            "p95": round(latencies[int(len(latencies) * 0.95)], 2) if latencies else None,
            "p99": round(latencies[int(len(latencies) * 0.99)], 2) if latencies else None,
        },
    }
    print("\nend to end summary")
    print(json.dumps(summary, indent=2))

    reports = config.path("reports")
    with open(reports / "end_to_end_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
