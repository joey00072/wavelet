from __future__ import annotations

import argparse
import json
import random
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_DATASET = Path("outputs/polaris_wait_recoveries/correct_recoveries.jsonl")

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Recovery Trace Viewer</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
    body { margin: 0; background: #080b12; color: #e5e7eb; }
    main { max-width: 1050px; margin: 0 auto; padding: 40px 24px 80px; }
    header { display: flex; gap: 20px; align-items: center; justify-content: space-between; }
    h1 { margin: 0; font-size: 24px; }
    .muted { color: #94a3b8; }
    button { border: 0; border-radius: 8px; padding: 11px 16px; background: #2563eb;
      color: white; font-weight: 650; cursor: pointer; }
    button:hover { background: #3b82f6; }
    .meta { display: flex; flex-wrap: wrap; gap: 8px; margin: 24px 0; }
    .pill { padding: 6px 10px; border: 1px solid #263244; border-radius: 999px;
      background: #111827; color: #cbd5e1; font-size: 13px; }
    section { margin-top: 18px; padding: 20px; border: 1px solid #1f2937;
      border-radius: 12px; background: #0d111b; }
    h2 { margin: 0 0 12px; color: #94a3b8; font-size: 12px;
      letter-spacing: .12em; text-transform: uppercase; }
    pre { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.55;
      font: 14px/1.55 "JetBrains Mono", ui-monospace, monospace; }
    #answer { color: #6ee7b7; }
    #error { color: #fca5a5; }
  </style>
</head>
<body>
<main>
  <header>
    <div><h1>Correct Recovery Traces</h1><div id="count" class="muted">Loading…</div></div>
    <button id="random">Random trace</button>
  </header>
  <div class="meta">
    <span id="phrase" class="pill"></span><span id="cut" class="pill"></span>
    <span id="source" class="pill"></span>
  </div>
  <section><h2>Question</h2><pre id="question"></pre></section>
  <section><h2>Reference answer</h2><pre id="answer"></pre></section>
  <section><h2>Generated recovery</h2><pre id="completion"></pre></section>
  <p id="error"></p>
</main>
<script>
  const fields = ["question", "completion"];
  async function randomTrace() {
    document.getElementById("error").textContent = "";
    try {
      const response = await fetch(`/api/random?t=${Date.now()}`, {cache: "no-store"});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const row = await response.json();
      for (const field of fields) document.getElementById(field).textContent = row[field];
      document.getElementById("answer").textContent = row.reference_answer;
      document.getElementById("phrase").textContent = `phrase: ${row.recovery_phrase}`;
      document.getElementById("cut").textContent = `cut: ${row.cut_newline_from_end}th-last newline`;
      document.getElementById("source").textContent = `source: ${row.source_trace_id}`;
    } catch (error) { document.getElementById("error").textContent = String(error); }
  }
  fetch("/api/stats").then(r => r.json()).then(data => {
    document.getElementById("count").textContent = `${data.count} verified-correct traces`;
  });
  document.getElementById("random").addEventListener("click", randomTrace);
  randomTrace();
</script>
</body>
</html>
"""


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No recovery traces found in {path}.")
    return rows


def make_handler(rows: list[dict[str, Any]]) -> type[BaseHTTPRequestHandler]:
    class RecoveryHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            route = urlparse(self.path).path
            if route == "/":
                self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if route == "/api/random":
                self._send_json(random.choice(rows))
                return
            if route == "/api/stats":
                self._send_json({"count": len(rows)})
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _send_json(self, payload: object) -> None:
            self._send(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return RecoveryHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8781)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(args.dataset)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(rows))
    print(f"Serving {len(rows)} recovery traces on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
