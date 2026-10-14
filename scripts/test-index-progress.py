"""Regression check for the app helper's collection-wide indexing progress.

Run after building the CLI:
    uv run --no-project scripts/test-index-progress.py /path/to/digup

Uses only synthetic files, a uv-created fake embedding worker, and a temporary index.
No model download, personal folders, app settings, or installed index are involved.
"""

import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time


COUNTS = ("files", "searchable", "pending", "failed", "skipped")
COUNTED_EVENTS = {"sync", "indexing", "file", "summary", "rest", "idle", "waiting"}
WORKER = '''import base64, json, struct, sys
info = dict(model="fixture", revision="progress", runtime="fake", dtype="f32",
            dim=768, text_only=False, load_seconds=0)
print(json.dumps(dict(ready=True, info=info)), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request["op"] == "quit":
        break
    count = len(request.get("inputs", []))
    vectors = struct.pack("<768f", 1, *([0] * 767)) * count
    print(json.dumps(dict(id=request["id"], ok=True, dim=768,
                         vectors=base64.b64encode(vectors).decode(),
                         errors=[None] * count)), flush=True)
'''


def write_pdf(path, pages=1):
    """A small valid PDF with enough text per page to avoid OCR."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{4 + 2 * page} 0 R" for page in range(pages))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for page in range(pages):
        text = (f"Research page {page + 1} describes searchable documents and useful "
                "scientific observations with enough words for text extraction.")
        stream = f"BT /F1 10 Tf 40 700 Td ({text}) Tj ET".encode()
        objects.append(("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                        "/Resources << /Font << /F1 3 0 R >> >> "
                        f"/Contents {5 + 2 * page} 0 R >>").encode())
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, content in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode() + content + b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend((f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref}\n%%EOF\n").encode())
    path.write_bytes(pdf)


def database_counts(path):
    with sqlite3.connect(path) as db:
        states = dict(db.execute("SELECT state, count(*) FROM files GROUP BY state"))
    return dict(files=sum(states.values()), searchable=states.get("done", 0) + states.get("partial", 0),
                pending=states.get("pending", 0), failed=states.get("failed", 0), skipped=states.get("skipped", 0))


def check_count_fields(events):
    for event in events:
        if event["event"] not in COUNTED_EVENTS:
            continue
        for prefix in ("", "code_"):
            for name in COUNTS:
                value = event.get(prefix + name)
                assert isinstance(value, int) and value >= 0, (prefix + name, event)
            assert event[prefix + "files"] == sum(event[prefix + name] for name in COUNTS[1:]), event


def run_service(binary, base, backfill):
    arguments = [str(binary), "index-service", str(base / "files"),
                 "--code-root", str(base / "code"), "--index", str(base / "index"),
                 "--rich-content-only", "--recent-first", "--linger", "0"]
    if backfill:
        arguments.append("--backfill")
    environment = dict(os.environ, DIGUP_WORKER_DIR=str(base / "worker"))
    # A file avoids blocking on stderr if an extractor writes diagnostics.
    with (base / "stderr.log").open("w+") as stderr:
        process = subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=stderr, text=True, env=environment)
        timeout = threading.Timer(60, process.kill)
        timeout.start()
        try:
            events = [json.loads(line) for line in process.stdout]
            code = process.wait()
            stderr.seek(0)
            assert code == 0, (code, stderr.read(), events)
            assert not any(event["event"] == "error" for event in events), events
            check_count_fields(events)
            return events
        finally:
            timeout.cancel()
            process.stdin.close()
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="Freshly built digup CLI")
    arguments = parser.parse_args()
    binary = arguments.binary.resolve()
    assert binary.is_file(), f"CLI not found: {binary}"
    with tempfile.TemporaryDirectory(prefix="digup-progress-") as temporary:
        base = Path(temporary).resolve()  # Match the crawler's /private/var path spelling on macOS.
        for name in ("files", "code", "worker"):
            (base / name).mkdir()
        subprocess.run(["uv", "venv", "--offline", "--python", sys.executable,
                        str(base / "worker" / ".venv")], check=True, capture_output=True,
                       env=dict(os.environ, UV_CACHE_DIR=str(base / "uv-cache")))
        (base / "worker" / "digup_worker.py").write_text(WORKER)
        (base / "files" / "ordinary.txt").write_text("Plain text stays in Spotlight.")
        recent = []
        for number in range(4):
            path = base / "files" / f"recent-{number}.pdf"
            write_pdf(path)
            os.utime(path, (time.time() - 86400, time.time() - 86400))
            recent.append(path)
        for number in range(9):
            write_pdf(base / "files" / f"bulk-{number}.pdf")
        write_pdf(base / "files" / "long.pdf", pages=90)
        (base / "files" / "broken.pdf").write_text("This is intentionally not a PDF.")
        for name in ("recent.swift", "bulk.swift"):
            (base / "code" / name).write_text('func importantWork() -> String { "research" }\n')
        (base / "code" / "empty.swift").write_text("")

        # Initial sync discovers names, but the recent-first power gate keeps all embedding paused.
        initial = run_service(binary, base, backfill=False)
        assert not any(event["event"] in ("file", "rest", "indexing") for event in initial), initial
        initial_idle = next(event for event in initial if event["event"] == "idle")
        assert (initial_idle["files"], initial_idle["pending"], initial_idle["searchable"]) == (15, 15, 0), initial_idle
        assert initial_idle["code_files"] == initial_idle["code_pending"] == 3, initial_idle
        with sqlite3.connect(base / "index" / "index.sqlite") as db:
            assert not db.execute("SELECT path FROM files WHERE path LIKE '%ordinary.txt'").fetchall()
            db.executemany("INSERT INTO activity(path,last_used,source) VALUES(?,?,?)",
                           [(str(path), time.time(), "fixture") for path in recent])
        with sqlite3.connect(base / "index" / "code.sqlite") as db:
            db.execute("INSERT INTO activity(path,last_used,source) VALUES(?,?,?)",
                       (str(base / "code" / "recent.swift"), time.time(), "fixture"))
        paused = run_service(binary, base, backfill=False)
        assert not any(event["event"] in ("file", "rest", "indexing") for event in paused), paused

        completed = run_service(binary, base, backfill=True)
        files = [event for event in completed if event["event"] == "file"]
        assert all(Path(event["path"]).name.startswith("recent-") for event in files[:4]), files[:4]
        # This was the regression: 4/4 describes a recent batch, not the 15-file collection.
        fourth = files[3]
        assert (fourth["done"], fourth["total"]) == (4, 4), fourth
        assert (fourth["files"], fourth["searchable"], fourth["pending"]) == (15, 4, 11), fourth
        assert fourth["code_searchable"] == 0 and fourth["code_pending"] == 3, fourth
        assert files[4]["code"] and Path(files[4]["path"]).name == "recent.swift", files[4]
        # Counts on every later event retain both collections while the active index changes.
        for event in completed:
            if event["event"] in {"indexing", "file", "summary", "rest", "idle"}:
                assert event["files"] == 15 and event["code_files"] == 3, event
        rest = [event for event in completed if event["event"] == "rest"]
        assert rest and all(event["files"] == 15 for event in rest), rest
        assert rest[0]["long_files"] == 1 and rest[-1]["long_files"] == 0, rest
        assert any(event["state"] == "partial" for event in files), files
        assert any(event["event"] == "summary" and event.get("rest") for event in completed), completed
        idle = next(event for event in completed if event["event"] == "idle")
        for prefix, filename in (("", "index.sqlite"), ("code_", "code.sqlite")):
            expected = database_counts(base / "index" / filename)
            assert {name: idle[prefix + name] for name in COUNTS} == expected, (idle, expected)
        assert (idle["pending"], idle["failed"], idle["searchable"]) == (0, 1, 14), idle
        assert (idle["code_pending"], idle["code_skipped"], idle["code_searchable"]) == (0, 1, 2), idle
        assert idle["unfinished"] == 0, idle

        repeated = run_service(binary, base, backfill=True)
        assert not any(event["event"] in ("file", "rest") for event in repeated), repeated
    print("PASS: authoritative main/code counts during sync, recent batches, bulk work, long PDFs, and completion")


if __name__ == "__main__":
    main()
