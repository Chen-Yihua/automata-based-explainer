"""
Parse the per-automaton summary tables out of every experiment_log.txt found
anywhere under test_result/ whose parent directory matches CONFIG_RE (e.g.
test_result/regular_0.8_1000/, or test_result/final_result/regular_0.8_2000/),
into a single tidy CSV. Read-only w.r.t. test_result/ — never writes there.

No manual staging step required: run_regular_experiment.py /
run_realworld_experiment.py already write directly to
test_result/{regular,realworld}_{threshold}_{batch_size}/, which this finds
on its own. If the same (domain, threshold, batch_size) combo exists in more
than one place under test_result/ (e.g. an old manually-archived copy under
final_result/ alongside a fresher direct pipeline run), the
most-recently-modified experiment_log.txt wins and a note is printed.
"""
from __future__ import annotations

import argparse
import os
import re
import csv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_SEARCH_ROOT = os.path.join(PROJECT_ROOT, "test_result")
OUT_CSV = os.path.join(os.path.dirname(__file__), "summary_table.csv")

CONFIG_RE = re.compile(r"^(regular|realworld)_(\d+\.\d+)_(\d+)$")

# "  DocumentReleaseWorkflow  (teacher_states=35  initial_states=31)"
# "  ECG  (clf_train=0.9240  clf_test=0.8929  initial_states=31)"
HEADER_RE = re.compile(
    r"^\s{2}(\w+)\s+\(([^)]*)\)\s*$"
)
INITIAL_RE = re.compile(
    r"^\s*Initial \(RPNI\):\s*train=([\d.]+)\s*validation=([\d.]+)\s*$"
)
ROW_RE = re.compile(
    r"^\s*\|\s*(\w+)\s*\|\s*([\d.]+)→([\d.]+)\s*[✓✗]?\s*\|\s*([\d.]+)→([\d.]+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*$"
)


def parse_kv_block(block: str) -> dict:
    kv = {}
    for part in block.split():
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k] = v
    return kv


def parse_log(path: str, config_name: str, domain: str, threshold: float, batch_size: int) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    i = 0
    n = len(lines)
    while i < n:
        m = HEADER_RE.match(lines[i].rstrip("\n"))
        if m:
            automaton, kvtext = m.group(1), m.group(2)
            kv = parse_kv_block(kvtext)
            # Look ahead a few lines for the "Initial (RPNI): ..." line.
            j = i + 1
            init_train = init_val = None
            while j < min(i + 5, n):
                im = INITIAL_RE.match(lines[j].rstrip("\n"))
                if im:
                    init_train, init_val = float(im.group(1)), float(im.group(2))
                    break
                j += 1
            if init_train is None:
                i += 1
                continue
            # Scan forward for table rows until we hit a blank-blank (end of block).
            k = j + 1
            found_any = False
            while k < n:
                rm = ROW_RE.match(lines[k].rstrip("\n"))
                if rm:
                    found_any = True
                    method, tr_i, tr_f, va_i, va_f, states, time_s = rm.groups()
                    rows.append({
                        "config": config_name,
                        "domain": domain,
                        "threshold": threshold,
                        "batch_size": batch_size,
                        "automaton": automaton,
                        "teacher_states": kv.get("teacher_states", ""),
                        "initial_states": kv.get("initial_states", ""),
                        "clf_train": kv.get("clf_train", ""),
                        "clf_test": kv.get("clf_test", ""),
                        "method": method,
                        "train_init": float(tr_i),
                        "train_final": float(tr_f),
                        "val_init": float(va_i),
                        "val_final": float(va_f),
                        "final_states": int(states),
                        "time_s": float(time_s),
                    })
                    k += 1
                    continue
                if found_any:
                    break
                # stop scanning if we've gone too far without any row match
                if k - j > 8:
                    break
                k += 1
            i = k
            continue
        i += 1
    return rows


def find_experiment_logs(root: str) -> dict[tuple[str, float, int], str]:
    """Recursively find every experiment_log.txt whose parent directory name
    matches CONFIG_RE anywhere under root. Config directories don't nest
    inside each other, so a match stops further descent below it. If the
    same (domain, threshold, batch_size) combo turns up more than once
    (e.g. a stale archived copy alongside a fresh direct pipeline run), the
    most-recently-modified experiment_log.txt wins.
    """
    found: dict[tuple[str, float, int], str] = {}
    mtimes: dict[tuple[str, float, int], float] = {}
    for dirpath, dirnames, _filenames in os.walk(root):
        name = os.path.basename(dirpath)
        cm = CONFIG_RE.match(name)
        if not cm:
            continue
        dirnames[:] = []  # don't descend into a matched config dir
        log_path = os.path.join(dirpath, "experiment_log.txt")
        if not os.path.isfile(log_path):
            continue
        key = (cm.group(1), float(cm.group(2)), int(cm.group(3)))
        mtime = os.path.getmtime(log_path)
        if key not in found or mtime > mtimes[key]:
            if key in found and found[key] != log_path:
                print(f"  [NOTE] {key}: using newer {log_path} (was {found[key]})")
            found[key] = log_path
            mtimes[key] = mtime
    return found


def main(search_root: str | None = None, out_csv: str | None = None) -> None:
    root = search_root or DEFAULT_SEARCH_ROOT
    out_path = out_csv or OUT_CSV

    all_rows = []
    logs = find_experiment_logs(root)
    for (domain, threshold, batch_size), log_path in sorted(logs.items()):
        entry = f"{domain}_{threshold:g}_{batch_size}"
        rows = parse_log(log_path, entry, domain, threshold, batch_size)
        print(f"{entry} ({log_path}): parsed {len(rows)} rows")
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit(f"No rows parsed - check log format / path under {root}.")

    fieldnames = list(all_rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search_root", default=None, help="Override the test_result/ root to scan (for testing).")
    parser.add_argument("--out_csv", default=None, help="Override the output CSV path (for testing).")
    args = parser.parse_args()
    main(search_root=args.search_root, out_csv=args.out_csv)
