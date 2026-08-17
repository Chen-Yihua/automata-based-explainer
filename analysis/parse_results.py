"""
Parse the per-automaton summary tables out of test_result/final_result/*/experiment_log.txt
into a single tidy CSV. Read-only w.r.t. final_result — never writes there.
"""
from __future__ import annotations

import os
import re
import csv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FINAL_RESULT_DIR = os.path.join(PROJECT_ROOT, "test_result", "final_result")
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


def main() -> None:
    all_rows = []
    for entry in sorted(os.listdir(FINAL_RESULT_DIR)):
        cm = CONFIG_RE.match(entry)
        if not cm:
            continue
        domain, threshold, batch_size = cm.group(1), float(cm.group(2)), int(cm.group(3))
        log_path = os.path.join(FINAL_RESULT_DIR, entry, "experiment_log.txt")
        if not os.path.isfile(log_path):
            continue
        rows = parse_log(log_path, entry, domain, threshold, batch_size)
        print(f"{entry}: parsed {len(rows)} rows")
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No rows parsed - check log format / path.")

    fieldnames = list(all_rows[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
