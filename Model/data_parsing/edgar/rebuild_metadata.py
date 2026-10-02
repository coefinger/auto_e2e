#!/usr/bin/env python3
"""Rebuild a rosbag2 metadata.yaml from the mcap files' own summaries.

rosbag2 can only open a split recording as one bag through metadata.yaml; without
it every chunk is a standalone bag and each split boundary costs a 12.8 s sample
window. A recording killed mid-write never gets one. Point it at a directory of
symlinks to the chunks when the originals must stay untouched:

    python3 Model/data_parsing/edgar/rebuild_metadata.py /mirror/of/bag_dir \
        --like /some/sibling/bag_with_metadata --write
 Everything here is read from the mcap summary sections (0.1 s/file), so
the result is the recording's real counts, not a guess. QoS profiles are not in
the mcap, so they are borrowed per-topic from --like (a sibling recording of the
same rig) and left empty for topics it does not have; rosbag2 only needs them to
re-offer a publisher, and reading does not.
"""
import argparse
import re
import sys
from pathlib import Path

import yaml
from mcap.reader import make_reader


def natural(p: Path):
    m = re.search(r"_(\d+)\.mcap$", p.name)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_dir")
    ap.add_argument("--like", help="sibling bag whose metadata.yaml supplies QoS strings")
    ap.add_argument("--write", action="store_true", help="write it (default: dry run)")
    a = ap.parse_args()

    d = Path(a.bag_dir)
    files = sorted(d.glob("*.mcap"), key=natural)
    if not files:
        sys.exit(f"no .mcap in {d}")
    if (d / "metadata.yaml").exists():
        sys.exit(f"{d}/metadata.yaml already exists; refusing to overwrite")

    qos = {}
    if a.like:
        like = yaml.safe_load(open(Path(a.like) / "metadata.yaml"))["rosbag2_bagfile_information"]
        qos = {t["topic_metadata"]["name"]: t["topic_metadata"].get("offered_qos_profiles", "")
               for t in like["topics_with_message_count"]}

    entries, counts, types = [], {}, {}
    start, end, total = None, None, 0
    unreadable = []
    for f in files:
        # A recording killed mid-write leaves its last chunk without a summary
        # (that is usually why metadata.yaml is missing at all). Such a file
        # cannot be indexed, so it is left out of the bag rather than guessed at.
        try:
            with open(f, "rb") as fh:
                s = make_reader(fh).get_summary()
            if s is None or s.statistics is None:
                raise ValueError("no summary section")
        except Exception as e:
            unreadable.append((f.name, type(e).__name__))
            continue
        st = s.statistics
        entries.append({"path": f.name,
                        "starting_time": {"nanoseconds_since_epoch": int(st.message_start_time)},
                        "duration": {"nanoseconds": int(st.message_end_time - st.message_start_time)},
                        "message_count": int(st.message_count)})
        total += int(st.message_count)
        start = st.message_start_time if start is None else min(start, st.message_start_time)
        end = st.message_end_time if end is None else max(end, st.message_end_time)
        for cid, n in st.channel_message_counts.items():
            ch = s.channels[cid]
            counts[ch.topic] = counts.get(ch.topic, 0) + int(n)
            types[ch.topic] = s.schemas[ch.schema_id].name

    topics = [{"topic_metadata": {"name": t, "type": types[t], "serialization_format": "cdr",
                                  "offered_qos_profiles": qos.get(t, "")},
               "message_count": counts[t]} for t in sorted(counts)]
    meta = {"rosbag2_bagfile_information": {
        "version": 5, "storage_identifier": "mcap",
        "duration": {"nanoseconds": int(end - start)},
        "starting_time": {"nanoseconds_since_epoch": int(start)},
        "message_count": total, "topics_with_message_count": topics,
        "compression_format": "", "compression_mode": "",
        "relative_file_paths": [e["path"] for e in entries], "files": entries}}

    print(f"{len(entries)}/{len(files)} files  {total} msgs  {(end-start)/1e9:.0f}s  {len(topics)} topics")
    for name, err in unreadable:
        print(f"  EXCLUDED (unfinalized, {err}): {name}")
    missing = [t for t in counts if t not in qos]
    print(f"topics without a QoS string from --like: {len(missing)}")
    if a.write:
        (d / "metadata.yaml").write_text(yaml.safe_dump(meta, sort_keys=False, width=10**6))
        print(f"wrote {d}/metadata.yaml")
    else:
        print("dry run; pass --write")


if __name__ == "__main__":
    main()
