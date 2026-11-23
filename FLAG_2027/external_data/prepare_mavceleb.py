#!/usr/bin/env python3
"""Turn a MAV-Celeb v1/v2/v3 training archive into an official-format training manifest.

Reads the official MAV-Celeb split lists (``<Language>_train.txt``: one
``<id>/<lang>/<video>/<clip>.wav <id>/<lang>/<video>/<frame>.jpg`` pair per line)
and extracts only the listed files from the zip into ``--work-dir``. The zip
stays untouched as the permanent raw copy; the work directory can be deleted
after feature extraction and rebuilt at any time.

Output manifest (same 6-column format as ``train_set/train_English.txt``)::

    <pair_id> 1 voices/<lang>/<identity>/<video>_<clip>.wav faces/<lang>/<identity>/<video>_<frame>.jpg <identity> <class>

Identities are namespaced (``mav3_id0001``) so they can never collide with the
official ``id001``-style names; class indices start at ``--class-offset``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--split-lists", type=Path, nargs="+", required=True)
    parser.add_argument("--prefix", required=True, help="identity namespace, e.g. mav3")
    parser.add_argument("--class-offset", type=int, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--zip-root", default="", help="folder inside the zip that holds faces/ and voices/")
    args = parser.parse_args()

    archive = zipfile.ZipFile(args.zip)
    names = set(archive.namelist())
    pairs = []
    for split in args.split_lists:
        for line_number, line in enumerate(split.open(), start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(f"{split}:{line_number}: expected 2 fields")
            pairs.append(tuple(fields))

    identities = sorted({voice.split("/")[0] for voice, _ in pairs})
    class_of = {identity: args.class_offset + i for i, identity in enumerate(identities)}
    rows, missing = [], []
    for index, (voice, face) in enumerate(pairs):
        identity, language, video, clip = voice.split("/")
        f_identity, _, f_video, frame = face.split("/")
        if f_identity != identity:
            raise ValueError(f"Face and voice identity differ: {voice} {face}")
        sources = (f"{args.zip_root}voices/{voice}", f"{args.zip_root}faces/{face}")
        if not all(s in names for s in sources):
            missing.append(voice)
            continue
        name = f"{args.prefix}_{identity}"
        targets = (f"voices/{language}/{name}/{video}_{clip}", f"faces/{language}/{name}/{f_video}_{frame}")
        for source, target in zip(sources, targets):
            out = args.work_dir / target
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(source) as src, out.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        rows.append(f"{args.prefix}_{index:06d} 1 {targets[0]} {targets[1]} {name} {class_of[identity]}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text("\n".join(rows) + "\n")
    report = {"zip": str(args.zip), "split_lists": [str(s) for s in args.split_lists], "pairs_listed": len(pairs),
              "rows_written": len(rows), "missing_in_zip": len(missing), "first_missing": missing[:5],
              "identities": len(identities), "class_range": [args.class_offset, args.class_offset + len(identities) - 1]}
    Path(f"{args.manifest}.report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
