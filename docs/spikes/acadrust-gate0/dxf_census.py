#!/usr/bin/env python3
"""AutoCAD's own statement of what each save contains, read from its ASCII DXF twin.

ACadSharp's samples/ holds `sample_AC10xx_ascii.dxf` next to each `sample_AC10xx.dwg`: AutoCAD
wrote both in one SAVEAS session, so the DXF is AutoCAD describing the same drawing with no DWG
reader in the loop. This script counts, per file, the LAYER table entries, the BLOCK_RECORD
table entries and the top-level model-space entities (ENTITIES section, no group 67 = 1, and
not a VERTEX, SEQEND or ATTRIB child), and prints them as TSV.

    python3 -I dxf_census.py <samples-dir> > dxf-census.tsv

It reads text and imports nothing outside the standard library.
"""
import os
import sys

CHILDREN = {"VERTEX", "SEQEND", "ATTRIB"}


def pairs(path):
    with open(path, "r", errors="replace") as f:
        lines = f.read().split("\n")
    for i in range(0, len(lines) - 1, 2):
        yield lines[i].strip(), lines[i + 1].strip()


def census(path):
    section = None
    table = None
    layers = block_records = 0
    model = 0
    cur = None
    prev = None
    for code, val in pairs(path):
        if code == "0" and val == "SECTION":
            prev = "SECTION"
            continue
        if code == "2" and prev == "SECTION":
            section = val
            prev = None
            continue
        if code == "0" and val == "ENDSEC":
            if section == "ENTITIES" and cur is not None:
                model += 1 if cur["model"] else 0
            section = table = cur = None
            continue
        if section == "TABLES":
            if code == "0" and val == "TABLE":
                prev = "TABLE"
                continue
            if code == "2" and prev == "TABLE":
                table = val
                prev = None
                continue
            if code == "0" and val == "ENDTAB":
                table = None
                continue
            if code == "0" and table == "LAYER" and val == "LAYER":
                layers += 1
            if code == "0" and table == "BLOCK_RECORD" and val == "BLOCK_RECORD":
                block_records += 1
        elif section == "ENTITIES":
            if code == "0":
                if cur is not None and cur["model"]:
                    model += 1
                cur = {"type": val, "model": val not in CHILDREN}
            elif code == "67" and val == "1" and cur is not None:
                cur["model"] = False
    return layers, block_records, model


def main():
    samples = sys.argv[1]
    print("file\tlayers\tblocks\tmodel_entities")
    for name in sorted(os.listdir(samples)):
        if name.startswith("sample_AC10") and name.endswith("_ascii.dxf"):
            l, b, m = census(os.path.join(samples, name))
            print(f"{name.replace('_ascii.dxf', '.dwg')}\t{l}\t{b}\t{m}")


main()
