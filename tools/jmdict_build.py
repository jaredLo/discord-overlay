#!/usr/bin/env python3
"""
Build a compact JMdict index for offline glossing.

Input:  JMdict_e or JMdict XML (optionally .gz)
Output: jmdict.min.json mapping string -> [gloss1, gloss2, ...]

Usage:
  python tools/jmdict_build.py /path/to/JMdict_e.gz jmdict.min.json

Notes:
- Only extracts English glosses. Non-English gloss entries are ignored.
- Keys include both kanji entries (keb) and readings (reb) for broad matching.
- Keeps a small list of glosses per key; your overlay uses the first by default.
"""
import sys, json, gzip, io
import xml.etree.ElementTree as ET

def open_any(path):
    if path.endswith('.gz'):
        return io.TextIOWrapper(gzip.open(path, 'rb'), encoding='utf-8')
    return open(path, 'r', encoding='utf-8')

def build_index(src_path: str) -> dict:
    idx = {}
    with open_any(src_path) as f:
        # Incremental parse to keep memory lower
        it = ET.iterparse(f, events=("start","end"))
        _, root = next(it)
        entry = None
        for ev, el in it:
            if ev == 'start' and el.tag == 'entry':
                entry = {'keb': [], 'reb': [], 'gloss': []}
            elif ev == 'end':
                if el.tag == 'keb' and entry is not None:
                    entry['keb'].append(el.text or '')
                elif el.tag == 'reb' and entry is not None:
                    entry['reb'].append(el.text or '')
                elif el.tag == 'gloss' and entry is not None:
                    lang = el.attrib.get('{http://www.w3.org/XML/1998/namespace}lang')
                    if lang in (None, 'eng'):
                        g = (el.text or '').strip()
                        if g:
                            entry['gloss'].append(g)
                elif el.tag == 'entry' and entry is not None:
                    keys = set(entry['keb'] + entry['reb'])
                    if entry['gloss'] and keys:
                        for k in keys:
                            if not k:
                                continue
                            lst = idx.setdefault(k, [])
                            # append unique glosses (first few only)
                            for g in entry['gloss']:
                                if g not in lst:
                                    lst.append(g)
                                    if len(lst) >= 5:
                                        break
                    # reset
                    root.clear()
                    entry = None
    return idx

def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/jmdict_build.py JMdict_e(.gz) jmdict.min.json", file=sys.stderr)
        sys.exit(2)
    src, out = sys.argv[1], sys.argv[2]
    idx = build_index(src)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(idx, f, ensure_ascii=False)
    print(f"Wrote {len(idx)} entries to {out}")

if __name__ == '__main__':
    main()

