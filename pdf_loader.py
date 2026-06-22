import re
import unicodedata
from collections import Counter

import fitz
import pymupdf4llm

fitz.TOOLS.mupdf_display_errors(False)
fitz.TOOLS.mupdf_display_warnings(False)

_PAGE_NUM = re.compile(r"^\d{1,4}$")
_DOTS = re.compile(r"\.{6,}")


def _norm(line):
    s = re.sub(r"[#*`>|]", "", line).strip()
    s = re.sub(r"\b\d{1,4}\b", "", s).strip()
    return s.lower()


def learn_boilerplate(pages, edge_lines=3, min_page_frac=0.25):
    n = len(pages)
    if n < 4:
        return set()
    freq = Counter()
    for p in pages:
        lines = [l for l in p["text"].split("\n") if l.strip()]
        edge = lines[:edge_lines] + lines[-edge_lines:]
        for key in {_norm(l) for l in edge if 0 < len(l.strip()) <= 80}:
            if key:
                freq[key] += 1
    threshold = max(3, int(min_page_frac * n))
    return {k for k, c in freq.items() if c >= threshold}


def strip_boilerplate(text, learned):
    out = []
    for ln in text.split("\n"):
        s = ln.strip()
        if _norm(ln) in learned or _PAGE_NUM.match(s) or _DOTS.search(s):
            continue
        out.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _repair_tables(page, text):
    """pymupdf4llm drops ﬁ/ﬂ ligatures in tables; re-extract them from a
    ligature-preserving textpage and swap the mangled blocks back in."""
    if "|" not in text:
        return text
    tp = page.get_textpage(flags=fitz.TEXTFLAGS_TEXT)
    clean = []
    for tbl in page.find_tables().tables:
        rows = [[unicodedata.normalize("NFKC", c or "") for c in r]
                for r in tbl.extract(textpage=tp)]
        clean.append("\n".join("|" + "|".join(r) + "|" for r in rows))

    blocks, buf, in_tbl = [], [], False
    for ln in text.split("\n"):
        is_row = ln.lstrip().startswith("|")
        if is_row != in_tbl:
            blocks.append(("\n".join(buf), in_tbl))
            buf, in_tbl = [ln], is_row
        else:
            buf.append(ln)
    blocks.append(("\n".join(buf), in_tbl))

    if sum(t for _, t in blocks) != len(clean):
        return text
    it = iter(clean)
    return "\n".join(next(it) if t else b for b, t in blocks)


def load_pdf(path, header_pts=36, footer_pts=36):
    doc = fitz.open(path)
    raw_pages = pymupdf4llm.to_markdown(
        path,
        page_chunks=True,
        margins=(0, header_pts, 0, footer_pts),
        table_strategy="lines",
        ignore_images=True,
        ignore_graphics=True,
        use_ocr=False,
        show_progress=True,
    )

    pages = []
    for i, chunk in enumerate(raw_pages, start=1):
        text = _repair_tables(doc[i - 1], chunk.get("text", ""))
        text = unicodedata.normalize("NFKC", text).strip()
        if not text:
            continue
        headings = [it[1] for it in chunk.get("toc_items", []) if len(it) > 1]
        pages.append({"page_number": i, "text": text, "headings": headings})
    doc.close()

    learned = learn_boilerplate(pages)
    for p in pages:
        p["text"] = strip_boilerplate(p["text"], learned)
    return pages