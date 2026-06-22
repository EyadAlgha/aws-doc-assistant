import re
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document

import json
import os
from pathlib import Path
from pdf_loader import load_pdf

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5",
                                   device='cuda',
                                   embed_batch_size=256)

def split_sentence(sent):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", sent)
    return [s.strip() for s in parts if s.strip()]

splitter = SemanticSplitterNodeParser(
    embed_model=embed_model,
    buffer_size=1,
    breakpoint_percentile_threshold=85,
    sentence_splitter=split_sentence
)
def merge_small_nodes(nodes, min_chars=200):
    if not nodes:
        return nodes
    merged = [nodes[0]]
    for nd in nodes[1:]:
        prev = merged[-1]
        if len(prev.text) < min_chars:
            prev.text = prev.text + "\n\n" + nd.text
            # pages -> always a sorted list, works whether scalar or list
            def pglist(m):
                v = m.get("page")
                return v if isinstance(v, list) else ([] if v is None else [v])
            prev.metadata["page"] = sorted(set(pglist(prev.metadata) + pglist(nd.metadata)))
            prev.metadata["headings"] = list(dict.fromkeys(
                (prev.metadata.get("headings") or []) + (nd.metadata.get("headings") or [])))
        else:
            merged.append(nd)
    if len(merged) > 1 and len(merged[-1].text) < min_chars:
        last = merged.pop()
        merged[-1].text += "\n\n" + last.text
        # fold trailing node's pages in too
        def pglist(m):
            v = m.get("page")
            return v if isinstance(v, list) else ([] if v is None else [v])
        merged[-1].metadata["page"] = sorted(set(pglist(merged[-1].metadata) + pglist(last.metadata)))
    return merged

def _parse_pipe_table(block):
    lines = block.split("\n")
    table_lines, started = [], False
    for ln in lines:
        if ln.startswith("|"):
            started = True
            table_lines.append(ln)
        elif started and "|" in ln:
            table_lines.append(ln)
        elif started and ln.strip() and "|" not in ln:
            table_lines.append(ln)
        elif started and not ln.strip():
            break
    raw = "\n".join(table_lines)

    rows, cur = [], ""
    for ln in raw.split("\n"):
        if ln.startswith("|"):
            if cur == "":
                cur = ln
            elif cur.rstrip().endswith("|") and cur.count("|") >= 4:
                rows.append(cur); cur = ln
            else:
                cur += "\n" + ln
        else:
            cur += "\n" + ln
    if cur:
        rows.append(cur)

    parsed = []
    for row in rows:
        parts = row.split("|")
        if parts and parts[0].strip() == "":
            parts = parts[1:]
        if parts and parts[-1].strip() == "":
            parts = parts[:-1]
        cells = [re.sub(r"\s+", " ", c.replace("\n", "")).strip() for c in parts]
        parsed.append(cells)
    return parsed


def explode_permission_table(text):
    if "|" not in text:
        return None
    if "policy action" not in text.lower():
        return None
    rows = _parse_pipe_table(text)
    if len(rows) < 2:
        return None
    header = [c.lower() for c in rows[0]]
    if not any("policy action" in h for h in header):
        return None
    out = []
    for r in rows[1:]:
        if len(r) < 2 or not r[0]:
            continue
        api = r[0]
        action = re.sub(r"^\(required\)\s*", "", r[1], flags=re.I).strip()
        desc = r[2].rstrip(" .") if len(r) > 2 else ""
        if not action:
            continue
        prose = f"To use the {api} API operation, the {action} permission is required."
        if desc:
            prose += f" {desc}."
        out.append(prose)
    return out or None

def semantic_chunk(pages):
    docs = [Document(text=p['text'],
                     metadata={'page': [p['page_number']], 'headings': p['headings']})
            for p in pages]
    nodes = splitter.get_nodes_from_documents(docs)
    nodes = merge_small_nodes(nodes)

    final = []
    for nd in nodes:
        rows = explode_permission_table(nd.text)
        if rows:
            final.append(nd)
            for prose in rows:
                from copy import deepcopy
                rn = deepcopy(nd)
                rn.text = prose
                rn.metadata = dict(nd.metadata)
                final.append(rn)
        else:
            final.append(nd)
    return final

def run(pdf_path, out_path):
    pages = load_pdf(pdf_path)
    chunks = semantic_chunk(pages)

    records = [
        {
            "chunk_id": i,
            "text": node.text,
            "metadata": node.metadata,
        }
        for i, node in enumerate(chunks)
    ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"total pages: {len(pages)}")
    print(f"total chunks: {len(chunks)}")

if __name__ == '__main__':
    print(f'Running document chunking...')
    processed = ['iam-ug', 'sagemaker-dg', 'bedrock-ug']
    data_path = Path('data')
    out_path = Path('output')
    pdf_files = list(data_path.glob('*.pdf'))

    print(f'Found {len(pdf_files) - len(processed)} unprocessed documents in data path...')

    for pfile in pdf_files:
        file_name = pfile.stem
        if file_name in processed:
            continue

        print(f'Chunking {file_name}...')
        chunked_name = out_path / (file_name.split('-')[0] + '_chunks.json')

        run(pfile, chunked_name)
