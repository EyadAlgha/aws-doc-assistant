import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
BASE = os.path.dirname(os.path.abspath(__file__))

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import json
from pathlib import Path

# Prompts
def load_prompt(name):
    with open(os.path.join(BASE, 'prompts', f'{name}.md')) as f:
        return f.read()
    
SYSTEM = load_prompt('system')
INTENT = load_prompt('intent')
SOCIAL = load_prompt('social')
CONTEXTUALIZE = load_prompt('contextualize')
DECOMPOSE = load_prompt('decompose')
 
print(f'Prompts loaded successfully.')

# load docs
def load_docs(path, source):
    with open(path) as f:
        rows = json.load(f)
    docs = []
    for r in rows:
        meta = r.get('metadata', {})
        pages = meta.get('page', [])
        headings = meta.get('headings', [])
        docs.append(Document(
            page_content = r['text'],
            metadata={
                'chunk_id': r['chunk_id'],
                'unique_id': f"{source}:{r['chunk_id']}",
                'source': source,
                'pages': ','.join(map(str, pages)),
                'headings': ' > '.join(headings)
            }
        ))
    return docs

docs = []
l = 0
for doc_name in Path('output').glob('*.json'):
    source_name = doc_name.stem.replace('_chunks', '')
    docs.extend(load_docs(str(doc_name), source_name))
    l +=1

print(f'{l} Documents loaded successfully.')

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name = 'BAAI/bge-base-en-v1.5',
    model_kwargs= {'device': 'cuda'},
    encode_kwargs = {'normalize_embeddings': True, 'batch_size': 256}
)

CHROMA_DIR = os.path.join(BASE, "storage/chroma")

if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name='aws_docs'
    )
    print(f'Chroma dataset loaded successfully.')
else:
    # missing/empty -> build + persist
    print(f'Building Chroma dataset...')
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name='aws_docs'
    )

dense_retriever = vectorstore.as_retriever(search_kwargs={'k':30})

# Sparse retriever
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 30

# Hybird ensembling fusing dense and sparse retrievals
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)


# Reranker
cross_encoder = HuggingFaceCrossEncoder(
    model_name = 'BAAI/bge-reranker-v2-m3',
    model_kwargs= {'device': 'cuda'}
)
reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)
retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybrid_retriever
)

# llm and grounded prompt
llm = Ollama(model="qwen3:14b", temperature=0, num_predict=-1)

print(f'LLM loaded successfully.')

# Helpers
def classify_intent(q):
    out = llm.invoke(INTENT.format(msg=q)).strip().lower()
    return "chat" if "chat" in out else "docs"
 
def chitchat_reply(q, history):
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    msgs = [SystemMessage(content=SOCIAL)]
    for role, text in history[-4:]:
        msgs.append(HumanMessage(content=text) if role == "human" else AIMessage(content=text))
    msgs.append(HumanMessage(content=q))
    return llm.invoke(msgs)
 
def _fmt_history(history):
    if not history:
        return "(none)"
    return "\n".join(f"{r}: {t}" for r, t in history[-6:])
 
def contextualize(q, history):
    if not history:
        return q
    hist = _fmt_history(history).replace("{", "{{").replace("}", "}}")  # escape braces
    out = llm.invoke(CONTEXTUALIZE.format(history=hist, q=q)).strip()
    return out or q
 
def decompose(q):
    """Return [q] for direct lookups, or [sub1, sub2, ...] for comparison/multi-part."""
    out = llm.invoke(DECOMPOSE.format(q=q)).strip()
    subs = [l.strip(" -•\t") for l in out.split("\n") if l.strip()]
    subs = [s for s in subs if s]
    if not subs:
        return [q]
    # cap at 3, always include original so a direct hit isn't lost
    subs = subs[:3]
    if q not in subs:
        subs = [q] + subs
    return subs[:4]
 
def retrieve(q, history):
    """Full retrieval: contextualize -> decompose -> multi-retrieve -> dedup -> rerank."""
    standalone = contextualize(q, history)
    queries = decompose(standalone)
 
    pool, seen = [], set()
    for sub in queries:
        for d in hybrid_retriever.invoke(sub):
            key = d.metadata.get("unique_id")
            if key in seen:
                continue
            seen.add(key)
            pool.append(d)
 
    if not pool:
        return []
    # rerank the merged pool against the ORIGINAL standalone question
    return reranker.compress_documents(pool, standalone)
 
def format_context(docs_):
    return "\n\n".join(
        f"[{i+1}] ({d.metadata.get('source','')} p{d.metadata.get('pages','')})\n{d.page_content}"
        for i, d in enumerate(docs_)
    )
 
def sources_of(docs_):
    return [
        {"source": d.metadata.get("source", ""),
         "pages": d.metadata.get("pages", ""),
         "snippet": d.page_content[:160]}
        for d in docs_
    ]
 
# Answer (non-stream and stream)
def answer(q, history):
    docs_ = retrieve(q, history)
    if not docs_:
        return "Not found in the provided AWS docs.", []
    ctx = format_context(docs_)
    text = llm.invoke(SYSTEM.format(context=ctx, question=q))
    return text, sources_of(docs_)
 
def answer_stream(q, history):
    docs_ = retrieve(q, history)
    if not docs_:
        yield ("sources", [])
        yield ("token", "Not found in the provided AWS docs.")
        return
    yield ("sources", sources_of(docs_))
    ctx = format_context(docs_)
    prompt = SYSTEM.format(context=ctx, question=q)
    for tok in llm.stream(prompt):
        yield ("token", tok)