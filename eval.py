from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall, context_precision, faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
import json
from rag import retrieve, format_context, SYSTEM, llm

import time

embeddings = HuggingFaceEmbeddings(
    model_name = 'BAAI/bge-base-en-v1.5',
    model_kwargs= {'device': 'cuda'},
    encode_kwargs = {'normalize_embeddings': True, 'batch_size': 256}
)

judge_llm = LangchainLLMWrapper(Ollama(model="qwen2.5:7b", temperature=0))
judge_emb = LangchainEmbeddingsWrapper(embeddings)

print(f'Judge-LLM and Judge-Embeddings loaded successfully...')

golden = json.load(open('eval/golden_set.json'))

rows = []
for item in golden:
    q = item['question']
    docs = retrieve(q, [])
    ctx = format_context(docs)
    ans = llm.invoke(SYSTEM.format(context=ctx, question=q))
    rows.append({
        'question': q,
        'answer': ans,
        'contexts': [d.page_content for d in docs],
        'ground_truth': item['ground_truth']
    })

dataset = Dataset.from_list(rows)
print(f'Dataset built successfully...')

start_time = time.perf_counter()
result = evaluate(
    dataset,
    metrics = [context_recall, context_precision, faithfulness, answer_relevancy],
    llm = judge_llm,
    embeddings = judge_emb,
    run_config=RunConfig(max_workers=1, timeout=600, max_retries=3),
)
end_time = time.perf_counter()
elapsed_seconds = end_time - start_time
elapsed_minutes = elapsed_seconds / 60

print(f'Evaluation finished in {elapsed_minutes:.2f} minutes.')

print(result)
result.to_pandas().to_csv('eval/eval_results.csv', index=False)