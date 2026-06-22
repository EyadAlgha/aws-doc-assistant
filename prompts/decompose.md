You are helping a search system. Decide if the question needs to be broken into simpler sub-questions to retrieve all needed facts.
 
- A direct lookup (one fact: a prefix, an ARN, a single permission) needs NO decomposition.
- A comparison, "difference between", "vs", or a multi-part question SHOULD be decomposed into 2-3 simple standalone search queries.
 
Return ONLY the search queries, one per line, no numbering, no extra text.
If no decomposition is needed, return the original question unchanged on a single line.
 
Question: {q}