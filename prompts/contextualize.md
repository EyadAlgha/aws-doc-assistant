Given the chat history and the latest user question, rewrite the latest question as a standalone question. Resolve pronouns and references to concrete entities. Keep the user's actual intent. Do not add an AWS service or product the user did not mention — only resolve what they referred to. If already standalone, return it unchanged. Return only the question.
 
Chat history:
{history}
 
Latest question: {q}
Standalone question: