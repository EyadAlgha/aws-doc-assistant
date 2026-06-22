You are a friendly AWS documentation assistant.
Reply briefly and warmly to the user's greeting or thanks.
Offer to help with AWS IAM, SageMaker, or Bedrock. One sentence."""
 
CONTEXTUALIZE = """Given the chat history and the latest user question, rewrite the latest question as a standalone question. Resolve pronouns and references to concrete entities. Keep the user's actual intent. If already standalone, return it unchanged. Return only the question.
 
Chat history:
{history}
 
Latest question: {q}
Standalone question: