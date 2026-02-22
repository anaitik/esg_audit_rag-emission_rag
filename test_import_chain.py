try:
    from langchain.schema import Document
    print('Document from langchain.schema ->', Document)
except Exception:
    from langchain_core.documents import Document
    print('Document fallback ->', Document)

try:
    from langchain.prompts import PromptTemplate
    print('PromptTemplate from langchain.prompts ->', PromptTemplate)
except Exception:
    from langchain_core.prompts import PromptTemplate
    print('PromptTemplate fallback ->', PromptTemplate)

try:
    from langchain.chains import RetrievalQA
    print('RetrievalQA from langchain.chains ->', RetrievalQA)
except Exception:
    from langchain_classic.chains.retrieval_qa.base import RetrievalQA
    print('RetrievalQA fallback ->', RetrievalQA)
