try:
    from langchain.schema import Document
    print('imported schema', Document)
except Exception as e:
    from langchain_core.documents import Document
    print('fallback', Document)
