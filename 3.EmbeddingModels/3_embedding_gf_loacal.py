from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2", dimensions=32)
texts = 'Delhi is the capital of India.'
vector= embeddings.embed_query(texts)
print(str(vector))