from langchain_openai import OpenAiEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings_model= OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [
    "virat kohli is an indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former indian cricketer and captain of the indian national team.",
    "Sachin Tendulkar is a legendary indian cricketer and one of the greatest batsmen in the history of cricket.",
    "Rohit Sharma is an indian cricketer known for his explosive batting and ability to score big centuries.",
    "Anil Kumble is a former indian cricketer and one of the greatest leg-spinners in the history of cricket."
]

query = "Who is the best indian cricketer?"

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)
similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
# most_similar_doc_index = np.argmax(similarities)
scores = {documents[i]: similarities[i] for i in range(len(documents))}
sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
print("Similarity Scores:")
for doc, score in sorted_scores.items():
    print(f"{doc}: {score:.4f}")


print(documents[np.argmax(similarities)])
