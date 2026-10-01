import json

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

with open("sat-reading-writing-mc.json", "r") as f:
    data = json.load(f)

question_text = []
question_num = []
question_ids = []
for question in data["items"]:
    question_text.append(question["text"])
    question_num.append(question["item_id"])
    question_ids.append(question["question_id"])

question_text_embeddings = model.encode(question_text)
similarity = cosine_similarity(question_text_embeddings, question_text_embeddings)

similarities = []
for i in range(len(question_text)):
    for j in range(i + 1, len(question_text)):
        similarities.append((i, j, similarity[i][j]))

found_similar = False
similarity_count = 0

for i in range(len(similarities)):
    if similarities[i][2] > 0.8:  # Adjust the threshold as needed
        found_similar = True
        similarity_count += 1
        print(f"Question {question_num[similarities[i][0]]} (ID: {question_ids[similarities[i][0]]}) and Question {question_num[similarities[i][1]]} (ID: {question_ids[similarities[i][1]]}) are similar with a similarity score of {similarities[i][2]}")

if not found_similar:
    print("No questions are similar")

print(f"\nPairs of similar questions found: {similarity_count}")

sorted_similarities = sorted(similarities, key=lambda x: x[2], reverse=True)
print("\nTop 10 most similar questions:")
for i in range(10):
    print(f"Question {question_num[sorted_similarities[i][0]]} (ID: {question_ids[sorted_similarities[i][0]]}) and Question {question_num[sorted_similarities[i][1]]} (ID: {question_ids[sorted_similarities[i][1]]}) with a similarity score of {sorted_similarities[i][2]}")

print("\n Top 10 least similar questions:")
for i in range(10):
    print(f"Question {question_num[sorted_similarities[-(i + 1)][0]]} (ID: {question_ids[sorted_similarities[-(i + 1)][0]]}) and Question {question_num[sorted_similarities[-(i + 1)][1]]} (ID: {question_ids[sorted_similarities[-(i + 1)][1]]}) with a similarity score of {sorted_similarities[-(i + 1)][2]}")