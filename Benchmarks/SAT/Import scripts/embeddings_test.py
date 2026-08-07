import json
from collections import Counter
from pathlib import Path

from numpy import argmax
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).parent / "sat-reading-writing-mc.json"
NUM_QUESTIONS = 50  # set to None to run over the full set

# Load the model (this will download ~16GB on first run)
model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")

topics = [
    "Information and Ideas",
    "Craft and Structure",
    "Expression of Ideas",
    "Standard English Conventions",
]

with open(DATA_PATH) as f:
    items = json.load(f)["items"][:NUM_QUESTIONS]

questions = [item["text"] for item in items]
true_domains = [item["domain"] for item in items]

# Encode everything once in batches, rather than per-question
question_embeddings = model.encode(questions)
topic_embeddings = model.encode(topics)

similarities = cosine_similarity(question_embeddings, topic_embeddings)
predicted_indices = argmax(similarities, axis=1)
predicted_domains = [topics[i] for i in predicted_indices]

correct = 0
mismatches = []
for item, true_domain, pred_domain in zip(items, true_domains, predicted_domains):
    if pred_domain == true_domain:
        correct += 1
    else:
        mismatches.append((item["item_id"], item["question_id"], true_domain, pred_domain))

total = len(items)
print(f"Accuracy: {correct}/{total} ({correct / total:.1%})\n")

print("Mismatches (item_id, question_id, true -> predicted):")
for item_id, question_id, true_domain, pred_domain in mismatches:
    print(f"  {item_id} ({question_id}): {true_domain} -> {pred_domain}")

print("\nPer-domain accuracy:")
domain_totals = Counter(true_domains)
domain_correct = Counter(
    true_domain for true_domain, pred_domain in zip(true_domains, predicted_domains) if true_domain == pred_domain
)
for topic in topics:
    total_t = domain_totals[topic]
    correct_t = domain_correct[topic]
    if total_t:
        print(f"  {topic}: {correct_t}/{total_t} ({correct_t / total_t:.1%})")

# Original single-question version, kept for reference:
#
# from numpy import argmax
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Load the model (this will download ~16GB on first run)
# model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
#
# question = [
#     "Marta Coll and colleagues 2010 Mediterranean Sea biodiversity census reported approximately 17,000 species, nearly double the number reported in Carlo Bianchi and Carla Morri s 2000 census — a difference only partly attributable to the description of new invertebrate species in the interim. Another factor is that the morphological variability of microorganisms is poorly understood compared to that of vertebrates, invertebrates, plants, and algae, creating uncertainty about how to evaluate microorganisms as species. Researchers decisions on such matters therefore can be highly consequential. Indeed, the two censuses reported similar counts of vertebrate, plant, and algal species, suggesting that ______ Which choice most logically completes the text?"
# ]
#
# topics = [
#     "Information and Ideas",
#     "Craft and Structure",
#     "Expression of Ideas",
#     "Standard English Conventions"
# ]
#
# # Encode questions into embeddings
# question_embeddings = model.encode(question)
# topics_embeddings = model.encode(topics)
#
# # Compute similarity between the question and each answer
# similarities = cosine_similarity(question_embeddings, topics_embeddings)[0]
# print("Similarities:", similarities)
# best_topic_index = argmax(similarities)  # Index of the most similar topic
# print(topics[best_topic_index])
