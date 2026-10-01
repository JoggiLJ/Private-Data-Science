import json
from collections import Counter
import matplotlib.pyplot as plt

def extract_categories(data):
    if isinstance(data, dict):
        if 'questions' in data and isinstance(data['questions'], list):
            questions = data['questions']
        elif 'items' in data and isinstance(data['items'], list):
            questions = data['items']
        else:
            questions = [data]
    elif isinstance(data, list):
        questions = data
    else:
        questions = []

    categories = []
    for question in questions:
        if not isinstance(question, dict):
            continue
        if 'category' in question:
            categories.append(question['category'])
        elif 'categories' in question:
            values = question['categories']
            if isinstance(values, list):
                categories.extend(values)
            else:
                categories.append(values)
    return categories

with open('HLE_MC.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

categories = extract_categories(data)
category_counts = Counter(categories)

print("Category Counts:")
for category, count in category_counts.items():
    print(f"{category}: {count}")

bars = plt.bar(category_counts.keys(), category_counts.values())
plt.title('Category Counts')
plt.xlabel('Categories')
plt.ylabel('Counts')
plt.xticks(rotation=45, ha='right')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)),
              ha='center', va='bottom')

plt.tight_layout()
plt.savefig('category_counts.png')  # Save the plot as a PNG file
plt.show()