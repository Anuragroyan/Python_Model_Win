import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

sentences = [
    "I'm so happy to be stuck in traffic for two hours",
    "What a wonderful day to forget my umbrella",
    "I love when my internet stops working during meetings",
    "The food was absolutely amazing, just kidding it was terrible",
    "I won the lottery! Just kidding, I lost my wallet",
    "Looking forward to Monday... said no one ever",
    "It's raining again, how lovely",
    "The product was great and worked as expected",
    "Thank you for your help, it was really appreciated",
    "This is the best movie I’ve seen all year"
]

labels = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  # 1 = sarcastic, 0 = not sarcastic

# Train
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)
model = MultinomialNB()
model.fit(X, labels)

# Extract model data
model_data = {
    "vocab": vectorizer.vocabulary_,
    "classes": model.classes_.tolist(),
    "class_log_prior": model.class_log_prior_.tolist(),
    "feature_log_prob": model.feature_log_prob_.tolist()
}

# Save to JSON
with open("sarcasm_model.json", "w") as f:
    json.dump(model_data, f, indent=2)
