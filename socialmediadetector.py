import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dummy text and labels
texts = [
    "OMG I love this! #fun", 
    "Check out my new photo album!", 
    "Follow me on my journey!", 
    "Just retweeted this cool article", 
    "Had a great time with friends today", 
    "New post on my story!"
]

labels = ["Twitter", "Facebook", "Instagram", "Twitter", "Facebook", "Instagram"]

# Train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
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
with open("socialmedia_model.json", "w") as f:
    json.dump(model_data, f, indent=2)
