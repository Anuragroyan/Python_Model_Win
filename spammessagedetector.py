import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Define labels and texts
labels = [
    'ham', 'spam', 'ham', 'spam', 'ham',
    'spam', 'ham', 'spam', 'ham', 'spam',
    'ham', 'spam', 'ham', 'spam', 'ham',
    'spam', 'ham', 'spam', 'ham', 'spam',
    'ham', 'spam', 'ham', 'spam', 'ham',
    'spam', 'ham', 'spam', 'ham', 'spam',
    'ham', 'spam', 'ham', 'spam', 'ham',
    'spam', 'ham', 'spam', 'ham', 'spam',
    'ham', 'spam', 'ham', 'spam', 'ham'
]

texts = [
    'Hey, how are you doing?',
    'Win a FREE iPhone now!!!',
    'Are we still on for lunch?',
    'Congratulations, you have won a prize!',
    'Let’s catch up tomorrow.',
    'Claim your $1000 gift card now!',
    'Can you send me the notes?',
    'Limited time offer, act now!',
    'I\'ll be there in 10 minutes.',
    'Exclusive deal just for you!',
    'Happy birthday!',
    'Get cash fast, no credit check!',
    'What time is the meeting?',
    'You have been selected for a survey.',
    'Dinner at 7?',
    'Earn money from home easily!',
    'Did you complete the assignment?',
    'Urgent! Your account is compromised.',
    'Thanks for the update!',
    'Click here to win a trip!',
    'Let\'s meet at the coffee shop.',
    'You are pre-approved for a loan!',
    'Please find the attachment.',
    'Lowest insurance rates guaranteed!',
    'Join the Zoom call now.',
    'Congratulations! You’ve won!',
    'Where should we go today?',
    'Unlock your special bonus now!',
    'See you at the party!',
    'Important notice: Final warning!',
    'Call me when you\'re free.',
    'You’re a lucky winner!',
    'How’s the new job?',
    'Get rich quick with this scheme!',
    'Let’s plan a movie night.',
    'Your account will be deactivated!',
    'Lunch at our usual place?',
    'Hurry up! Limited slots left!',
    'Good luck with your exam!',
    'Don\'t miss this limited-time deal!',
    'That sounds perfect!',
    'Act now to claim your reward!',
    'Catch you later!',
    'Earn $5000 from your phone!',
    'Meeting postponed to tomorrow.'
]

# Train model
vectorizer = CountVectorizer(lowercase=True)
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

# Extract model data
model_data = {
    "vocab": vectorizer.vocabulary_,  # Word to index mapping
    "classes": model.classes_.tolist(),  # ['ham', 'spam']
    "class_log_prior": model.class_log_prior_.tolist(),  # Prior log probs for ham/spam
    "feature_log_prob": model.feature_log_prob_.tolist()  # log(P(word|class)) matrix
}

# Save model to JSON
with open("spam_model.json", "w") as f:
    json.dump(model_data, f, indent=2)

print("Model exported to spam_model.json")