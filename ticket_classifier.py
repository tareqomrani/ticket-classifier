from transformers import pipeline

# Load zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define candidate labels
labels = ["Billing", "Bug", "Feature Request", "Account Access", "General Inquiry"]

# Example ticket input
ticket = input("Paste a support ticket or message:\n")

# Classify ticket
result = classifier(ticket, labels)
top_label = result['labels'][0]
score = result['scores'][0]

print(f"Predicted Category: {top_label} (confidence: {score:.2f})")