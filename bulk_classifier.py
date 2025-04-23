import pandas as pd
from transformers import pipeline

# Load model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Labels
labels = ["Billing", "Bug", "Feature Request", "Account Access", "General Inquiry"]

# Load CSV
df = pd.read_csv("sample_tickets.csv")

# Add classification results
results = []
for ticket in df["message"]:
    result = classifier(ticket, labels)
    results.append((ticket, result["labels"][0], result["scores"][0]))

# Create result DataFrame
result_df = pd.DataFrame(results, columns=["message", "predicted_category", "confidence"])

# Save results
result_df.to_csv("classified_tickets.csv", index=False)

print("Classification complete. Results saved to 'classified_tickets.csv'.")