# AI-Powered Ticket Classifier (Zero-Shot)

This project uses a local transformer model (BART MNLI) from HuggingFace to classify support tickets into common categories without training.

## Categories
- Billing
- Bug
- Feature Request
- Account Access
- General Inquiry

## Requirements
Install dependencies:
```bash
pip install transformers torch
```

## How to Run
```bash
python ticket_classifier.py
```

Paste a support ticket or customer message when prompted.

---

This project blends my support background with AI, showing how to streamline triage workflows in SaaS environments.