class PhishGuard:
    def __init__(self, sensitivity_threshold=0.5):
        self.sensitivity_threshold = sensitivity_threshold

    def detect_phishing(self, email_content):
        # Simulated phishing detection algorithm
        # In a real-world scenario, this would involve more sophisticated techniques
        if 'urgent action required' in email_content.lower():
            return 1  # Phishing detected
        else:
            return 0  # Legitimate email

def evaluate_performance(true_positives, true_negatives, total_samples):
    accuracy = (true_positives + true_negatives) / total_samples
    true_positive_rate = true_positives / total_samples
    true_negative_rate = true_negatives / total_samples
    return accuracy, true_positive_rate, true_negative_rate

# Simulated dataset
emails = [
    ("Urgent action required: Verify your account now!", 1),  # Phishing email
    ("Your monthly newsletter", 0),  # Legitimate email
    ("Claim your prize now!", 1),  # Phishing email
    ("Meeting reminder", 0)  # Legitimate email
]

# Initialize PhishGuard
phish_guard = PhishGuard()

# Evaluation variables
true_positives = 0
true_negatives = 0

# Evaluate each email
for email_content, label in emails:
    # Detect phishing
    prediction = phish_guard.detect_phishing(email_content)

    # Update evaluation metrics
    if label == 1 and prediction == 1:
        true_positives += 1
    elif label == 0 and prediction == 0:
        true_negatives += 1

# Calculate evaluation metrics
total_samples = len(emails)
accuracy, true_positive_rate, true_negative_rate = evaluate_performance(true_positives, true_negatives, total_samples)

# Print evaluation results
print("Evaluation Results:")
print("True Positives:", true_positives)
print("True Negatives:", true_negatives)
print("Accuracy:", accuracy)
print("True Positive Rate:", true_positive_rate)
print("True Negative Rate:", true_negative_rate)
