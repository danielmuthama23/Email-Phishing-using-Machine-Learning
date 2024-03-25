from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Sample data representing email content features (e.g., word frequencies)
# Features: [Word1 frequency, Word2 frequency, ..., WordN frequency]
# Labels: 1 for phishing email, 0 for legitimate email
X = np.array([[10, 5, 0, 3], [2, 15, 8, 0], [0, 3, 20, 10], [5, 7, 3, 2], [0, 0, 2, 18]])
y = np.array([1, 0, 0, 1, 0])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Random Forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation Metrics
conf_matrix = confusion_matrix(y_test, y_pred)
tp = conf_matrix[1][1]
tn = conf_matrix[0][0]
fp = conf_matrix[0][1]
fn = conf_matrix[1][0]

accuracy = accuracy_score(y_test, y_pred)
rate_score = tp / (tp + fn)

# Visualization
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0, 1], ['Legitimate', 'Phishing'])
plt.yticks([0, 1], ['Legitimate', 'Phishing'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(conf_matrix[i][j]), horizontalalignment='center', verticalalignment='center')
plt.show()

# Print Metrics
print("True Positive:", tp)
print("True Negative:", tn)
print("Accuracy:", accuracy)
print("Rate Score:", rate_score)
