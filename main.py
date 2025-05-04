import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

print("\n--- Hand Gesture Recognition ---")

print("Dataset: https://www.kaggle.com/gti-upm/leapgestrecog\n")


# let 5 gesture 
num_samples = 500
image_size = 50 * 50
num_classes = 5
X_dummy = np.random.rand(num_samples, image_size) + np.random.normal(0, 0.1, (num_samples, image_size)) # Add some noise

y_dummy = np.random.randint(0, num_classes, num_samples)
labels = [f'Gesture_{i}' for i in range(num_classes)]



X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size=0.25, random_state=42, stratify=y_dummy)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')

print("Training KNN model (K=5)...")
knn_model.fit(X_train_scaled, y_train)
print("Training complete.")


y_pred = knn_model.predict(X_test_scaled)


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=labels)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nModel Evaluation (on dummy data):")
print(f"  Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
# print(conf_matrix)

# --- 6. Visualize Confusion Matrix ---
plt.figure(figsize=(7, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - KNN Gestures (Dummy Data)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
print("\nShowing Confusion Matrix Visualization...")
plt.show()

# Note: Visualizing actual gesture predictions would need real image loading.
