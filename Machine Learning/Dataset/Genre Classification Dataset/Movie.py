import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Step 1: Load Dataset from provided files
train_data = pd.read_csv(r'D:/Mustufahussain/CodSoft/Machine Learning/Dataset/Genre Classification Dataset/train_data.txt', delimiter=':::', engine='python', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
test_data = pd.read_csv(r'D:/Mustufahussain/CodSoft/Machine Learning/Dataset/Genre Classification Dataset/test_data.txt', delimiter=':::', engine='python', names=['ID', 'TITLE', 'DESCRIPTION'])
test_data_solution = pd.read_csv(r'D:/Mustufahussain/CodSoft/Machine Learning/Dataset/Genre Classification Dataset/test_data_solution.txt', delimiter=':::', engine='python', names=['GENRE'])

# Step 2: Limit Dataset for Testing
train_data = train_data.head(1000)  # Limit to 1000 rows for faster processing
test_data = test_data.head(1000)    # Limit to 1000 rows for faster processing
test_data_solution = test_data_solution.head(1000)  # Match test data size

# Step 3: Text Preprocessing

# Inspect the initial data
print(f"Initial size of train_data: {train_data.shape}")
print(f"Initial size of test_data: {test_data.shape}")

# Drop rows with NaN values in 'DESCRIPTION' or 'GENRE' in training data
train_data = train_data.dropna(subset=['DESCRIPTION', 'GENRE'])

# Check size after dropping NaN values
print(f"Size after dropping NaN values in train_data: {train_data.shape}")

# Convert 'DESCRIPTION' column to string to avoid errors with .str accessor
train_data['DESCRIPTION'] = train_data['DESCRIPTION'].astype(str)

# Remove completely empty strings or whitespace-only strings in 'DESCRIPTION'
train_data = train_data[train_data['DESCRIPTION'].str.strip() != '']

# Check size after removing empty/whitespace-only 'DESCRIPTION'
print(f"Size after removing empty/whitespace-only 'DESCRIPTION' in train_data: {train_data.shape}")

# Update X_train and y_train
X_train = train_data['DESCRIPTION']
y_train = train_data['GENRE']

# Ensure X_train is non-empty
if X_train.empty:
    raise ValueError("X_train is empty after preprocessing. Please check your dataset.")

# Print the first few entries of X_train for inspection
print(f"Sample of X_train:\n{X_train.head()}")

# Ensure no NaN in test data either
test_data = test_data.dropna(subset=['DESCRIPTION'])
test_data_solution = test_data_solution.dropna(subset=['GENRE'])

# Remove completely empty strings or whitespace-only strings in test data
test_data['DESCRIPTION'] = test_data['DESCRIPTION'].astype(str)
test_data = test_data[test_data['DESCRIPTION'].str.strip() != '']
X_test = test_data['DESCRIPTION']
y_test = test_data_solution['GENRE']

# Step 4: Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Ensure X_train is non-empty
if X_train.empty:
    raise ValueError("X_train is empty after preprocessing. Please check your dataset.")

X_train_tfidf = vectorizer.fit_transform(X_train)

# Check if the vocabulary is empty
if not vectorizer.vocabulary_:
    raise ValueError("The TF-IDF vocabulary is empty. Check your data and preprocessing steps.")

X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Model Training with Class Weights to handle class imbalance
model = LogisticRegression(max_iter=200, class_weight='balanced')

# Train the model
model.fit(X_train_tfidf, y_train)

# Step 6: Evaluation
y_pred = model.predict(X_test_tfidf)

# Print accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Create a formatted classification report
report = classification_report(y_test, y_pred, zero_division=1, output_dict=True)

# Print the classification report in a readable tabular format
print("\nClassification Report:")
print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
print("-" * 60)
for label, metrics in report.items():
    if label not in ["accuracy", "macro avg", "weighted avg"]:  # Skip overall averages for now
        print(f"{label:<15} {metrics['precision']:<10.2f} {metrics['recall']:<10.2f} {metrics['f1-score']:<10.2f} {int(metrics['support']):<10}")
print("-" * 60)

# Include averages at the end
for avg in ["macro avg", "weighted avg"]:
    metrics = report[avg]
    print(f"{avg:<15} {metrics['precision']:<10.2f} {metrics['recall']:<10.2f} {metrics['f1-score']:<10.2f} {'-':<10}")

# Print overall accuracy
print(f"{'Overall Accuracy':<15} {accuracy_score(y_test, y_pred):<10.2f}")
