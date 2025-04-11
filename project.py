# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
# Load training and testing data
train_df = pd.read_csv("IMDB Dataset.csv")
test_df = pd.read_csv("testing data.csv")

# Extract review columns
train_reviews = train_df['review'].values
test_reviews = test_df['review'].values

# Extract labels
train_labels = train_df['sentiment'].values
test_labels = test_df['sentiment'].values

# Convert labels to 1 (positive) and 0 (negative)
train_labels = [1 if label == "positive" else 0 for label in train_labels]
test_labels = [1 if label == "positive" else 0 for label in test_labels]

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Initialize tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_reviews)  # fit on training data only

# Convert reviews to sequences
train_sequences = tokenizer.texts_to_sequences(train_reviews)
test_sequences = tokenizer.texts_to_sequences(test_reviews)

sequence_lengths = [len(seq) for seq in train_sequences]

# Find the 95th percentile of sequence lengths
max_length = int(np.percentile(sequence_lengths, 95))
print(f"95th percentile of sequence lengths: {max_length}")
# it shows 601 so we will use 600 as our max lengh
max_length=600

# Pad sequences to ensure they are all of length 600
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

train_padded = np.array(train_padded)
test_padded = np.array(test_padded)

print("Padded training data shape:", train_padded.shape)


model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=max_length),  # Embedding layer
    Flatten(),  # Flatten the embedded sequences
    Dense(100, activation='relu'),  # Hidden layer
    Dense(1, activation='sigmoid')  # Output layer (binary classification)
])
# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # For binary classification
              metrics=['accuracy'])

# Train the model
model.fit(train_padded, train_labels, epochs=2)  # Use padded sequences and labels

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_padded, test_labels, verbose=1)

# Print test accuracy
print('Test accuracy:', test_acc)

# Step 1: Get the expected distribution (true labels)
positive_expected = np.sum(test_labels == 1)  # Count of positive labels
negative_expected = np.sum(test_labels == 0)  # Count of negative labels

# Step 2: Generate predictions using the trained model
predictions = model.predict(test_padded)  # Get probabilities for each sample
predicted_labels = (predictions > 0.5).astype(int).flatten()  # Convert probabilities to binary labels

# Count the actual predictions
positive_actual = np.sum(predicted_labels == 1)  # Count of positive predictions
negative_actual = np.sum(predicted_labels == 0)  # Count of negative predictions

# Step 3: Plot the pie charts
fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Create two subplots side by side

# Pie chart 1: Expected distribution
axes[0].pie(
    [positive_expected, negative_expected],
    labels=["Positive", "Negative"],
    autopct="%1.1f%%",
    startangle=90,
    colors=["lightgreen", "lightcoral"]
)
axes[0].set_title("Expected Distribution (True Labels)")

# Pie chart 2: Actual predictions
axes[1].pie(
    [positive_actual, negative_actual],
    labels=["Positive", "Negative"],
    autopct="%1.1f%%",
    startangle=90,
    colors=["lightgreen", "lightcoral"]
)
axes[1].set_title("Actual Predictions (Model Output)")

# Show the plots
plt.tight_layout()
plt.show()

correct_predictions = np.sum(predicted_labels == test_labels)  # Count of correct predictions
wrong_predictions = np.sum(predicted_labels != test_labels)    # Count of wrong predictions

print("correct predictions:", correct_predictions, "wrong predictions:", wrong_predictions)
# Breakdown of wrong predictions
false_positives = np.sum((predicted_labels == 1) & (test_labels == 0))  # Model predicts 1, true label is 0
false_negatives = np.sum((predicted_labels == 0) & (test_labels == 1))  # Model predicts 0, true label is 1

# Total number of samples
total_samples = len(test_labels)

# Step 3: Calculate percentages
correct_percentage = (correct_predictions / total_samples) * 100
wrong_percentage = (wrong_predictions / total_samples) * 100

fp_percentage = (false_positives / wrong_predictions) * 100 if wrong_predictions > 0 else 0
fn_percentage = (false_negatives / wrong_predictions) * 100 if wrong_predictions > 0 else 0

# Step 4: Plot the bar graphs
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # Create two subplots side by side

# Bar graph 1: Percentage of correct vs. wrong predictions
axes[0].bar(["Correct", "Wrong"], [correct_percentage, wrong_percentage], color=["lightgreen", "lightcoral"])
axes[0].set_title("Percentage of Correct vs. Wrong Predictions")
axes[0].set_ylabel("Percentage (%)")
axes[0].set_ylim(0, 100)  # Set y-axis limit to 100%

# Bar graph 2: Percentage of false positives vs. false negatives
axes[1].bar(["False Positives", "False Negatives"], [fp_percentage, fn_percentage], color=["orange", "red"])
axes[1].set_title("Percentage of False Positives vs. False Negatives")
axes[1].set_ylabel("Percentage (%)")
axes[1].set_ylim(0, 100)  # Set y-axis limit to 100%

# Show the plots
plt.tight_layout()
plt.show()