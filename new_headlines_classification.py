import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Suppress warnings
warnings.filterwarnings("ignore")

nltk_data_path = "/mnt/c/python_projects/.nltk_data"

# Ensure path exists
os.makedirs(nltk_data_path, exist_ok=True)

# Download necessary NLTK resources
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('omw-1.4', download_dir=nltk_data_path)  # Required for WordNet in newer NLTK versions


# Define a helper function for displaying confusion matrices
def plot_confusion_matrix(cm, classes, title, normalize=False, cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.title(f"Normalized {title}")
    else:
        plt.title(title)

    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


# 1. Data Preprocessing
print("Step 1: Loading and preprocessing the data...")

# Load the dataset
# Try different file paths and encodings to handle variations
try:
    # Define possible locations and filenames
    directories = ['./', './data/', '../data/', 'data/']
    filenames = ['uci-news-aggregator.csv', 'news-aggregator-dataset.csv', 'newsCorpora.csv', 'newsdata.csv']

    # Try each combination
    found = False
    for directory in directories:
        for filename in filenames:
            filepath = directory + filename
            try:
                df = pd.read_csv(filepath, encoding='utf-8')
                print(f"Successfully loaded {filepath} with utf-8 encoding")
                found = True
                break
            except FileNotFoundError:
                continue
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(filepath, encoding='latin1')
                    print(f"Successfully loaded {filepath} with latin1 encoding")
                    found = True
                    break
                except:
                    continue
        if found:
            break

    if not found:
        # If the loop completes without finding a file, prompt for path
        print("Could not automatically find the dataset file.")
        # Just raise an error - in a real application we could prompt for input
        raise FileNotFoundError(
            "Could not find the dataset file. Please download it from the Kaggle link and place it in the 'data' folder, or update the path in the code.")
except Exception as e:
    print(f"Error loading the dataset: {e}")
    raise

# Check basic information about the dataset
print(f"Dataset shape: {df.shape}")
print(df.head())
print(df['CATEGORY'].value_counts())

# Focus on TITLE and CATEGORY columns
df = df[['TITLE', 'CATEGORY']]

# Handle missing values
df.dropna(subset=['TITLE', 'CATEGORY'], inplace=True)

# Filter to keep only the specified categories
df = df[df['CATEGORY'].isin(['b', 't', 'e', 'm'])]


# Clean text data
def clean_text(text):
    try:
        # Convert to lowercase
        text = str(text).lower()

        # Remove special characters, punctuation, and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Simple tokenization as fallback in case NLTK's word_tokenize fails
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()

        # Remove stop words
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
        except:
            # If stopwords fail, just remove very common words
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were'}
            tokens = [word for word in tokens if word not in common_words]

        # Lemmatization with error handling
        try:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
        except:
            # Skip lemmatization if it fails
            pass

        # Join the tokens back into a string
        cleaned_text = ' '.join(tokens)

        # Ensure there's at least some content (if not, return a placeholder)
        if not cleaned_text.strip():
            return "empty_text"

        return cleaned_text
    except Exception as e:
        print(f"Error in text cleaning: {e}")
        return "error_text"


# Apply text cleaning to the TITLE column
df['cleaned_title'] = df['TITLE'].apply(clean_text)

# Encode the CATEGORY column
label_encoder = LabelEncoder()
df['encoded_category'] = label_encoder.fit_transform(df['CATEGORY'])

# Print the mapping of categories to encoded values
category_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Category encoding mapping:")
for category, code in category_mapping.items():
    if category == 'b':
        print(f"Business (b): {code}")
    elif category == 't':
        print(f"Science and Technology (t): {code}")
    elif category == 'e':
        print(f"Entertainment (e): {code}")
    elif category == 'm':
        print(f"Health (m): {code}")

# Check if the dataset is imbalanced
print("\nCategory distribution after preprocessing:")
print(df['CATEGORY'].value_counts())

# Optionally, balance the dataset using undersampling or oversampling
# Uncomment the following code if you want to balance the dataset
"""
# Get the count of the minority class
min_class_count = df['CATEGORY'].value_counts().min()

# Create balanced dataframe
balanced_df = pd.DataFrame()
for category in df['CATEGORY'].unique():
    category_df = df[df['CATEGORY'] == category]
    if len(category_df) > min_class_count:
        # Undersample
        category_df = resample(category_df, replace=False, n_samples=min_class_count, random_state=42)
    balanced_df = pd.concat([balanced_df, category_df])

# Shuffle the balanced dataframe
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
df = balanced_df
print("\nCategory distribution after balancing:")
print(df['CATEGORY'].value_counts())
"""

# 2. Feature Extraction
print("\nStep 2: Extracting features from text...")

# Split the dataset into training and testing sets
X = df['cleaned_title']
y = df['encoded_category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert text to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")

# 3. Train Classification Models
print("\nStep 3: Training classification models...")

# Option to skip or simplify GridSearchCV for faster execution
fast_mode = True  # Set to False for more thorough hyperparameter tuning

if fast_mode:
    # Logistic Regression with default parameters
    print("\nTraining Logistic Regression model with default parameters (fast mode)...")
    lr_best = LogisticRegression(C=1.0, solver='liblinear', max_iter=200, random_state=42)
    lr_best.fit(X_train_tfidf, y_train)

    # Random Forest with default parameters
    print("\nTraining Random Forest model with default parameters (fast mode)...")
    rf_best = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    rf_best.fit(X_train_tfidf, y_train)

    # MLP Neural Network with default parameters
    print("\nTraining MLP Neural Network model with default parameters (fast mode)...")
    mlp_best = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=200, random_state=42)
    mlp_best.fit(X_train_tfidf, y_train)
else:
    # Logistic Regression with GridSearchCV
    print("\nTraining Logistic Regression model with GridSearchCV...")
    lr_params = {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [200]
    }
    lr_grid = GridSearchCV(LogisticRegression(random_state=42), lr_params, cv=3, n_jobs=-1)
    lr_grid.fit(X_train_tfidf, y_train)
    lr_best = lr_grid.best_estimator_

    # Random Forest with GridSearchCV
    print("\nTraining Random Forest model with GridSearchCV...")
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, n_jobs=-1)
    rf_grid.fit(X_train_tfidf, y_train)
    rf_best = rf_grid.best_estimator_

    # MLP Neural Network with GridSearchCV
    print("\nTraining MLP Neural Network model with GridSearchCV...")
    mlp_params = {
        'hidden_layer_sizes': [(100,), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001],
        'max_iter': [200]
    }
    mlp_grid = GridSearchCV(MLPClassifier(random_state=42), mlp_params, cv=3, n_jobs=-1)
    mlp_grid.fit(X_train_tfidf, y_train)
    mlp_best = mlp_grid.best_estimator_

# 4. Model Evaluation
print("\nStep 4: Evaluating models...")


# Define a function to evaluate and display results for a model
def evaluate_model(model, X_test, y_test, model_name):
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{model_name} Accuracy: {accuracy:.4f}")

    # Display classification report
    print(f"\n{model_name} Classification Report:")
    class_labels = ['Business (b)', 'Entertainment (e)', 'Health (m)', 'Science and Technology (t)']
    print(classification_report(y_test, y_pred, target_names=class_labels))

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=class_labels, title=f'{model_name} Confusion Matrix')
    plot_confusion_matrix(cm, classes=class_labels, title=f'{model_name} Confusion Matrix (Normalized)', normalize=True)

    return accuracy, y_pred


# Evaluate each model
lr_accuracy, lr_pred = evaluate_model(lr_best, X_test_tfidf, y_test, "Logistic Regression")
rf_accuracy, rf_pred = evaluate_model(rf_best, X_test_tfidf, y_test, "Random Forest")
mlp_accuracy, mlp_pred = evaluate_model(mlp_best, X_test_tfidf, y_test, "MLP Neural Network")

# Compare models
print("\nModel Comparison:")
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"MLP Neural Network Accuracy: {mlp_accuracy:.4f}")

# Determine the best model
best_accuracy = max(lr_accuracy, rf_accuracy, mlp_accuracy)
if best_accuracy == lr_accuracy:
    best_model = lr_best
    best_model_name = "Logistic Regression"
elif best_accuracy == rf_accuracy:
    best_model = rf_best
    best_model_name = "Random Forest"
else:
    best_model = mlp_best
    best_model_name = "MLP Neural Network"

print(f"\nBest model: {best_model_name} with accuracy {best_accuracy:.4f}")

# 5. Model Improvements (Deep Learning - LSTM)
# Option to skip LSTM training for faster execution
train_lstm = False  # Set to True to train the LSTM model

if train_lstm:
    print("\nStep 5: Training a deep learning model (LSTM)...")

    # Tokenize and pad sequences for LSTM
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    # Determine maximum sequence length (you can adjust this)
    max_len = 30
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

    # Convert labels to categorical for deep learning
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=len(label_encoder.classes_))
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=len(label_encoder.classes_))

    # Build LSTM model
    lstm_model = Sequential([
        Embedding(input_dim=5000, output_dim=64, input_length=max_len),  # Smaller embedding dimension
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),  # Smaller LSTM size
        Dense(32, activation='relu'),  # Smaller dense layer
        Dropout(0.2),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])

    # Compile the model
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model (with early stopping to prevent overfitting)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    lstm_history = lstm_model.fit(
        X_train_pad, y_train_cat,
        epochs=3,  # Limited epochs for demonstration
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate LSTM model
    lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test_pad, y_test_cat)
    print(f"\nLSTM Model Accuracy: {lstm_accuracy:.4f}")

    # Get predictions
    lstm_pred_probs = lstm_model.predict(X_test_pad)
    lstm_pred = np.argmax(lstm_pred_probs, axis=1)

    # Display classification report for LSTM
    print("\nLSTM Classification Report:")
    class_labels = ['Business (b)', 'Entertainment (e)', 'Health (m)', 'Science and Technology (t)']
    print(classification_report(y_test, lstm_pred, target_names=class_labels))

    # Create confusion matrix for LSTM
    cm = confusion_matrix(y_test, lstm_pred)
    plot_confusion_matrix(cm, classes=class_labels, title='LSTM Confusion Matrix')
    plot_confusion_matrix(cm, classes=class_labels, title='LSTM Confusion Matrix (Normalized)', normalize=True)

    # Compare all models including LSTM
    print("\nFinal Model Comparison:")
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"MLP Neural Network Accuracy: {mlp_accuracy:.4f}")
    print(f"LSTM Model Accuracy: {lstm_accuracy:.4f}")

    # Update the best model if LSTM is better
    if lstm_accuracy > best_accuracy:
        best_accuracy = lstm_accuracy
        best_model_name = "LSTM"
        print(f"\nNew best model: {best_model_name} with accuracy {best_accuracy:.4f}")
    else:
        print(f"\nBest model remains: {best_model_name} with accuracy {best_accuracy:.4f}")
else:
    print("\nSkipping LSTM training for faster execution...")
    # Define variables needed for later parts of the code
    lstm_accuracy = 0
    tokenizer = None
    max_len = 30
    lstm_model = None

    # Compare traditional models only
    print("\nModel Comparison (excluding LSTM):")
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"MLP Neural Network Accuracy: {mlp_accuracy:.4f}")

# 6. Predictions on sample headlines
print("\nStep 6: Making predictions on sample headlines...")

sample_headlines = [
    "Tech giants invest heavily in AI research.",
    "A breakthrough in cancer treatment raises hopes worldwide.",
    "The global economy shows signs of recovery.",
    "New blockbuster movie breaks box office records."
]

# Clean the sample headlines
cleaned_samples = [clean_text(headline) for headline in sample_headlines]

# Make predictions using the best traditional model
# Transform the cleaned headlines into TF-IDF features
sample_tfidf = tfidf_vectorizer.transform(cleaned_samples)

# Make predictions
sample_preds = best_model.predict(sample_tfidf)

# Convert the encoded predictions back to category labels
sample_categories = label_encoder.inverse_transform(sample_preds)

# Print the predictions
print("\nPredictions using the best traditional model:")
for i, headline in enumerate(sample_headlines):
    category = sample_categories[i]
    category_name = 'Business' if category == 'b' else 'Science and Technology' if category == 't' else 'Entertainment' if category == 'e' else 'Health'
    print(f"Headline: \"{headline}\"")
    print(f"Predicted category: {category_name} ({category})\n")

# Make predictions using the LSTM model if it was trained
if train_lstm and lstm_model is not None:
    # Convert the cleaned headlines to sequences
    sample_seqs = tokenizer.texts_to_sequences(cleaned_samples)
    sample_pad = pad_sequences(sample_seqs, maxlen=max_len)

    # Make predictions
    sample_lstm_probs = lstm_model.predict(sample_pad)
    sample_lstm_preds = np.argmax(sample_lstm_probs, axis=1)

    # Convert the encoded predictions back to category labels
    sample_lstm_categories = label_encoder.inverse_transform(sample_lstm_preds)

    # Print the predictions
    print("\nPredictions using the LSTM model:")
    for i, headline in enumerate(sample_headlines):
        category = sample_lstm_categories[i]
        category_name = 'Business' if category == 'b' else 'Science and Technology' if category == 't' else 'Entertainment' if category == 'e' else 'Health'
        print(f"Headline: \"{headline}\"")
        print(f"Predicted category: {category_name} ({category})\n")

# Define expected categories for the sample headlines (for comparison)
expected_categories = [
    'Science and Technology (t)',  # "Tech giants invest heavily in AI research."
    'Health (m)',  # "A breakthrough in cancer treatment raises hopes worldwide."
    'Business (b)',  # "The global economy shows signs of recovery."
    'Entertainment (e)'  # "New blockbuster movie breaks box office records."
]

print("\nExpected categories for the sample headlines:")
for i, headline in enumerate(sample_headlines):
    print(f"Headline: \"{headline}\"")
    print(f"Expected category: {expected_categories[i]}\n")

print("\nAssignment completed!")