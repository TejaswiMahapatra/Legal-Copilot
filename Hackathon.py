import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score


# Create a DataFrame from the dataset
df = pd.read_csv('dataset (1).csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['User Input'], df['Service Category'], test_size=0.2,
                                                    random_state=42)

# Initialize a CountVectorizer
count_vectorizer = CountVectorizer()

# Fit and transform the training data
X_train_counts = count_vectorizer.fit_transform(X_train)

# Initialize a Multinomial Naive Bayes classifier
classifier = MultinomialNB()

# Train the classifier
classifier.fit(X_train_counts, y_train)

# Transform the test data using the same vectorizer
X_test_counts = count_vectorizer.transform(X_test)

# Make predictions on the test data
y_pred = classifier.predict(X_test_counts)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)


# print("Accuracy:", accuracy)
# print("Classification Report:")
# print(report)


# Function to make recommendations
def make_recommendations(user_input):
    # Vectorize user input using the same CountVectorizer
    user_input_vector = count_vectorizer.transform([user_input])

    # Predict the service category
    prediction = classifier.predict(user_input_vector)

    # Return the recommended service category
    return prediction[0]


# Sample user interaction loop
while True:
    user_input = input("Please describe your needs (type 'exit' to quit): ")

    if user_input.lower() == 'exit':
        break

    recommendation = make_recommendations(user_input)
    print(f"Based on your needs, you should consider services from: {recommendation}")
