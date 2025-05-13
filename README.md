# ham-or-spam-detector

import streamlit as st import pandas as pd from sklearn.feature_extraction.text import CountVectorizer from sklearn.model_selection import train_test_split from sklearn.naive_bayes import MultinomialNB from sklearn.metrics import accuracy_score

**Load dataset**

messages = pd.read_csv('spam.csv', sep='\t', names=['LABEL', 'MESSAGES']) messages.columns = ['label', 'message']

**Preprocess data**

messages['label'] = messages['label'].map({'ham': 0, 'spam': 1})

**Split data**

X_train, X_test, y_train, y_test = train_test_split(messages['message'], messages['label'], test_size=0.2, random_state=42)

**Vectorize text**

vectorizer = CountVectorizer() X_train_vec = vectorizer.fit_transform(X_train) X_test_vec = vectorizer.transform(X_test)

**Train model**

model = MultinomialNB() model.fit(X_train_vec, y_train)

**Evaluate model**

predictions = model.predict(X_test_vec) accuracy = accuracy_score(y_test, predictions)

**Streamlit web app**

st.title('Spam Detection Web App') st.write(f'Model Accuracy: {accuracy * 100:.2f}%')

user_input = st.text_area('Enter a message:') if st.button('Predict'): if user_input: input_vec = vectorizer.transform([user_input]) prediction = model.predict(input_vec)[0] st.write('Prediction:', 'Spam' if prediction == 1 else 'Ham') else: st.write('Please enter a message.')
