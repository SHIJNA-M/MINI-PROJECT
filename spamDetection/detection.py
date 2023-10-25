import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import joblib
import os


# Run the below piece of code for the first time
# nltk.download('stopwords')


def stemmer(text):
    text = text.split()
    words = ""
    for i in text:
        stemmer = SnowballStemmer("english")
        words += (stemmer.stem(i))+" "
    return words


def text_preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower()
            not in stopwords.words('english')]
    return " ".join(text)


def check(message):

    loaded_model = joblib.load(os.path.join("spamDetection", 'spam_model.pkl'))
    loaded_vectorizer = joblib.load(os.path.join(
        "spamDetection", 'tfidf_vectorizer.pkl'))

    custom_email_text = message

    cleaned_email_text = text_preprocess(custom_email_text)
    cleaned_email_text = stemmer(cleaned_email_text)

    # Vectorize the preprocessed email text
    email_features = loaded_vectorizer.transform([cleaned_email_text])

    # Make a prediction
    prediction = loaded_model.predict(email_features)

    print(prediction)

    return prediction[0]
