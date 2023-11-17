import pickle
import pandas as pd
import imblearn
from nltk.corpus import stopwords
import contractions
import nltk
nltk.download('stopwords')
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

filtered_words = pd.read_csv("filtered_words.csv")["Words"].to_list()

model_path = 'best_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)




# Set up stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define a function to get the WordNet POS tag for a word
def get_wordnet_pos(word):
    # Map POS tag to the first character used by the WordNetLemmatizer
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag = {'N': 'n', 'V': 'v', 'R': 'r', 'J': 'a'}.get(tag, 'n')
    return tag

# Assuming you have initialized the lemmatizer and have filtered_words defined

def preprocess(text):
    # Expand contractions
    text = contractions.fix(text)

    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token))
              for token in tokens
              if token not in stop_words and token in filtered_words and token not in string.punctuation
              and len(token) > 2 and not any(c.isdigit() for c in token)]

    return tokens


def run_model(text):

    test = " ".join(preprocess("Test Review Here"))
    print(test)

    input = " ".join(preprocess(text))

    return model.predict([input])
