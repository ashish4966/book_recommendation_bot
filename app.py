import streamlit as st
import numpy as np
import pickle
import random
import nltk
import requests
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import json
import os


# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')


model = load_model('chatbot_model.h5')


# Functions from your project
def scrape_goodreads(category):
    url = f"https://www.goodreads.com/search?utf8=%E2%9C%93&q={category}&search_type=books"
    req = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'})

    content = BeautifulSoup(req.content, 'html.parser')
    books = content.find_all('a', class_='bookTitle')
    authors = content.find_all('a', class_="authorName")

    result_html = "<h2>Search Results</h2>"
    for book, author in zip(books, authors):
        book_name = book.find('span', itemprop='name').text.strip()
        author_name = author.find('span', itemprop='name').text.strip()
        result_html += f"<p><strong>{book_name}</strong> by {author_name}</p>"

    return result_html

def clean_up_sentence(sentence):
    lemmatizer = WordNetLemmatizer()  # Declare lemmatizer here
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)

    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:

                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    
    return(np.array(bag))

# Load intents and populate the words list
words = []
classes = []
ignore_words = ['?', '!']

# intents = pickle.load(open('intents.pkl','rb'))
data_file = open('capstoneIndentPart1.json').read()
intents = json.loads(data_file)
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize and lemmatize each word in the pattern
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        # Add the tag of the intent to the classes list
        classes.append(intent['tag'])

# Lemmatize and remove duplicates from the words list
lemmatizer = WordNetLemmatizer()  # Declare lemmatizer here
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.3
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            print(tag)
            if tag == 'book_search':
                category = st.text_input("Sure, I'd be happy to recommend a book. What type of book are you in the mood for?")
                if category:
                    result = scrape_goodreads(category)
                else:
                    result = "Please enter a category."
            else:
                result = random.choice(i['responses'])
            break

    return result

def chatbot_response(msg):
    # Your existing chatbot code here
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

# Streamlit UI
st.title('Book Recommendation Chatbot')

# Sidebar for user input
option = st.sidebar.selectbox(
    'Select an action',
    ('Book Recommendation', 'Chat with the Bot')
)

if option == 'Book Recommendation':
    category = st.text_input('Enter the type of book you want to read:')
    if st.button('Get Recommendations'):
        if category:
            result = scrape_goodreads(category)
            st.markdown(result, unsafe_allow_html=True)
        else:
            st.warning('Please enter a category.')

elif option == 'Chat with the Bot':
    msg = st.text_input('You:', '')
    if st.button('Send'):
        if msg:
            response = chatbot_response(msg)
            st.text_area('Bot:', value=response, height=200, max_chars=None, key=None)
        else:
            st.warning('Please enter a message.')
