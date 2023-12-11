import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import nltk
from nltk.stem.lancaster import LancasterStemmer

# Define a SessionState class to store variables
class SessionState:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Function to cleanup text
def cleanup(sentence):
    word_tok = nltk.word_tokenize(sentence)
    stemmed_words = [stemmer.stem(w) for w in word_tok]
    return ' '.join(stemmed_words)

# Initialize Streamlit session state
session_state = SessionState(cnt=0)

stemmer = LancasterStemmer()
le = LE()
tfv = TfidfVectorizer(min_df=1, stop_words='english')

data = pd.read_csv('BankFAQs.csv')
questions = data['Question'].values

X = []
for question in questions:
    X.append(cleanup(question))

tfv.fit(X)
le.fit(data['Class'])

X = tfv.transform(X)
y = le.transform(data['Class'])

trainx, testx, trainy, testy = tts(X, y, test_size=.25, random_state=42)

model = DecisionTreeClassifier()
model.fit(trainx, trainy)
st.title("Bank FAQ Bot")

# Function to get top 5 indices
def get_max5(arr):
    ixarr = [(el, ix) for ix, el in enumerate(arr)]
    ixarr.sort()
    ixs = [i[1] for i in ixarr[-5:]]
    return ixs[::-1]

# Function to handle chat
def chat():
    TOP5 = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if usr := st.chat_input("You:"):
        st.chat_message("user").markdown(usr)
        
        st.session_state.messages.append({"role": "user", "content": usr})

        if usr is not None:
            t_usr = tfv.transform([cleanup(usr.strip().lower())])
            class_ = le.inverse_transform(model.predict(t_usr))
            questionset = data[data['Class'] == class_[0] if class_ else '']

            cos_sims = []
            for question in questionset['Question']:
                sims = cosine_similarity(tfv.transform([question]), t_usr)
                cos_sims.append(sims)

            ind = cos_sims.index(max(cos_sims))

            if not TOP5:
                bot_response = f"Bot: {data['Answer'][questionset.index[ind]]}"
                st.write(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
            else:
                inds = get_max5(cos_sims)
                for ix in inds:
                    bot_response = f"Question: {data['Question'][questionset.index[ix]]}\nAnswer: {data['Answer'][questionset.index[ix]]}\n{'-' * 50}"
                    st.write(bot_response)
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})

            st.write("\n" * 2)
            outcome = st.chat_input("Was this answer helpful? Yes/No: ")
            if outcome in ['yes', 'y']:
                cnt = 0
            elif outcome in ['no', 'n']:
                inds = get_max5(cos_sims)
                sugg_choice = st.chat_input("Bot: Do you want me to suggest you questions? Yes/No: ")
                if sugg_choice in ['yes', 'y']:
                    q_cnt = 1
                    for ix in inds:
                        st.write(q_cnt, f"Question: {data['Question'][questionset.index[ix]]}\n{'-' * 50}")


if __name__ == "__main__":
    chat()
