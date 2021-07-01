from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sentone import sentiment_s 
import streamlit as st

filename = 'linearsvc1600k001.sav'
clf_loaded_model = pickle.load(open("linearsvc1600k001.sav", 'rb'))
tfidf_var = TfidfVectorizer()

#print("heyefw")
#sentiment_s("fuck this")
#print("hey")
st.title("Sentiment Analysis ")

#sentiment 
if st.checkbox("Get Started"):
    st.subheader("Hmmm..")
    message = st.text_area("Enter your text","Type here")
    if st.button('Analyze'):
        c=sentiment_s(message)
        st.success(c)