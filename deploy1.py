import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from matplotlib import pyplot as plt

def prepreprocessing(text):
    text = text.lower() #converting text to lowercase
    text = re.sub(r'https?:\S+|www\S+', '', text, flags=re.MULTILINE)  #removing urls
    text = re.sub(r'\@\S+|\#', '', text)  #removing special characters
    text = re.sub(r'[^\w\s]', '', text)  #removing punctuations
    text = re.sub(r'\d+', '', text)#removing numbers
    text_tokens = word_tokenize(text)  # Tokenization
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in text_tokens if not word in stop_words] #removing stop words

    wnl = WordNetLemmatizer()
    lemma_words = [wnl.lemmatize(word) for word in filtered_tokens] #lemmatization

    return " ".join(lemma_words)

tfidf = pickle.load(open('vectorizer-1.pkl', 'rb'))
model = pickle.load(open('model-1.pkl', 'rb'))

st.title('Real and Fake News Classifier')

input_text = st.text_area('Enter the text')

if st.button('Predict'):
    # 1. Preprocess
    transformed_text = prepreprocessing(input_text)
 
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_text])

    # 3. Predict
    result = model.predict(vector_input)[0]
    probabilities = model.predict_proba(vector_input)[0]
    fake_probability = probabilities[0]  # Probability of being fake news
    real_probability = probabilities[1]  # Probability of being real news

    # 4. Display
    if result == 1:
        st.header('Real News')
        st.write(f"Confidence in prediction: {real_probability:.2%}")
    else:
        st.header('Fake News')
        st.write(f"Confidence in prediction: {fake_probability:.2%}")
        
    # Word Cloud Visualization
    wordcloud = WordCloud(width=3000, height=2000, background_color='teal', colormap='Pastel1').generate(transformed_text)
    plt.figure(figsize=(8, 5), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(plt)
 