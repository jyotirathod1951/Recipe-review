import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load model
model = joblib.load("sentiment_model.pkl")

# Load data
df = pd.read_csv("Recipe Reviews and User Feedback Dataset.csv")
df = df.dropna(subset=['text', 'stars'])
df['Sentiment'] = df['stars'].apply(lambda x: 'Positive' if x >= 4 else 'Negative')

# App title
st.title("🍽 Recipe Review Sentiment Analysis App")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Raw Data", "Summary", "Graphs & Charts", "Sentiment Predictor"])

# Tab 1: Raw Dataset
with tab1:
    st.subheader("Raw Dataset")
    st.dataframe(df[['text', 'stars', 'Sentiment']].head(100))

# Tab 2: Summary
with tab2:
    st.subheader("Summary")
    st.write(df[['stars']].describe())

# Tab 3: Graphs & Charts
with tab3:
    st.subheader("Rating (Stars) Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='stars', ax=ax)
    st.pyplot(fig)

    st.subheader("Sentiment Distribution")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x='Sentiment', ax=ax2)
    st.pyplot(fig2)

# Tab 4: Sentiment Predictor
with tab4:
    st.subheader("Enter Review for Sentiment Prediction")
    user_input = st.text_area("Review:")
    user_star = st.slider(
        "Star Rating:",
        min_value=1, max_value=5,
        value=3, step=1
    )

    if st.button("Predict"):
        # Predict sentiment
        prediction = model.predict([user_input])
        sentiment = "👍 Positive" if prediction[0] == 1 else "👎 Negative"
        st.success(f"Predicted Sentiment: {sentiment}")
        st.info(f"Your Star Rating Input: {user_star}")

   
        
