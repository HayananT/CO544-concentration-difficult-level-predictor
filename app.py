# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

def user_input_features():
    # Create columns for two-column layout
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider('Age', 0, 50, 21)
        sex = st.selectbox('Sex', ['Male', 'Female', 'Non-binary'])
        relationship_status = st.selectbox('Relationship Status', ['Single', 'In a relationship', 'Married', 'Divorced'])
        occupation = st.selectbox('Occupation', ['University Student', 'School Student', 'Salaried Worker', 'Retired'])
        time_spent = st.slider('Time Spent on Social Media (hours)', 0.0, 5.0, 2.0)
        st.write('For the Follwing questions , Rate it on a scale of 1 to 5 based on its impact.')

        q1 = st.slider('How often do you find yourself using Social media without a specific purpose?', 1, 5, 3)
        q2 = st.slider('How often do you get distracted by Social media when you are busy doing something?', 1, 5, 3)

    with col2:
        q3 = st.slider('Do you feel restless if you haven\'t used Social media in a while?', 1, 5, 3)
        q4 = st.slider('How often do you look to seek validation from features of social media?', 1, 5, 3)
        q5 = st.slider('How much are you bothered by worries?', 1, 5, 3)
        q6 = st.slider('How often do you feel depressed or down?', 1, 5, 3)
        q7 = st.slider('How often do you compare yourself to other successful people through the use of social media?', 1, 5, 3)
        q8 = st.slider('Following the previous question, how do you feel about these comparisons, generally speaking?', 1, 5, 3)

    
    st.write('What platforms are you using in the following? If you are using that platform, then select Yes; if not, select No.')

    # Social media platforms usage
    col3, col4, col5 = st.columns(3)
    with col3: 
        facebook = st.selectbox("Facebook", options=["Yes", "No"])
        snapchat = st.selectbox("Snapchat", options=["Yes", "No"])
        reddit = st.selectbox("Reddit", options=["Yes", "No"])
    with col4:
        instagram = st.selectbox("Instagram", options=["Yes", "No"])
        twitter = st.selectbox("Twitter", options=["Yes", "No"])
        pinterest = st.selectbox("Pinterest", options=["Yes", "No"])
    with col5:
        youtube = st.selectbox("YouTube", options=["Yes", "No"])
        discord = st.selectbox("Discord", options=["Yes", "No"])
        tiktok = st.selectbox("TikTok", options=["Yes", "No"])

    def convert_to_numeric(value):
        return 1 if value == "Yes" else 0

    social_media_user = 1
    facebook = convert_to_numeric(facebook)
    instagram = convert_to_numeric(instagram)
    youtube = convert_to_numeric(youtube)
    snapchat = convert_to_numeric(snapchat)
    twitter = convert_to_numeric(twitter)
    discord = convert_to_numeric(discord)
    reddit = convert_to_numeric(reddit)
    pinterest = convert_to_numeric(pinterest)
    tiktok = convert_to_numeric(tiktok)

    data = {
        'Age': age,
        'Sex': sex,
        'Relationship Status': relationship_status,
        'Occupation': occupation,
        'Social Media User?': social_media_user,
        'Time Spent': time_spent,
        'Q1': q1,
        'Q2': q2,
        'Q3': q3,
        'Q4': q4,
        'Q5': q5,
        'Q6': q6,
        'Q7': q7,
        'Q8': q8,
        'Facebook': facebook,
        'Instagram': instagram,
        'YouTube': youtube,
        'Snapchat': snapchat,
        'Twitter': twitter,
        'Discord': discord,
        'Reddit': reddit,
        'Pinterest': pinterest,
        'TikTok': tiktok
    }
    features = pd.DataFrame(data, index=[0])
    return features

def main():
    st.title('Concentration Difficulty Prediction due to Social Media')
    st.write('Predicting Concentration Levels in the Social Media Age')
    st.write('In today\'s digital age, maintaining concentration can be challenging, especially with the pervasive influence of social media. The Concentration Difficulty Prediction Model assesses factors like age, occupation, and patterns of social media usage to predict concentration levels. By analyzing these inputs, the model forecasts potential concentration difficulties and highlights the impact of social media on focus and productivity. This empowers users to take proactive steps to improve concentration and achieve goals more effectively in the digital era.')

    input_df = user_input_features()

    if st.button('PREDICT'):
        # Load the model
        model = joblib.load('random_forest_model.joblib')

        # Convert categorical inputs to numerical using the same mappings as training
        label_encoders = {
            'Sex': {'Male': 0, 'Female': 1, 'Non-binary': 2},
            'Relationship Status': {'Single': 0, 'In a relationship': 1, 'Married': 2, 'Divorced': 3},
            'Occupation': {'University Student': 0, 'School Student': 1, 'Salaried Worker': 2, 'Retired': 3},
            'Social Media User?': {'Yes': 1, 'No': 0},
        }

        for col, mapping in label_encoders.items():
            input_df[col] = input_df[col].map(mapping)

        # Ensure all columns are in the same order as during training
        expected_columns = ['Age', 'Sex', 'Relationship Status', 'Occupation', 'Social Media User?', 'Time Spent', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8',
                            'Facebook', 'Instagram', 'YouTube', 'Snapchat', 'Twitter', 'Discord', 'Reddit', 'Pinterest', 'TikTok']

        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Assuming default value for missing columns

        input_df_imputed = pd.DataFrame(input_df, columns=expected_columns)  # Ensure columns are set correctly

        # Handle NaN values by replacing them with 0
        input_df_imputed = input_df_imputed.fillna(0)

        # Predict the difficulty level
        prediction = model.predict(input_df_imputed)[0]

        # Map prediction back to categorical value
        difficulty_mapping = {0: 'low', 1: 'medium', 2: 'high'}
        predicted_difficulty = difficulty_mapping[prediction]

        # Display predicted difficulty level
        st.subheader('ESTIMATED DIFFICULTY LEVEL STATUS')
        st.subheader(predicted_difficulty.upper())

        # Display notes based on predicted difficulty level
        if prediction == 0:
            st.info("Good! You have minimal social media distraction. Keep up the good work!")
        elif prediction == 1:
            st.warning("Caution! You seem to be moderately distracted by social media. Try to maintain better control to enhance your focus.")
        elif prediction == 2:
            st.error("Warning! You are highly distracted by social media. It's crucial to reduce distractions to improve your concentration.")

if __name__ == '__main__':
    main()
