import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the generator and the pretend "GAN discriminator" (actually the BiLSTM model)
generator = load_model('D:/ML-PROJECTS/Gan/GAN_Phishing_Email_Project-20241129T123229Z-001/GAN_Phishing_Email_Project/GAN Phishing/Models/generator.h5')
discriminator = load_model('D:/ML-PROJECTS/Gan/GAN_Phishing_Email_Project-20241129T123229Z-001/GAN_Phishing_Email_Project/GAN Phishing/Models/discriminator.h5')  # Using BiLSTM as discriminator

# Load tokenizers
with open('D:/ML-PROJECTS/Gan/GAN_Phishing_Email_Project-20241129T123229Z-001/GAN_Phishing_Email_Project/GAN Phishing/Models/tokenizergenerator.pkl', 'rb') as f:
    generator_tokenizer = pickle.load(f)

with open('D:/ML-PROJECTS/Gan/GAN_Phishing_Email_Project-20241129T123229Z-001/GAN_Phishing_Email_Project/GAN Phishing/Models/tokenizerdescriminator.pkl', 'rb') as f:
    bilstm_tokenizer = pickle.load(f)

# Parameters
latent_dim = 100
max_len = 150


# Function to generate an email using the generator
def generate_email():
    noise = np.random.normal(0, 1, (1, latent_dim))  # Generate random noise
    synthetic_email_embedding = generator.predict(noise)  # Generate embedding
    token_indices = np.argmax(synthetic_email_embedding[0], axis=-1)  # Map embedding to tokens
    generated_email = ' '.join(
        generator_tokenizer.index_word.get(idx, "<OOV>") for idx in token_indices if idx > 0
    )
    return generated_email


# Function to predict phishing status using the BiLSTM discriminator
def predict_email(email_text):
    # Preprocess the email text for the BiLSTM discriminator
    sequences = bilstm_tokenizer.texts_to_sequences([email_text])  # Convert text to sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')  # Pad to fixed length
    
    # Make prediction using the BiLSTM model
    prediction = discriminator.predict(padded_sequences)[0][0]  # Get the prediction score
    
    return "Phishing Email" if prediction > 0.5 else "Safe Email", prediction


# Streamlit app interface
st.title("GAN-Based Phishing Email Detection")

# Email Generation Section
st.header("Generate an Email")
if st.button("Generate Email"):
    generated_email = generate_email()
    st.subheader("Generated Email:")
    st.write(generated_email)

# Email Detection Section
st.header("Detect Email")
uploaded_email = st.text_area("Paste or type an email:")
if st.button("Detect Email"):
    if uploaded_email.strip():
        result, score = predict_email(uploaded_email)
        st.subheader("Detection Result:")
        st.write(f"Prediction: {result}")
        st.write(f"Confidence Score: {score:.2f}")
    else:
        st.error("Please input an email to detect.")

# Placeholder GAN Discriminator Architecture
st.header("GAN Discriminator Architecture")

# Style the app
st.markdown("---")
st.markdown("Developed using a GAN-based approach for phishing detection. The generator creates synthetic email data, and the discriminator predicts its authenticity.")
