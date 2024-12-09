import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import joblib
# Load pre-trained models
@st.cache_resource
def load_models():
    generator = load_model("C:/Users/N.NAGA DEEPTHI/Downloads/GAN Phishing/GAN Phishing/Models/generator.h5")
    discriminator= load_model("C:/Users/N.NAGA DEEPTHI/Downloads/GAN Phishing/GAN Phishing/Models/discriminator.h5")
    return generator, discriminator

@st.cache_resource
def load_desc_model():
    return joblib.load('C:/Users/N.NAGA DEEPTHI/Downloads/GAN Phishing/GAN Phishing/discriminator.pkl')

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    with open("./Models/tokenizerdescriminator.pkl", "rb") as file:
        tokenizer = pickle.load(file)
    return tokenizer

# Generate synthetic email using the generator
def generate_email(generator, tokenizer, latent_dim=100):
    noise = np.random.normal(0, 1, (1, latent_dim))
    synthetic_email = generator.predict(noise)[0]
    word_index = tokenizer.word_index
    index_word = {v: k for k, v in word_index.items()}
    email = ' '.join(index_word.get(np.argmax(word_vector), "<OOV>") for word_vector in synthetic_email)
    return email

# Detect fake email using BiLSTM
def detect_fake_email(email, descriminator, tokenizer, max_length=150):
    sequence = tokenizer.texts_to_sequences([email])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding="post")
    st.write("Analyzing email with Discriminator...")
    print(padded_sequence)
    prediction = descriminator.predict([email])
    print(prediction)
    prediction_label = "FAKE" if prediction[0] == 'Phishing Email' else "SAFE"
    return prediction_label
    # print(padded_sequence)
    # predictions = descriminator.predict(email)
    return prediction_label
    

# Streamlit App UI
def main():
    st.title("Phishing Email Detection")
    generator, _ = load_models()
    descriminator = load_desc_model()
    tokenizer = load_tokenizer()

    st.header("Generate Synthetic Email")
    if st.button("Generate Email"):
        email = generate_email(generator, tokenizer)
        st.text_area("Generated Email", email, height=200)
    
    st.header("Detect Fake Email")
    user_email = st.text_area("Enter Email Text", height=200)
    if st.button("Analyze Email"):
        if user_email.strip():
            result = detect_fake_email(user_email, descriminator, tokenizer, max_length=150)
            st.info(f"The email is likely: {result}")
        else:
            st.warning("Please enter an email to analyze.")

if __name__ == "__main__":
    main()


















# import streamlit as st
# import numpy as np
# import pickle
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import Tokenizer



# # Load pre-trained models
# @st.cache_resource
# def load_models():
#     generator = load_model("./Models/generator.h5")
#     bilstm_model = load_model("./Models/discriminator.h5")
#     return generator, bilstm_model

# # Load tokenizer
# @st.cache_resource
# def load_tokenizer():
#     with open("./Models/tokenizergenerator.pkl", "rb") as file:
#         tokenizer = pickle.load(file)
#     return tokenizer

# # Generate synthetic email using the generator
# def generate_email(generator, tokenizer, latent_dim=100):
#     noise = np.random.normal(0, 1, (1, latent_dim))
#     synthetic_email = generator.predict(noise)[0]
#     word_index = tokenizer.word_index
#     index_word = {v: k for k, v in word_index.items()}
#     email = ' '.join(index_word.get(np.argmax(word_vector), "<OOV>") for word_vector in synthetic_email)
#     return email

# # Detect fake email using BiLSTM
# def detect_fake_email(email, bilstm_model, tokenizer, max_length=150):
#     sequence = tokenizer.texts_to_sequences([email])
#     padded_sequence = pad_sequences(sequence, maxlen=max_length, padding="post")
#     prediction_prob = bilstm_model.predict(padded_sequence)[0][0]
#     is_fake = prediction_prob > 0.5
#     return is_fake, prediction_prob

# # Streamlit App UI
# def main():
#     st.title("Phishing Email Detection")
#     generator, bilstm_model = load_models()
#     tokenizer = load_tokenizer()

#     st.header("Generate Synthetic Email")
#     if st.button("Generate Email"):
#         email = generate_email(generator, tokenizer)
#         st.text_area("Generated Email", email, height=200)
    
#     st.header("Detect Fake Email")
#     user_email = st.text_area("Enter Email Text", height=200)
#     if st.button("Analyze Email"):
#         if user_email.strip():
#             is_fake, prediction_prob = detect_fake_email(user_email, bilstm_model, tokenizer)
#             if is_fake:
#                 st.error(f"The email is likely FAKE. (Confidence: {prediction_prob:.2f})")
#             else:
#                 st.success(f"The email is likely SAFE. (Confidence: {prediction_prob:.2f})")
#         else:
#             st.warning("Please enter an email to analyze.")

# if __name__ == "__main__":
#     main()
