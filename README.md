# Fake-news-detection-using-AI
📰 Fake News Detection using LSTM and GloVe Embeddings
This project is a machine learning-based fake news detector that uses Natural Language Processing (NLP) techniques to classify news headlines as real or fake. It is built using TensorFlow, LSTM networks, and pre-trained GloVe word embeddings.
<br>
🔍 Project Overview
Fake news has become a major problem in the digital age. This project aims to provide an automated way to detect the authenticity of news headlines using deep learning.
<br>
The model takes a news title as input and classifies it as either:
<br>
✅ True News
<br>
❌ Fake News
<br>
🧠 Technologies & Tools Used
Python
<br>
Pandas & NumPy – Data preprocessing
<br>
TensorFlow/Keras – Model building and training
<br>
Scikit-learn – Label encoding and data splitting
<br>
GloVe (Global Vectors) – Pre-trained word embeddings
<br>
LSTM (Long Short-Term Memory) – Deep learning architecture
<br>
CNN (Conv1D) – Feature extraction from word sequences
<br>
📂 Dataset
The dataset contains news titles, text, and labels.
<br>
It is processed by:
<br>
Removing unnecessary columns
<br><br>
Encoding labels (True = 1, Fake = 0)
<br>
Tokenizing titles
<br>
Padding sequences
<br>
Embedding using GloVe vectors
<br>

🏗️ Model Architecture<br>
The model includes:
<br>
Embedding layer with pre-trained GloVe vectors
<br>
Dropout layer to prevent overfitting
<br>
Conv1D + MaxPooling for pattern extraction
<br>
LSTM layer to understand word context
<br>
Dense (Sigmoid) output layer for binary classification
