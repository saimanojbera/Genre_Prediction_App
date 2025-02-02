# Genre Prediction App

This project is a **Movie Genre Prediction App** that classifies movie descriptions as either **Horror** or **Romance** using **Machine Learning** and **Deep Learning** techniques. The app is built using **Streamlit** and leverages **Word2Vec embeddings** and a trained **TensorFlow model**.

## 🚀 Features
- **Predicts movie genres** (Horror/Romance) based on user-input descriptions.
- **Uses a trained neural network** to classify text.
- **Streamlit-based web app** for easy interaction.
- **Word2Vec embeddings** for text representation.
- **Spacy tokenizer** for text preprocessing.

---

## 📂 Project Structure
```
Genre_Prediction/
│── venv/                 # Virtual environment 
│── word2vec.model        # Word2Vec embeddings
│── spacy_model/          # Spacy tokenizer model
│── prediction.keras      # Trained TensorFlow model
│── requirements.txt      # Dependencies
│── streamlit_genre.py    # Main Streamlit app
│── README.md             # Project documentation
│── .gitignore            # Ignored files
```

---

## 🛠️ Setup & Installation

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/saimanojbera/Genre_Prediction_App.git
cd Genre_Prediction_App
```

### 2️⃣ **Create & Activate a Virtual Environment**
```bash
python -m venv venv
# Activate on Windows
venv\Scripts\activate
# Activate on macOS/Linux
source venv/bin/activate
```

### 3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4️⃣ **Run the Streamlit App**
```bash
streamlit run streamlit_genre.py
```

---

## 📌 How It Works
1. Enter a movie description in the text box.
2. Click the **Predict** button.
3. The app will classify the description as **Horror** or **Romance**, along with a confidence score.

---

## 🧩 Technologies Used
- **Python** (TensorFlow, Gensim, Spacy, Streamlit)
- **Machine Learning** (Neural Networks, Word2Vec)
- **Deep Learning** (Text classification with TensorFlow)
- **Streamlit** (Web app framework)

---

## 🔥 Example Movie Descriptions for Testing
| Movie Description | Expected Genre |
|------------------|---------------|
| "A woman moves into a haunted mansion and experiences eerie events every night." | Horror |
| "A young couple falls in love in a small town, despite their families’ objections." | Romance |
| "A grieving widow starts receiving love letters from her late husband." | Horror/Romance |

---

## 🤝 Contributing
Want to improve this project? Feel free to fork the repo and submit a pull request!

---

## 👨‍💻 Author
Developed by **Sai Manoj Bera**

