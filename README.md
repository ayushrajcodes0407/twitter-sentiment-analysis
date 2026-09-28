# Twitter Sentiment Analysis

> A production-ready Natural Language Processing application for analyzing sentiment in Twitter/X-style social media text using VADER, NLP preprocessing, and machine learning.

---

## 📌 Overview

**Twitter Sentiment Analysis** is an intelligent natural language processing and machine learning analytics platform designed to evaluate, classify, and visualize emotional polarity across social media text in real time.

Social media conversations represent a dynamic stream of public sentiment. Analyzing these interactions provides critical intelligence for:
- **Public Opinion Analysis:** Gauging community reactions to policy changes, major announcements, and global events.
- **Brand Perception & Reputation Management:** Monitoring brand mentions, product launches, and customer satisfaction.
- **Social Media Monitoring:** Tracking real-time social dynamics and sentiment shifts.
- **Customer Feedback Intelligence:** Automated triage of positive feedback versus critical support issues.
- **Market & Trend Research:** Understanding macro consumer sentiment and emerging social topics.
- **Text Classification Research:** Benchmarking rule-based lexicon heuristics against corpus-trained statistical models.

---

## ✨ Features

- **Single Tweet Sentiment Classification:** Real-time sentiment prediction (Positive 🟢, Negative 🔴, Neutral 🟡) with sub-second inference.
- **Emotional Polarity Scoring:** Compound sentiment polarity spectrum (`-1.0` to `+1.0`) with granular positive, negative, and neutral distribution percentages.
- **Confidence Calibration:** Calibrated confidence scoring reflecting model valence strength.
- **VADER Social NLP Engine:** Lexicon and rule-based sentiment intensity analysis optimized specifically for microblogging text, informal slang, abbreviations, and emojis.
- **Supervised ML Architecture:** Support for Logistic Regression classification trained on TF-IDF term vectors.
- **Text Preprocessing Pipeline:** Automatic Unicode sanitization, URL stripping, user handle elimination, punctuation removal, stopword filtering, and Porter stemming.
- **Batch CSV Analysis:** Bulk sentiment processing supporting CSV file uploads or multi-line text input with automated column detection (`text`, `tweet`, `content`).
- **Data Export & Reporting:** Downloadable annotated CSV datasets and formatted summary text reports.
- **Interactive Visualizations:** Polarity breakdown spectrums, sentiment distribution bar charts, and compound score density curves.
- **NLP Pipeline Transparency:** Step-by-step architectural breakdown of token normalization and stemming diagnostics.
- **Preloaded Example Presets:** One-click sample tweets covering positive, negative, neutral, and mixed sentiments.
- **Production-Grade API Security:** Zero-credential exposure with optional Streamlit Secrets / Environment Variable integration for live Twitter/X timeline lookups.
- **Luxury SaaS UI/UX:** Responsive, light-theme interface built with ivory, white, and champagne gold metallic aesthetics.

---

## 🧠 Machine Learning & NLP Architecture

The application implements a multi-stage Natural Language Processing pipeline:

```
Raw Social Media Text
        ↓
Regex Sanitization & Cleaning (URL, handle, hashtag, special characters)
        ↓
Case Normalization & Tokenization
        ↓
Stopwords Elimination (NLTK English Corpus)
        ↓
Porter Stemmer Root Word Reduction
        ↓
Feature Representation & Lexicon Valence Scoring (TF-IDF / VADER)
        ↓
Sentiment Classification & Polarity Calibration
        ↓
Confidence Score & Polarity Distribution
```

### Lexicon Heuristics (VADER) vs. Supervised Classification
- **Runtime Inference Engine:** The live interactive dashboard utilizes NLTK's **VADER (Valence Aware Dictionary and sEntiment Reasoner)**. VADER is specifically attuned to social media microtext, accurately interpreting capitalization emphasis (*"GREAT"* vs *"great"*), punctuation intensity (*"amazing!!!"*), emojis (*"🚀"*, *"😡"*), and complex negation handling (*"not bad"*).
- **Offline Training Notebook:** The included research notebook (`twittersentanalysis.ipynb`) demonstrates end-to-end supervised training using **TF-IDF vectorization** and **Logistic Regression** on 1.6 million tweets.

---

## 📊 Dataset

- **Corpus:** [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Total Records:** 1,600,000 annotated tweets
- **Target Polarity:** Binary classification (0 = Negative, 4 = Positive)
- **Features:** Target, Tweet ID, Timestamp, Flag, User Handle, Tweet Text

> *Note: Due to size constraints and licensing, the raw 1.6M CSV dataset is not bundled in this repository. The standalone application runs out-of-the-box using the bundled NLP engine and tokenizer.*

---

## 🏷️ Sentiment Classes

| Class | Compound Polarity Range | Semantic Meaning | Indicator |
| :--- | :---: | :--- | :---: |
| **Positive** | `Compound Score >= +0.05` | Favorable, enthusiastic, appreciative, or constructive | 🟢 |
| **Neutral** | `-0.05 < Compound Score < +0.05` | Objective, descriptive, factual, or balanced | 🟡 |
| **Negative** | `Compound Score <= -0.05` | Critical, frustrated, adverse, or dissatisfied | 🔴 |

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core application runtime |
| **Web Framework** | Streamlit | Reactive dashboard UI and component state management |
| **Natural Language Processing** | NLTK | VADER intensity analyzer, Porter stemmer, stopwords |
| **Machine Learning** | Scikit-Learn | TF-IDF vectorizer, Logistic Regression model |
| **Data Processing** | Pandas, NumPy | Dataframe manipulation, batch arrays, numerical matrices |
| **Scientific Computing** | SciPy | Sparse matrix calculations for ML features |
| **Design System** | Custom Vanilla CSS | Luxury editorial light-theme styling and responsive layouts |
| **Version Control** | Git | Semantic versioning and local repository lifecycle |

---

## 📂 Project Architecture

```
Twitter_Sentiment_Analysis/
├── .streamlit/
│   ├── config.toml                 # Light-theme configuration & server settings
│   └── secrets.toml.example        # Secrets template for optional Twitter API
├── docs/
│   └── screenshots/
│       └── dashboard.png           # Application UI preview
├── .env.example                    # Environment variables template
├── .gitignore                      # Git exclusion rules
├── app.py                          # Streamlit application entry point & dashboard
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Python package dependencies
└── twittersentanalysis.ipynb       # Model training & exploratory data analysis
```

---

## 🚀 Installation & Quickstart

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Twittersentimentanalysis
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Application
```bash
streamlit run app.py
```
The application will automatically open in your browser at `http://localhost:8501`.

---

## 🔐 Production API Configuration (Optional)

Core sentiment analysis and batch CSV processing operate **100% independently** without any API keys.

If you wish to configure live Twitter/X timeline querying, provide your developer keys via **Streamlit Secrets** or **Environment Variables**:

### Option A: Streamlit Secrets (Recommended)
Create `.streamlit/secrets.toml`:
```toml
TWITTER_CONSUMER_KEY = "your_consumer_key"
TWITTER_CONSUMER_SECRET = "your_consumer_secret"
TWITTER_ACCESS_TOKEN = "your_access_token"
TWITTER_ACCESS_TOKEN_SECRET = "your_access_token_secret"
```

### Option B: Environment Variables
Create a local `.env` file:
```env
TWITTER_CONSUMER_KEY=your_consumer_key
TWITTER_CONSUMER_SECRET=your_consumer_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret
```

> ⚠️ **Security Notice:** Never commit `.env` or `.streamlit/secrets.toml` to Git. Both files are strictly excluded in `.gitignore`.

---

## 💡 Usage Guide

### Single Tweet Analysis
1. Navigate to **Dashboard & Analysis**.
2. Type or paste any tweet, or select a preset from **Example Tweets**.
3. Click **Analyze Sentiment →**.
4. View the predicted classification, confidence score, compound polarity metric, and polarity distribution spectrum.
5. Click **Download Summary Report** to export text results.

### Batch CSV Processing
1. Navigate to **Batch Processing & CSV**.
2. Upload a CSV file (containing a column like `text`, `tweet`, or `content`) or paste multi-line tweets.
3. Review the automated summary metrics and distribution chart.
4. Click **Download Processed CSV Results** for full tabular export.

---

## 🖼️ Application Preview

```
[ Application Dashboard Screenshot ]
docs/screenshots/dashboard.png
```

---

## 👤 Author

**Ayush Raj**  
*B.Tech in Computer Science & Engineering*  
- **Email:** [ayushrajcodes0407@gmail.com](mailto:ayushrajcodes0407@gmail.com)  
- **LinkedIn:** [Add LinkedIn Profile URL]

---

## 🎯 Project Purpose

This project demonstrates the practical application of Natural Language Processing and Machine Learning to real-world social data. It bridges the gap between raw statistical text modeling and accessible decision intelligence through a responsive, human-centered SaaS user interface.

---

## ⚠️ Limitations

- **Context & Sarcasm:** Automated lexicon and statistical models may misclassify subtle sarcasm, rhetorical humor, or deeply nuanced slang.
- **Spelling & Noise:** Heavy typographical errors or unsegmented hashtags can diminish feature extraction fidelity.
- **Domain Adaptation:** While calibrated for microblogging, domain-specific terminology (e.g., specialized medical or legal discourse) may require custom fine-tuning.

---

## 🔮 Future Improvements

- [ ] Fine-tuned Transformer models (e.g., RoBERTa-Twitter / DistilBERT).
- [ ] Aspect-Based Sentiment Analysis (ABSA) for entity-level polarity extraction.
- [ ] Multilingual sentiment support across global languages.
- [ ] Real-time hashtag and trend velocity tracking.
- [ ] Explainable AI (XAI) attention heatmaps.

---

## 📄 License

License information to be added.
