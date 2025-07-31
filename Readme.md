# 📰 Fake News Detection System

A comprehensive machine learning pipeline for detecting fake news using NLP, data visualization, and multiple classification algorithms. This project demonstrates data exploration, feature engineering, and model evaluation on real-world news datasets.

---

## 📂 Project Structure

```
Fake-News-Detection-main/
│
├── fake-news-detection-sys.ipynb   # Main Jupyter Notebook (all code & analysis)
└── data/
    ├── Fake.zip                    # Compressed CSV of fake news
    └── True.zip                    # Compressed CSV of true news
```

---

## 🚀 Features

- **Data Exploration:** Word clouds, character/word count analysis, bigram frequency, and more.
- **Text Preprocessing:** Lemmatization, stopword removal, feature engineering.
- **Visualization:** Histograms, word clouds, and statistical plots for insightful EDA.
- **Machine Learning:** Logistic Regression, Naive Bayes, SVM, Random Forest with hyperparameter tuning.
- **Evaluation:** Cross-validation, accuracy, classification reports, and model comparison.

---

---

## 🛠️ How to Run

1. **Clone the repository:**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Or manually install: pandas, numpy, matplotlib, seaborn, scikit-learn, spacy, wordcloud, nltk)*

3. **Download spaCy model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Unzip datasets:**
   - Unzip `data/Fake.zip` and `data/True.zip` to get `Fake.csv` and `True.csv`.

5. **Run the notebook:**
   - Open `fake-news-detection-sys.ipynb` in Jupyter or VS Code and run all cells.

---

## 📁 Data

- **True.csv:** Real news articles.
- **Fake.csv:** Fake news articles.
- Each entry contains: `title`, `text`, `subject`, `date`.

---

## 🧠 Model Pipeline

1. **Preprocessing:** Merge title & text, lemmatize, remove stopwords.
2. **Feature Engineering:** Character/word counts, bigrams, TF-IDF.
3. **Model Training:** Multiple classifiers with cross-validation.
4. **Evaluation:** Accuracy, confusion matrix, classification report.

---

## 📈 Results & Insights

- Fake news tends to have longer titles and texts.
- Fake news uses more stopwords, possibly to mimic real articles.
- Random Forest and SVM yield the best accuracy after hyperparameter tuning.

---

## 📚 References

- [scikit-learn documentation](https://scikit-learn.org/)
- [spaCy documentation](https://spacy.io/)
- [NLTK documentation](https://www.nltk.org/)
- [Original dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

This project is licensed under the MIT License.

---

*Made with ❤️ for learning and research.*