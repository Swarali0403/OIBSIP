# Autocomplete and Autocorrect System Using NLP

## Objective
To develop an NLP-based autocomplete and autocorrect system that predicts the next word based on previous words and identifies and corrects spelling mistakes using different techniques.

## Steps Performed
1. Loaded the Enron Email Dataset using KaggleHub.
2. Preprocessed email text by converting it to lowercase, removing stopwords, and extracting words.
3. Analyzed and visualized the 20 most frequently occurring words.
4. Implemented Bigram and Trigram models for next-word prediction.
5. Evaluated autocomplete performance using precision and recall.
6. Developed two autocorrect approaches: PySpellChecker and Custom Edit Distance.
7. Tested both autocorrect methods using 20 commonly misspelled words.
8. Compared model performance using accuracy metrics, confusion matrices, and visualizations.

## Tools & Technologies Used
- **Programming Language:** Python
- **Dataset:** Enron Email Dataset (Kaggle)
- **Libraries:** Pandas, NLTK, Matplotlib, Scikit-learn, PySpellChecker
- **Other Modules:** Regex, Collections, KaggleHub
- **Techniques:** Natural Language Processing (NLP), Bigram, Trigram, Edit Distance, Spell Checking

## Outcome
Successfully developed an autocomplete and autocorrect system using NLP techniques. The autocomplete models generate up to three next-word suggestions using Bigram and Trigram approaches. The autocorrect component compares PySpellChecker with a custom Edit Distance algorithm to identify and correct spelling errors. Model performance is evaluated using accuracy, precision, recall, confusion matrices, and graphical comparisons.

The project demonstrates the application of basic NLP techniques in text prediction and spelling correction.
