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



# House Price Prediction Using Machine Learning

## Objective
To develop a machine learning model that predicts house prices based on features such as area, number of bedrooms, bathrooms, location, condition, and other property details using regression techniques.

## Steps Performed
1. Loaded the House Price Prediction dataset using Pandas.
2. Explored the dataset by checking its size, missing values, duplicates, and statistical summaries.
3. Cleaned the data by removing duplicates, handling missing values, and converting prices into numeric format.
4. Analyzed and visualized the distribution of house prices.
5. Selected relevant features and converted categorical variables into numerical values using One-Hot Encoding.
6. Performed correlation analysis and generated a correlation heatmap.
7. Split the dataset into 80% training and 20% testing data.
8. Trained a Linear Regression model and evaluated its performance using MSE, RMSE, and R² score.
9. Compared actual and predicted house prices using scatter plots and residual analysis.
10. Identified important features using regression coefficients.
11. Compared Linear Regression with Ridge Regression.
12. Developed an interactive input-based prediction system to estimate the price of a new house.

## Tools & Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Algorithms:** Linear Regression, Ridge Regression
- **Techniques:** Data Cleaning, Exploratory Data Analysis (EDA), One-Hot Encoding, Correlation Analysis, Regression, Model Evaluation
- **Evaluation Metrics:** MSE, RMSE, R² Score

## Outcome
Successfully developed a house price prediction system using Linear Regression and Ridge Regression. The models were evaluated and compared using regression performance metrics. The system also identifies influential features and predicts estimated house prices based on user-provided property details.
