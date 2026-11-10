
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Download stopwords
nltk.download("stopwords", quiet=True)

# Load dataset
#df = pd.read_csv(
 #   "enron.csv",
  #  encoding="utf-8",
   # encoding_errors="replace",
    #on_bad_lines="skip"
#)
# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = ""

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "wcukierski/enron-email-dataset",
  file_path,
  # Provide any additional arguments like
  # sql_query or pandas_kwargs. See the
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())

print("\nDataset size:", df.shape)


# TEXT PREPROCESSING


df["Message"] = df["Message"].fillna("").astype(str)

stop_words = set(stopwords.words("english"))

word_count = Counter()
clean_words = []

# Process only the first 10000 emails
max_emails = min(10000, len(df))

for msg in df["Message"].iloc[:max_emails]:
    small_words = re.findall(r"[a-z]+", msg.lower())

    small_words = [
        word for word in small_words
        if word not in stop_words
    ]

    clean_words.extend(small_words)
    word_count.update(small_words)

print("\nEmails processed:", max_emails)
print("Total cleaned words:", len(clean_words))
print("Different words:", len(word_count))
print("Sample cleaned words:", clean_words[:30])


# MOST FREQUENT WORDS


top_words = word_count.most_common(20)

print("\nTop 20 most frequent words:")

for word, count in top_words:
    print(word, ":", count)

plt.figure(figsize=(12, 5))
plt.bar(
    [x[0] for x in top_words],
    [x[1] for x in top_words]
)
plt.title("Top 20 Most Frequent Words")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# AUTOCOMPLETE MODELS

# Split words into training and testing parts
cut = int(len(clean_words) * 0.80)

train_words = clean_words[:cut]
test_words = clean_words[cut:]

# Build bigram model
bigram = defaultdict(Counter)

for i in range(len(train_words) - 1):
    bigram[train_words[i]][train_words[i + 1]] += 1

# Build trigram model
trigram = defaultdict(Counter)

for i in range(len(train_words) - 2):
    pair = (train_words[i], train_words[i + 1])
    trigram[pair][train_words[i + 2]] += 1

print("\nBigram and trigram models created.")

# Predict the next word using one previous word
def bigram_predict(prefix):
    return [
        word for word, count
        in bigram.get(prefix.lower(), Counter()).most_common(3)
    ]

# Predict the next word using two previous words
def trigram_predict(prefix):
    pair = tuple(prefix.lower().split()[-2:])

    return [
        word for word, count
        in trigram.get(pair, Counter()).most_common(3)
    ]

# Test 10 prefixes from the training vocabulary
print("\nAUTOCOMPLETE TEST RESULTS")
print("-" * 55)

test_prefixes = [
    "business", "meeting", "please", "thank", "market",
    "company", "power", "energy", "report", "email"
]

for prefix in test_prefixes:
    print("\nPrefix:", prefix)
    print("Bigram top 3:", bigram_predict(prefix))

    # Find a common second word for the trigram test
    next_words = bigram_predict(prefix)

    if next_words:
        two_word_prefix = prefix + " " + next_words[0]
        print("Trigram prefix:", two_word_prefix)
        print("Trigram top 3:", trigram_predict(two_word_prefix))
    else:
        print("Trigram top 3: No prediction available")


# AUTOCOMPLETE EVALUATION

bigram_hits = 0
bigram_total = 0
bigram_suggestions = 0

# Evaluate on held-out words
for i in range(cut + 1, len(clean_words)):
    prefix = clean_words[i - 1]
    actual_word = clean_words[i]

    guesses = bigram_predict(prefix)

    if guesses:
        bigram_total += 1
        bigram_suggestions += len(guesses)

        if actual_word in guesses:
            bigram_hits += 1

    if bigram_total >= 1000:
        break

trigram_hits = 0
trigram_total = 0
trigram_suggestions = 0

for i in range(cut + 2, len(clean_words)):
    pair = (clean_words[i - 2], clean_words[i - 1])
    actual_word = clean_words[i]

    guesses = [
        word for word, count
        in trigram.get(pair, Counter()).most_common(3)
    ]

    if guesses:
        trigram_total += 1
        trigram_suggestions += len(guesses)

        if actual_word in guesses:
            trigram_hits += 1

    if trigram_total >= 1000:
        break

def show_auto_results(name, hits, total, suggestions):
    precision = hits / suggestions if suggestions else 0
    recall = hits / total if total else 0

    print("\n" + name)
    print("Correct top-3 predictions:", hits)
    print("Test cases:", total)
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall:", round(recall * 100, 2), "%")

show_auto_results(
    "Bigram Model",
    bigram_hits,
    bigram_total,
    bigram_suggestions
)

show_auto_results(
    "Trigram Model",
    trigram_hits,
    trigram_total,
    trigram_suggestions
)


# AUTOCORRECT MODELS


wrong_words = [
    "busines", "meetng", "presntation", "travling",
    "productiv", "discusson", "suggeston", "necesary",
    "tomorow", "recieve", "managment", "schedul",
    "employe", "departmant", "finacial", "commitee",
    "agreemant", "developement", "importent", "oppertunity"
]

right_words = [
    "business", "meeting", "presentation", "traveling",
    "productive", "discussion", "suggestion", "necessary",
    "tomorrow", "receive", "management", "schedule",
    "employee", "department", "financial", "committee",
    "agreement", "development", "important", "opportunity"
]

# Keep the most common 15000 words from the email corpus
vocab = set(word for word, count in word_count.most_common(15000))

# Add correct test words so both methods have candidates
vocab.update(right_words)

spell = SpellChecker()

for word in vocab:
    spell.word_frequency.add(word)

# M1: PySpellChecker
def spell_correct(word):
    answer = spell.correction(word)
    return answer if answer else word

# M2 :Custom edit distance
def edit_distance(a, b):
    row = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        new_row = [i]

        for j, char_b in enumerate(b, start=1):
            change = 0 if char_a == char_b else 1

            new_row.append(min(
                new_row[j - 1] + 1,
                row[j] + 1,
                row[j - 1] + change
            ))

        row = new_row

    return row[-1]

# Check only words of similar length
def my_correct(word):
    possible = [
        w for w in vocab
        if abs(len(w) - len(word)) <= 2
    ]

    if not possible:
        return word

    best_word = min(
        possible,
        key=lambda w: (
            edit_distance(word, w),
            -word_count.get(w, 0)
        )
    )

    if edit_distance(word, best_word) <= 2:
        return best_word

    return word

# Test the two methods
spell_right = []
my_right = []

spell_answers = []
my_answers = []

print("\nAUTOCORRECT TEST RESULTS")

print(f"{'Wrong Word':<18}{'SpellChecker':<22}{'Custom Method'}")


for wrong, correct in zip(wrong_words, right_words):
    ans1 = spell_correct(wrong)
    ans2 = my_correct(wrong)

    spell_answers.append(ans1)
    my_answers.append(ans2)

    spell_right.append(ans1 == correct)
    my_right.append(ans2 == correct)

    print(f"{wrong:<18}{ans1:<22}{ans2}")


# AUTOCORRECT METRICS


def show_spell_results(name, predictions):
    accuracy = sum(predictions) / len(predictions)

    print("\n" + name)
    print("Correct corrections:",
          sum(predictions), "out of", len(predictions))
    print("Accuracy:", round(accuracy * 100, 2), "%")

    return accuracy * 100

spell_accuracy = show_spell_results(
    "PySpellChecker",
    spell_right
)

my_accuracy = show_spell_results(
    "Custom Edit Distance",
    my_right
)

# For this test, every input is a typo and the correct
# intended word is known. Exact-match precision and recall
# are reported as successful corrections divided by tests.
print("\nPrecision and recall for this typo-only test:")

print("PySpellChecker precision:",
      round(spell_accuracy, 2), "%")
print("PySpellChecker recall:",
      round(spell_accuracy, 2), "%")

print("Custom method precision:",
      round(my_accuracy, 2), "%")
print("Custom method recall:",
      round(my_accuracy, 2), "%")


# CONFUSION MATRICES


# 1 means the corrected word matches the expected word
# 0 means the corrected word does not match
actual = [1] * len(right_words)

before = [0] * len(right_words)
pred1 = [int(x) for x in spell_right]
pred2 = [int(x) for x in my_right]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for ax, title, pred in [
    (axes[0], "Before Correction", before),
    (axes[1], "PySpellChecker", pred1),
    (axes[2], "Custom Edit Distance", pred2)
]:
    cm = confusion_matrix(actual, pred, labels=[0, 1])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not matched", "Matched"]
    )

    display.plot(ax=ax, colorbar=False)
    ax.set_title(title)

plt.tight_layout()
plt.show()


# COMPARE AUTOCORRECT METHODS


plt.figure(figsize=(7, 4))

plt.bar(
    ["PySpellChecker", "Custom Edit Distance"],
    [spell_accuracy, my_accuracy]
)

plt.ylabel("Accuracy (%)")
plt.title("Autocorrect Accuracy Comparison")
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

print("\nFINAL COMPARISON")
print("PySpellChecker accuracy:",
      round(spell_accuracy, 2), "%")
print("Custom Edit Distance accuracy:",
      round(my_accuracy, 2), "%")


# PROJECT CONCLUSION


print("\nPROJECT CONCLUSION")

print(
    "The bigram model predicts words using one previous word, "
    "while the trigram model uses two previous words."
)

print(
    "PySpellChecker uses a spelling dictionary, while the "
    "custom method uses edit distance to find similar words."
)

print(
    "These models are simple and useful for learning NLP, "
    "but they do not understand context as well as production "
    "systems like Google Keyboard."
)

print(
    "This analysis uses a sample of emails to reduce memory "
    "usage. Results may change when more emails are processed."
)
