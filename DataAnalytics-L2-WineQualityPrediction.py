
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# Load dataset
df = pd.read_csv("WineQT.csv")

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

# Remove missing and duplicate rows
df = df.dropna()
df = df.drop_duplicates()

# Remove Id column
if "Id" in df.columns:
    df = df.drop("Id", axis=1)

# Check wine quality count
print("\nWine Quality Count:")
print(df["quality"].value_counts().sort_index())

df["quality"].value_counts().sort_index().plot(
    kind="bar", color="orange"
)
plt.title("Wine Quality Distribution")
plt.xlabel("Quality Score")
plt.ylabel("Number of Wines")
plt.tight_layout()
plt.show()

# Show chemical feature distributions
df.drop("quality", axis=1).hist(
    figsize=(12, 8), color="skyblue"
)
plt.suptitle("Chemical Features Distribution")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Group quality scores
def group_quality(q):
    if q <= 4:
        return "Low"
    elif q <= 6:
        return "Medium"
    else:
        return "High"

df["quality_group"] = df["quality"].apply(group_quality)

print("\nQuality Group Count:")
print(df["quality_group"].value_counts())

# Select input and output
X = df.drop(["quality", "quality_group"], axis=1)
y = df["quality_group"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale data for SGD and SVC
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create three models
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42,
        class_weight="balanced"
    ),
    "SGD Classifier": SGDClassifier(
        max_iter=1000, random_state=42,
        class_weight="balanced"
    ),
    "SVC": SVC(class_weight="balanced")
}

results = []

# Train and test each model
for name, model in models.items():

    if name == "Random Forest":
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, pred)

    print("\nModel:", name)
    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(
        y_test, pred, zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(
        y_test, pred, labels=["Low", "Medium", "High"]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Low", "Medium", "High"]
    )

    display.plot(cmap="Blues")
    plt.title(name + " Confusion Matrix")
    plt.tight_layout()
    plt.show()

    results.append([name, round(accuracy * 100, 2)])

# Feature importance of Random Forest
importance = pd.Series(
    models["Random Forest"].feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

importance.plot(kind="bar", color="green")
plt.title("Random Forest Feature Importance")
plt.xlabel("Chemical Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nFeature Importance:")
print(importance)

importance.plot(kind="bar", color="green")
plt.title("Random Forest Feature Importance")
plt.xlabel("Chemical Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Compare all models
result = pd.DataFrame(
    results, columns=["Model", "Accuracy (%)"]
)

print("\nModel Comparison:")
print(result.to_string(index=False))

# Find the best model based on accuracy
best = result.loc[result["Accuracy (%)"].idxmax()]

print("\nBest Model:", best["Model"])
print("Accuracy:", best["Accuracy (%)"], "%")

print("\nConclusion:")
print("The best model is selected based on test accuracy.")
print("Precision, recall and F1-score should also be checked.")
