import os
import sys
import pandas as pd
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from preprocess import preprocess_text
from hashing import rolling_hash
from winnowing import winnow_hashes
from similarity import compute_similarity, compute_semantic_similarity

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def compute_lexical_semantic_scores(text1, text2, k=7, window_size=5):
    """
    Compute lexical and semantic scores for given text pair.

    Complexity:
    - Preprocessing: O(n)
    - Hashing for k-grams and winnowing: O(n)
    - Semantic embedding: O(n)
    Overall: O(n) for each text pair.

    n = length of the text in characters or tokens.
    """
    p1 = preprocess_text(text1)
    p2 = preprocess_text(text2)
    if len(p1) < k or len(p2) < k:
        lexical_score = 0.0
    else:
        hashes1 = [rolling_hash(p1[i:i+k]) for i in range(len(p1)-k+1)]
        hashes2 = [rolling_hash(p2[i:i+k]) for i in range(len(p2)-k+1)]
        fp1 = winnow_hashes(hashes1, window_size)
        fp2 = winnow_hashes(hashes2, window_size)
        lexical_score = compute_similarity(fp1, fp2)

    semantic_score = compute_semantic_similarity(p1, p2)
    return lexical_score, semantic_score

data_path = os.path.join("data", "pairs.csv")
df = pd.read_csv(data_path)
df = df.dropna(subset=["question1","question2"])
df = df.head(5000)  # reduce for faster experimentation if needed

X_lex = []
X_sem = []
y = []

print("Extracting features...")
for idx, row in df.iterrows():
    if idx % 500 == 0:
        print(f"Processing row {idx} of {len(df)}...")
    text1, text2, label = row['question1'], row['question2'], row['is_duplicate']
    lex_score, sem_score = compute_lexical_semantic_scores(text1, text2)
    X_lex.append(lex_score)
    X_sem.append(sem_score)
    y.append(label)

X = list(zip(X_lex, X_sem))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Comparing models...")

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
}

best_model_name = None
best_f1 = -1
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model = model

print(f"Best model based on F1 score: {best_model_name} (F1={best_f1:.4f})")

# Save the best model for final use
with open("plagiarism_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save data for plotting or analysis
with open("scores_data.pkl", "wb") as f:
    pickle.dump((X_lex, X_sem, y), f)

print("Training and comparison done.")
