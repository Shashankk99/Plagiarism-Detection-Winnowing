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
from xgboost import XGBClassifier

def compute_lexical_semantic_scores(text1, text2, k=7, window_size=5):
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
df = df.head(10000)

X_lex = []
X_sem = []
y = []

for idx, row in df.iterrows():
    if idx % 200 == 0:
        print(f"Processing row {idx} of {len(df)}...")
    text1, text2, label = row['question1'], row['question2'], row['is_duplicate']
    lex_score, sem_score = compute_lexical_semantic_scores(text1, text2)
    X_lex.append(lex_score)
    X_sem.append(sem_score)
    y.append(label)

X = list(zip(X_lex, X_sem))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training XGBoost model...")
model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

print("Evaluating model...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Test Accuracy:", acc)
print("Test F1 Score:", f1)

with open("plagiarism_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Done.")

# Save data for plotting
with open("scores_data.pkl", "wb") as f:
    pickle.dump((X_lex, X_sem, y), f)
