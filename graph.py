import pickle
import matplotlib.pyplot as plt

with open("scores_data.pkl", "rb") as f:
    X_lex, X_sem, y = pickle.load(f)

plt.figure(figsize=(8,5))
plt.scatter(X_sem, X_lex, c=y, cmap='coolwarm', alpha=0.5)
plt.xlabel("Semantic Similarity (%)")
plt.ylabel("Lexical Similarity (%)")
plt.title("Lexical vs. Semantic Similarity Distribution")
plt.colorbar(label='Label (1=Plagiarized)')
plt.savefig("lex_sem_plot.png")
plt.close()
