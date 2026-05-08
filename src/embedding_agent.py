# -------------------------------
# 2. Word Embeddings: Word2Vec & SBERT
# -------------------------------
print("\nComputing SBERT embeddings...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
sbert_embeddings = sbert_model.encode(texts, show_progress_bar=True)

# ✅ Save embeddings for Task 3
import numpy as np
sbert_embeddings = np.array(sbert_embeddings, dtype=np.float32)  # ensure float32
np.save("/content/drive/MyDrive/results/sbert_embeddings.npy", sbert_embeddings)
print("✅ SBERT embeddings saved for Task 3")



# Tokenize text
tokenized_texts = [t.split() for t in texts]
w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=2, workers=4)
w2v_embeddings = np.array([np.mean([w2v_model.wv[word] for word in words if word in w2v_model.wv], axis=0)
                           for words in tokenized_texts])
# Remove None rows
w2v_embeddings = np.array([e for e in w2v_embeddings if e is not None])

print("Computing SBERT embeddings...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
sbert_embeddings = sbert_model.encode(texts, show_progress_bar=True)

# -------------------------------
# 3. Visualization via PCA/t-SNE
# -------------------------------
print("\nVisualizing embeddings...")
# Reduce dimensions with PCA to speed up t-SNE
pca = PCA(n_components=50, random_state=42)
w2v_pca = pca.fit_transform(w2v_embeddings)
sbert_pca = pca.fit_transform(sbert_embeddings)

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
w2v_tsne = tsne.fit_transform(w2v_pca)
sbert_tsne = tsne.fit_transform(sbert_pca)

# Plot
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.scatter(w2v_tsne[:,0], w2v_tsne[:,1], s=5, alpha=0.6)
plt.title("Word2Vec Embeddings t-SNE")

plt.subplot(1,2,2)
plt.scatter(sbert_tsne[:,0], sbert_tsne[:,1], s=5, alpha=0.6)
plt.title("SBERT Embeddings t-SNE")

plt.savefig(OUTPUT_IMAGE_PATH, dpi=300)
plt.show()

print(f"\n✅ Embedding map saved to {OUTPUT_IMAGE_PATH}")