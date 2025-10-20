import numpy as np
from typing import List, Tuple

class AttentionRanker:
    """
    Lightweight attention mechanism for content relevance scoring.
    
    Architecture:
    1. Embed words using pretrained GloVe-like vectors
    2. Compute scaled dot-product attention with learned query
    3. Generate context-aware relevance score
    
    Similar to single-head self-attention in transformers, but simplified.
    """
    
    def __init__(self, vocab_size: int = 5000, embed_dim: int = 128):
        """
        Initialize embeddings and learnable parameters.
        
        Args:
            vocab_size: Size of vocabulary (hashed)
            embed_dim: Dimension of embedding space
        """
        # Initialize random embeddings (would be pretrained in production)
        self.embeddings = np.random.randn(vocab_size, embed_dim) * 0.1
        
        # Learnable query vector (represents user interests)
        self.query_vector = np.random.randn(embed_dim) * 0.1
        
        # Scaling factor for dot-product attention (prevents saturation)
        self.scale = np.sqrt(embed_dim)
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
    
    def tokenize(self, text: str) -> np.ndarray:
        """
        Simple hash-based tokenization.
        
        In production, would use proper tokenizer (BPE, WordPiece).
        Hash ensures consistent token IDs without maintaining vocabulary.
        """
        words = text.lower().split()
        # Hash each word to vocab index
        token_ids = np.array([hash(word) % self.vocab_size for word in words])
        return token_ids
    
    def compute_attention(self, embeddings: np.ndarray, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scaled dot-product attention mechanism.
        
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        
        Args:
            embeddings: Token embeddings (seq_len, embed_dim)
            query: Query vector (embed_dim,)
        
        Returns:
            attention_weights: Normalized attention scores (seq_len,)
            context_vector: Weighted sum of embeddings (embed_dim,)
        """
        # Compute attention scores (dot product with query)
        # Shape: (seq_len,)
        scores = np.dot(embeddings, query) / self.scale
        
        # Apply softmax to get attention weights
        # Numerically stable softmax
        exp_scores = np.exp(scores - np.max(scores))
        attention_weights = exp_scores / np.sum(exp_scores)
        
        # Compute context vector (weighted sum of embeddings)
        # Shape: (embed_dim,)
        context_vector = np.sum(embeddings * attention_weights[:, np.newaxis], axis=0)
        
        return attention_weights, context_vector
    
    def score_content(self, text: str) -> Tuple[float, np.ndarray]:
        """
        Score content relevance using attention mechanism.
        
        Pipeline:
        1. Tokenize text
        2. Embed tokens
        3. Compute attention-weighted context
        4. Score context similarity to query
        
        Args:
            text: Input text to score
        
        Returns:
            relevance_score: Scalar relevance (higher = more relevant)
            attention_weights: Per-token attention weights (for visualization)
        """
        # Tokenize and embed
        token_ids = self.tokenize(text)
        token_embeddings = self.embeddings[token_ids]  # (seq_len, embed_dim)
        
        # Compute attention
        attention_weights, context_vector = self.compute_attention(
            token_embeddings, 
            self.query_vector
        )
        
        # Final relevance: cosine similarity between context and query
        relevance_score = np.dot(context_vector, self.query_vector) / (
            np.linalg.norm(context_vector) * np.linalg.norm(self.query_vector) + 1e-8
        )
        
        return float(relevance_score), attention_weights
    
    def update_interests(self, interest_texts: List[str]):
        """
        Update query vector based on explicit interest descriptions.
        
        Averages embeddings of interest keywords to form new query.
        Allows dynamic interest adaptation.
        """
        all_embeddings = []
        for text in interest_texts:
            tokens = self.tokenize(text)
            embeddings = self.embeddings[tokens]
            all_embeddings.append(np.mean(embeddings, axis=0))
        
        # New query = average of all interest embeddings
        self.query_vector = np.mean(all_embeddings, axis=0)
    
    def rank_posts(self, posts: List[str], threshold: float = 0.0) -> List[Tuple[str, float, np.ndarray]]:
        """
        Rank multiple posts by relevance.
        
        Returns:
            List of (post, score, attention_weights) sorted by score descending
        """
        scored = []
        for post in posts:
            score, attn_weights = self.score_content(post)
            if score >= threshold:
                scored.append((post, score, attn_weights))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# ============================================
# Usage Example
# ============================================

if __name__ == "__main__":
    # Initialize ranker
    ranker = AttentionRanker(vocab_size=5000, embed_dim=128)
    
    # Define interests (updates query vector)
    interests = [
        "machine learning research",
        "AI safety alignment",
        "deep learning papers"
    ]
    ranker.update_interests(interests)
    
    # Example posts
    posts = [
        "New paper on transformer attention mechanisms",
        "Check out this cute cat video!",
        "Breakthrough in reinforcement learning from human feedback",
        "My lunch today was amazing"
    ]
    
    # Rank posts
    ranked = ranker.rank_posts(posts, threshold=0.3)
    
    # Display results
    print("Ranked Posts (by relevance):\n")
    for i, (post, score, attn) in enumerate(ranked, 1):
        print(f"{i}. [{score:.3f}] {post}")
        # Optionally visualize attention
        words = post.split()
        if len(words) == len(attn):
            print(f"   Attention: {dict(zip(words, attn.round(3)))}\n")
