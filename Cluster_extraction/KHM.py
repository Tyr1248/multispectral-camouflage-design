import numpy as np
from collections import defaultdict, Counter


class OptimizedKHM:
    def __init__(self, n_clusters=4, p=2, max_iter=100, eps=1e-8, min_center_distance=10.0):
        """
        Optimized K-Harmonic Means (KHM) clustering with histogram-based initialization.

        Parameters:
        - n_clusters: number of clusters (default: 4)
        - p: harmonic parameter (typically >= 2)
        - max_iter: maximum iterations
        - eps: small value to avoid division by zero
        - min_center_distance: minimum Euclidean distance between initial centers in Lab space
        """
        self.n_clusters = n_clusters
        self.p = p
        self.max_iter = max_iter
        self.eps = eps
        self.min_center_distance = min_center_distance

    def initialize_centers_color_histogram(self, X):
        """
        Initialize cluster centers using color histogram peaks with distance constraint.
        Assumes X is in standard CIE Lab: L∈[0,100], a,b∈[-128,127].
        """
        # Step 1: Quantize colors and build mapping from quantized index to list of original pixels
        quantized_indices = []
        quant_to_pixels = defaultdict(list)

        for pixel in X:
            q_idx = self._quantize_pixel(pixel)
            quantized_indices.append(q_idx)
            quant_to_pixels[q_idx].append(pixel)

        # Step 2: Count frequency
        color_counts = Counter(quantized_indices)

        # Step 3: Sort by frequency descending
        sorted_colors = color_counts.most_common()

        # Step 4: Select centers with distance constraint
        selected_centers = []
        for q_idx, _ in sorted_colors:
            # Use mean of all pixels in this bin as representative (more stable than first sample)
            candidate = np.mean(quant_to_pixels[q_idx], axis=0)

            # Check distance to already selected centers
            if all(np.linalg.norm(candidate - c) >= self.min_center_distance for c in selected_centers):
                selected_centers.append(candidate)
                if len(selected_centers) == self.n_clusters:
                    break

        # Fallback: if not enough distinct peaks, pad with random samples
        while len(selected_centers) < self.n_clusters:
            idx = np.random.randint(len(X))
            candidate = X[idx]
            if all(np.linalg.norm(candidate - c) >= self.min_center_distance for c in selected_centers):
                selected_centers.append(candidate)

        return np.array(selected_centers)

    def _quantize_pixel(self, pixel):
        """
        Quantize a single Lab pixel according to paper:
        - L: [0,100] → 5 bins → step ≈ 20
        - a: [-128,127] → 5 bins → step ≈ 51.2
        - b: [-128,127] → 12 bins → step ≈ 21.33
        Returns a scalar index P = 25*L_idx + 5*a_idx + b_idx
        """
        L, a, b = pixel

        # Clamp to valid range (defensive)
        L = np.clip(L, 0, 100)
        a = np.clip(a, -128, 127)
        b = np.clip(b, -128, 127)

        L_idx = min(int(L / 20), 4)  # 0–4 (5 bins)
        a_idx = min(int((a + 128) / 51.2), 4)  # map [-128,127] → [0,255] → 5 bins
        b_idx = min(int((b + 128) / 21.333), 11)  # 12 bins

        P = 25 * L_idx + 5 * a_idx + b_idx
        return int(P)

    def fit(self, X):
        """
        Fit KHM model to data X (shape: [n_samples, 3]), assumed in CIE Lab space.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != 3:
            raise ValueError("Input X must have shape (n_samples, 3) for Lab channels.")

        centers = self.initialize_centers_color_histogram(X)

        for iteration in range(self.max_iter):
            U = self._calculate_membership(X, centers)
            W = self._calculate_weights(X, centers)
            new_centers = self._update_centers(X, U, W)

            if np.allclose(centers, new_centers, atol=1e-5, rtol=0):
                break
            centers = new_centers

        self.cluster_centers_ = centers
        return self

    def _calculate_membership(self, X, centers):
        n_samples = X.shape[0]
        n_centers = centers.shape[0]

        distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)  # (n, k)
        distances = np.maximum(distances, self.eps)

        numerator = distances ** (-self.p - 2)
        denominator = np.sum(numerator, axis=1, keepdims=True)
        U = numerator / denominator  # (n, k)

        return U.T  # (k, n)

    def _calculate_weights(self, X, centers):
        n_samples = X.shape[0]
        distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)  # (n, k)
        distances = np.maximum(distances, self.eps)

        numerator = np.sum(distances ** (-self.p - 2), axis=1)  # (n,)
        denominator_term = np.sum(distances ** (-self.p), axis=1)  # (n,)
        denominator = denominator_term ** 2

        W = numerator / np.maximum(denominator, self.eps)
        return W

    def _update_centers(self, X, U, W):
        n_centers, n_samples = U.shape
        weighted_X = X * W[:, None]  # (n, 3)

        numerator = np.dot(U, weighted_X)  # (k, 3)
        denominator = np.sum(U * W, axis=1, keepdims=True)  # (k, 1)

        # Avoid division by zero
        denominator = np.maximum(denominator, self.eps)
        new_centers = numerator / denominator
        return new_centers

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        distances = np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
        return np.argmin(distances, axis=1)