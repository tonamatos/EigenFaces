"""
utils.py — Eigenfaces core computations.

All linear algebra is intentionally written from first principles
to match the lecture notebook. No sklearn wrappers.
"""

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from PIL import Image
import io

# ── Data loading ────────────────────────────────────────────────────────────

def load_faces():
    """
    Returns:
        X      : (400, 4096) float64 array, pixel values in [0, 1]
        images : (400, 64, 64) float64 array
        target : (400,) int array, subject IDs 0–39
    """
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    return data.data, data.images, data.target


# ── Core linear algebra ──────────────────────────────────────────────────────

def compute_mean_face(X):
    """Returns mean face vector of shape (4096,)."""
    return X.mean(axis=0)


def center_data(X, mean_face):
    """Returns X̃ = X − mean, shape (400, 4096)."""
    return X - mean_face


def compute_covariance(X_centered):
    """
    Returns the (4096, 4096) sample covariance matrix C = X̃ᵀX̃ / (n−1).
    Note: symmetric by construction (C = Cᵀ).
    WARNING: this is a 4096×4096 matrix (~128 MB). For the full version
    use compute_pca_via_svd instead.
    """
    n = X_centered.shape[0]
    return (X_centered.T @ X_centered) / (n - 1)


def compute_pca_via_svd(X_centered, n_components=None):
    """
    Efficient PCA using the dual trick: eigenvectors of X̃X̃ᵀ (400×400)
    recover eigenvectors of X̃ᵀX̃ (4096×4096) without forming the big matrix.

    Steps:
      1. Compute SVD of X̃: X̃ = U Σ Vᵀ
      2. Eigenvalues of C = X̃ᵀX̃/(n−1) are σᵢ²/(n−1)
      3. Eigenvectors (eigenfaces) are the columns of V

    Returns:
        eigenvalues  : (k,) array, sorted descending, = σᵢ²/(n−1)
        eigenfaces   : (k, 4096) array, rows are eigenvectors φᵢ
        projections  : (400, k) array, coordinates of each face in eigenface basis
    """
    n = X_centered.shape[0]
    if n_components is None:
        n_components = min(X_centered.shape) - 1

    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    eigenvalues = (s ** 2) / (n - 1)
    eigenfaces = Vt  # shape (min(n,d), 4096)

    eigenvalues = eigenvalues[:n_components]
    eigenfaces = eigenfaces[:n_components]
    projections = X_centered @ eigenfaces.T  # (400, k)

    return eigenvalues, eigenfaces, projections


def reconstruct_face(face, mean_face, eigenfaces, k):
    """
    Reconstruct a face using the first k eigenfaces.

    f̂_k = f̄ + Σᵢ₌₁ᵏ ⟨f−f̄, φᵢ⟩ φᵢ

    Args:
        face       : (4096,) array
        mean_face  : (4096,) array
        eigenfaces : (K, 4096) array  (K ≥ k)
        k          : int

    Returns:
        reconstruction : (4096,) array clipped to [0, 1]
        coefficients   : (k,) array of projection coordinates
    """
    centered = face - mean_face
    coeffs = eigenfaces[:k] @ centered          # (k,) dot products
    reconstruction = mean_face + eigenfaces[:k].T @ coeffs
    return np.clip(reconstruction, 0, 1), coeffs


def reconstruction_error(face, mean_face, eigenfaces, k):
    """
    Returns the squared reconstruction error ||f − f̂_k||².
    By Parseval: this equals Σᵢ>k ⟨f−f̄, φᵢ⟩².
    """
    recon, _ = reconstruct_face(face, mean_face, eigenfaces, k)
    return np.sum((face - recon) ** 2)


def variance_explained(eigenvalues):
    """Returns cumulative fraction of variance explained (0 to 1)."""
    total = eigenvalues.sum()
    return np.cumsum(eigenvalues) / total


def projection_matrix(eigenface):
    """
    Returns the rank-1 orthogonal projection Pᵢ = φᵢ φᵢᵀ onto span{φᵢ}.
    Shape: (4096, 4096). Expensive — only use for small demos.
    """
    phi = eigenface.reshape(-1, 1)
    return phi @ phi.T


# ── Image preprocessing ──────────────────────────────────────────────────────

def preprocess_uploaded_image(image_bytes):
    """
    Convert an uploaded image (bytes) to a (4096,) float array
    matching the Olivetti faces format: 64×64 grayscale, values in [0, 1].
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((64, 64))
    arr = np.array(img, dtype=np.float64) / 255.0
    return arr.flatten()
