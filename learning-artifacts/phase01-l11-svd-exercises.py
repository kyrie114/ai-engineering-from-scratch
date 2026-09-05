# Learner practice for: phases/01-math-foundations/11-singular-value-decomposition/docs/en.md
# Rules: predict BEFORE you run; compare against the lesson's code/svd.py; numpy only.
# All TODOs are yours to fill. Commit your progress as you complete each exercise.

import numpy as np


def exercise_1_svd_via_eigendecomposition(A):
    """Exercise 1 (en.md): SVD WITHOUT power iteration.

    Route: eigendecompose A^T A -> get V and sigma_i = sqrt(eigenvalue_i),
    then U = A @ V / sigma. Compare accuracy against np.linalg.svd.
    PREDICT FIRST: where will this route lose precision, and why?
    (Hint: what happens to the condition number when you form A^T A?)
    """
    raise NotImplementedError


def exercise_2_image_compression_ranks():
    """Exercise 2 (en.md): compress a grayscale image at ranks 1/5/10/25/50/100.

    For each rank report compression ratio and relative error, and find the
    rank where the image is still visually acceptable.
    You can build a synthetic image exactly like code/svd.py demo_image_compression()
    does (no external image file needed).
    PREDICT FIRST: which rank will be the first "visually acceptable" one?
    """
    raise NotImplementedError


def exercise_3_tiny_recommender():
    """Exercise 3 (en.md): 10x8 user-movie ratings matrix with missing entries.

    Fill missing entries with row means, take a rank-3 SVD approximation,
    use it to predict the missing ratings, and check predictions against truth.
    PREDICT FIRST: will rank-3 predictions beat the row-mean baseline? By how much?
    """
    raise NotImplementedError


def exercise_4_topic_clusters_via_svd():
    """Exercise 4 (en.md): 100x50 document-term matrix with 3 synthetic topics.

    Each topic owns 5 terms. Add noise. Verify the top 3 singular values
    dominate the rest, then project documents into 3D latent space and
    check same-topic documents cluster together.
    PREDICT FIRST: what should the sigma_3 / sigma_4 gap look like if
    the topics are well separated?
    """
    raise NotImplementedError


def exercise_5_noise_vs_truncation_rank():
    """Exercise 5 (en.md): rank-3 clean matrix (50x40) + Gaussian noise.

    For noise sigma in [0.1, 0.5, 1.0, 2.0], sweep truncation rank k=1..40,
    measure reconstruction error against the CLEAN matrix, and find the
    optimal k for each noise level.
    PREDICT FIRST: as noise grows, does the optimal k go UP or DOWN? Why?
    """
    raise NotImplementedError


if __name__ == "__main__":
    # Run exercises one at a time as we cover them in the lesson.
    print("Fill the TODOs, then run each exercise here.")
