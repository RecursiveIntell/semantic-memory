#include "similarity.h"
#include <math.h>

/*
 * Cosine similarity between two equal-length f32 vectors.
 *
 * Returns NaN when either norm is zero (degenerate vectors), so the Rust
 * FFI wrapper can convert NaN -> None.  The wrapper also checks length
 * mismatch / emptiness before calling, but we guard defensively anyway.
 *
 * Written as straightforward loops so GCC -O3 -mavx2 -mfma can auto-vectorize
 * the dot product and norm sums into packed FMA instructions.
 */
float sm_cosine_similarity(const float *a, const float *b, size_t n)
{
    if (a == NULL || b == NULL || n == 0) {
        return NAN;
    }

    /* Single pass: accumulate dot product and both squared norms together
     * to improve cache locality.  GCC auto-vectorizes all three reductions. */
    float dot = 0.0f;
    float norm_a_sq = 0.0f;
    float norm_b_sq = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float ai = a[i];
        float bi = b[i];
        dot       += ai * bi;
        norm_a_sq += ai * ai;
        norm_b_sq += bi * bi;
    }

    float norm_a = sqrtf(norm_a_sq);
    float norm_b = sqrtf(norm_b_sq);

    if (norm_a == 0.0f || norm_b == 0.0f) {
        return NAN;
    }

    return dot / (norm_a * norm_b);
}