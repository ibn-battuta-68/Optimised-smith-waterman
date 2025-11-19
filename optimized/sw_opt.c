#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

// === OpenMP Configuration ===
// #define NUM_THREADS 8  // Set this to control parallelization (1 = no parallelization overhead)
// ============================

#define MAX(A,B) ((A) > (B) ? (A) : (B))
#define MATCH     2
#define MISMATCH -1
#define GAP      -2

void generate_sequence(char *seq, int n) {
    const char alphabet[] = "ACGT";
    for (int i = 0; i < n; i++)
        seq[i] = alphabet[rand() % 4];
    seq[n] = '\0';
}

int smith_waterman_optimized(const char *seq1, const char *seq2, int len1, int len2) {
    // Set number of threads
    
    // Use full matrix with tiled/blocked computation
    int **matrix = malloc((len1 + 1) * sizeof(int*));
    for (int i = 0; i <= len1; i++) {
        matrix[i] = calloc(len2 + 1, sizeof(int));
    }
    
    int max_score = 0;
    int tile_size = 512;  // Tile size for cache locality
    
    // Process matrix in tiles along anti-diagonals
    int num_tiles_i = (len1 + tile_size - 1) / tile_size;
    int num_tiles_j = (len2 + tile_size - 1) / tile_size;
    int total_tile_diagonals = num_tiles_i + num_tiles_j - 1;
    
    for (int tile_diag = 0; tile_diag < total_tile_diagonals; tile_diag++) {
        int max_tile_i = (tile_diag < num_tiles_i) ? tile_diag : num_tiles_i - 1;
        
        #pragma omp parallel for reduction(max:max_score) schedule(dynamic)
        for (int tile_i = 0; tile_i <= max_tile_i; tile_i++) {
            int tile_j = tile_diag - tile_i;
            if (tile_j >= num_tiles_j || tile_j < 0) continue;
            
            int i_start = tile_i * tile_size + 1;
            int i_end = ((tile_i + 1) * tile_size <= len1) ? (tile_i + 1) * tile_size : len1;
            int j_start = tile_j * tile_size + 1;
            int j_end = ((tile_j + 1) * tile_size <= len2) ? (tile_j + 1) * tile_size : len2;
            
            // Process this tile with AVX2 SIMD optimization
            // Smart approach: use SIMD for vectorizing arithmetic operations
            for (int i = i_start; i <= i_end; i++) {
                int j = j_start;
                char seq1_char = seq1[i-1];
                
                // AVX2 vectorized loop - process 8 elements at a time
                int simd_limit = j_end - 7;
                
                __m256i gap_vec = _mm256_set1_epi32(GAP);
                __m256i zero_vec = _mm256_setzero_si256();
                
                for (; j <= simd_limit; j += 8) {
                    // Load 8 diagonal values: matrix[i-1][j-1] through matrix[i-1][j+6]
                    __m256i diag_vec = _mm256_loadu_si256((__m256i*)&matrix[i-1][j-1]);
                    
                    // Load 8 up values: matrix[i-1][j] through matrix[i-1][j+7]
                    __m256i up_vec = _mm256_loadu_si256((__m256i*)&matrix[i-1][j]);
                    
                    // Create match/mismatch scores using SIMD comparison
                    // Load 8 sequence characters as bytes, then compare
                    __m128i seq2_chars = _mm_loadl_epi64((__m128i*)&seq2[j-1]);  // Load 8 bytes
                    __m128i seq1_broadcast = _mm_set1_epi8(seq1_char);  // Broadcast seq1 char
                    __m128i cmp_mask = _mm_cmpeq_epi8(seq1_broadcast, seq2_chars);  // Compare 8 chars
                    
                    // Convert byte mask to 32-bit integers (1 for match, 0 for mismatch)
                    // Extend each byte to 32-bit integer
                    __m256i cmp_mask_32 = _mm256_cvtepi8_epi32(cmp_mask);
                    
                    // Create match/mismatch scores: use mask to select MATCH or MISMATCH
                    __m256i match_score_vec = _mm256_set1_epi32(MATCH);
                    __m256i mismatch_score_vec = _mm256_set1_epi32(MISMATCH);
                    __m256i match_vec = _mm256_blendv_epi8(mismatch_score_vec, match_score_vec, cmp_mask_32);
                    
                    // Vectorized addition: diag + match/mismatch
                    __m256i score_match = _mm256_add_epi32(diag_vec, match_vec);
                    
                    // Vectorized addition: up + gap
                    __m256i score_del = _mm256_add_epi32(up_vec, gap_vec);
                    
                    // Take max of match and deletion scores
                    __m256i max_vec = _mm256_max_epi32(score_match, score_del);
                    
                    // Store results temporarily
                    int temp_scores[8];
                    _mm256_storeu_si256((__m256i*)temp_scores, max_vec);
                    
                    // Handle insertion dependency sequentially (must be done due to left dependency)
                    // But we've already vectorized the expensive diagonal and deletion calculations
                    for (int k = 0; k < 8; k++) {
                        int ins = matrix[i][j + k - 1] + GAP;
                        int final_score = MAX(0, MAX(temp_scores[k], ins));
                        matrix[i][j + k] = final_score;
                        if (final_score > max_score) max_score = final_score;
                    }
                }
                
                // 6x unrolled cleanup
                int limit = j_end - 5;
                for (; j <= limit; j += 6) {
                    int match1 = matrix[i-1][j-1] + (seq1[i-1] == seq2[j-1] ? MATCH : MISMATCH);
                    int del1 = matrix[i-1][j] + GAP;
                    int ins1 = matrix[i][j-1] + GAP;
                    int score1 = MAX(0, MAX(match1, MAX(del1, ins1)));
                    matrix[i][j] = score1;
                    if (score1 > max_score) max_score = score1;
                    
                    int match2 = matrix[i-1][j] + (seq1[i-1] == seq2[j] ? MATCH : MISMATCH);
                    int del2 = matrix[i-1][j+1] + GAP;
                    int ins2 = score1 + GAP;
                    int score2 = MAX(0, MAX(match2, MAX(del2, ins2)));
                    matrix[i][j+1] = score2;
                    if (score2 > max_score) max_score = score2;
                    
                    int match3 = matrix[i-1][j+1] + (seq1[i-1] == seq2[j+1] ? MATCH : MISMATCH);
                    int del3 = matrix[i-1][j+2] + GAP;
                    int ins3 = score2 + GAP;
                    int score3 = MAX(0, MAX(match3, MAX(del3, ins3)));
                    matrix[i][j+2] = score3;
                    if (score3 > max_score) max_score = score3;
                    
                    int match4 = matrix[i-1][j+2] + (seq1[i-1] == seq2[j+2] ? MATCH : MISMATCH);
                    int del4 = matrix[i-1][j+3] + GAP;
                    int ins4 = score3 + GAP;
                    int score4 = MAX(0, MAX(match4, MAX(del4, ins4)));
                    matrix[i][j+3] = score4;
                    if (score4 > max_score) max_score = score4;
                    
                    int match5 = matrix[i-1][j+3] + (seq1[i-1] == seq2[j+3] ? MATCH : MISMATCH);
                    int del5 = matrix[i-1][j+4] + GAP;
                    int ins5 = score4 + GAP;
                    int score5 = MAX(0, MAX(match5, MAX(del5, ins5)));
                    matrix[i][j+4] = score5;
                    if (score5 > max_score) max_score = score5;
                    
                    int match6 = matrix[i-1][j+4] + (seq1[i-1] == seq2[j+4] ? MATCH : MISMATCH);
                    int del6 = matrix[i-1][j+5] + GAP;
                    int ins6 = score5 + GAP;
                    int score6 = MAX(0, MAX(match6, MAX(del6, ins6)));
                    matrix[i][j+5] = score6;
                    if (score6 > max_score) max_score = score6;
                }
                
                // Final cleanup loop
                for (; j <= j_end; j++) {
                    int match = matrix[i-1][j-1] + (seq1[i-1] == seq2[j-1] ? MATCH : MISMATCH);
                    int del = matrix[i-1][j] + GAP;
                    int ins = matrix[i][j-1] + GAP;
                    int score = MAX(0, MAX(match, MAX(del, ins)));
                    matrix[i][j] = score;
                    if (score > max_score) max_score = score;
                }
            }
        }
    }
    
    for (int i = 0; i <= len1; i++) {
        free(matrix[i]);
    }
    free(matrix);
    
    return max_score;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <sequence_length> [num_threads]\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    srand(42);

    int NUM_THREADS = (argc >= 3) ? atoi(argv[2]) : omp_get_num_procs();
    omp_set_num_threads(NUM_THREADS);

    char *seq1 = malloc((N + 1) * sizeof(char));
    char *seq2 = malloc((N + 1) * sizeof(char));

    generate_sequence(seq1, N);
    generate_sequence(seq2, N);

    double start = omp_get_wtime();
    int score = smith_waterman_optimized(seq1, seq2, N, N);
    double end = omp_get_wtime();

    double elapsed = end - start;
    printf("Sequence length: %d\n", N);
    printf("Smith-Waterman optimized score: %d\n", score);
    printf("Execution time: %.6f seconds\n", elapsed);

    free(seq1);
    free(seq2);
    return 0;
}