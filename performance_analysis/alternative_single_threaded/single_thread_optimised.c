// Alternative Single-Threaded Implementation
// Two-Row Memory Optimization for Smith-Waterman Algorithm
//
// This implementation maintains only 2 rows of the DP matrix in memory,
// achieving superior single-threaded performance through improved cache locality.


// This is NOT the main submission. See ../optimized/sw_opt.c for the primary implementation.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>

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

static inline int sw_cell(char a, char b, int diag, int up, int left) {
    int match = diag + (a == b ? MATCH : MISMATCH);
    int del = up + GAP;
    int ins = left + GAP;
    return MAX(0, MAX(match, MAX(del, ins)));
}

int smith_waterman_optimized(const char *seq1, const char *seq2, int len1, int len2) {
    // Memory-optimized: only keep two rows instead of full matrix
    // Space complexity: O(len2) instead of O(len1 * len2)
    int *prev_row = calloc(len2 + 1, sizeof(int));
    int *curr_row = calloc(len2 + 1, sizeof(int));
    
    int max_score = 0;
    
    for (int i = 1; i <= len1; i++) {
        curr_row[0] = 0;  // First column is always 0
        
        // 6x loop unrolling for better instruction-level parallelism
        int j = 1;
        int limit = len2 - 5;  // Process 6 elements at a time
        
        for (; j <= limit; j += 6) {
            // Unrolled iteration 1
            int match1 = prev_row[j - 1] + (seq1[i - 1] == seq2[j - 1] ? MATCH : MISMATCH);
            int del1 = prev_row[j] + GAP;
            int ins1 = curr_row[j - 1] + GAP;
            curr_row[j] = MAX(0, MAX(match1, MAX(del1, ins1)));
            if (curr_row[j] > max_score) max_score = curr_row[j];
            
            // Unrolled iteration 2
            int match2 = prev_row[j] + (seq1[i - 1] == seq2[j] ? MATCH : MISMATCH);
            int del2 = prev_row[j + 1] + GAP;
            int ins2 = curr_row[j] + GAP;
            curr_row[j + 1] = MAX(0, MAX(match2, MAX(del2, ins2)));
            if (curr_row[j + 1] > max_score) max_score = curr_row[j + 1];
            
            // Unrolled iteration 3
            int match3 = prev_row[j + 1] + (seq1[i - 1] == seq2[j + 1] ? MATCH : MISMATCH);
            int del3 = prev_row[j + 2] + GAP;
            int ins3 = curr_row[j + 1] + GAP;
            curr_row[j + 2] = MAX(0, MAX(match3, MAX(del3, ins3)));
            if (curr_row[j + 2] > max_score) max_score = curr_row[j + 2];
            
            // Unrolled iteration 4
            int match4 = prev_row[j + 2] + (seq1[i - 1] == seq2[j + 2] ? MATCH : MISMATCH);
            int del4 = prev_row[j + 3] + GAP;
            int ins4 = curr_row[j + 2] + GAP;
            curr_row[j + 3] = MAX(0, MAX(match4, MAX(del4, ins4)));
            if (curr_row[j + 3] > max_score) max_score = curr_row[j + 3];
            
            // Unrolled iteration 5
            int match5 = prev_row[j + 3] + (seq1[i - 1] == seq2[j + 3] ? MATCH : MISMATCH);
            int del5 = prev_row[j + 4] + GAP;
            int ins5 = curr_row[j + 3] + GAP;
            curr_row[j + 4] = MAX(0, MAX(match5, MAX(del5, ins5)));
            if (curr_row[j + 4] > max_score) max_score = curr_row[j + 4];
            
            // Unrolled iteration 6
            int match6 = prev_row[j + 4] + (seq1[i - 1] == seq2[j + 4] ? MATCH : MISMATCH);
            int del6 = prev_row[j + 5] + GAP;
            int ins6 = curr_row[j + 4] + GAP;
            curr_row[j + 5] = MAX(0, MAX(match6, MAX(del6, ins6)));
            if (curr_row[j + 5] > max_score) max_score = curr_row[j + 5];
        }
        
        // Handle remaining elements (cleanup loop)
        for (; j <= len2; j++) {
            int match = prev_row[j - 1] + (seq1[i - 1] == seq2[j - 1] ? MATCH : MISMATCH);
            int del = prev_row[j] + GAP;
            int ins = curr_row[j - 1] + GAP;
            curr_row[j] = MAX(0, MAX(match, MAX(del, ins)));
            if (curr_row[j] > max_score)
                max_score = curr_row[j];
        }
        
        // Swap pointers (no data copy needed)
        int *temp = prev_row;
        prev_row = curr_row;
        curr_row = temp;
    }
    
    free(prev_row);
    free(curr_row);
    
    return max_score;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <sequence_length>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    srand(42);

    char *seq1 = malloc((N + 1) * sizeof(char));
    char *seq2 = malloc((N + 1) * sizeof(char));

    generate_sequence(seq1, N);
    generate_sequence(seq2, N);

    clock_t start = clock();
    int score = smith_waterman_optimized(seq1, seq2, N, N);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Sequence length: %d\n", N);
    printf("Smith-Waterman optimized score: %d\n", score);
    printf("Execution time: %.6f seconds\n", elapsed);

    free(seq1);
    free(seq2);
    return 0;
}