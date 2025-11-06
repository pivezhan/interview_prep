#include <iostream>
#include <vector>
using namespace std;

// Cache-optimized matrix multiplication with loop tiling
void matrixmult(vector<vector<int>>& A, vector<vector<int>>& B, vector<vector<int>>& C, int n, int m, int p, int block_size = 32) {
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < p; j += block_size) {
            for (int k = 0; k < m; k += block_size) {
                // Process a block of size block_size x block_size
                for (int ii = i; ii < min(i + block_size, n); ii++) {
                    for (int jj = j; jj < min(j + block_size, p); jj++) {
                        for (int kk = k; kk < min(k + block_size, m); kk++) {
                            C[ii][jj] += A[ii][kk] * B[kk][jj];
                        }
                    }
                }
            }
        }
    }
}

int main() {
    int n, m, p;
    // Read dimensions
    cin >> n >> m >> p;
    
    // Initialize matrices
    vector<vector<int>> A(n, vector<int>(m));
    vector<vector<int>> B(m, vector<int>(p));
    vector<vector<int>> C(n, vector<int>(p, 0));

    // Read matrix A
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cin >> A[i][j];
        }
    }
    
    // Read matrix B
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            cin >> B[i][j];
        }
    }
    
    // Perform cache-optimized matrix multiplication
    matrixmult(A, B, C, n, m, p);
    
    // Print result matrix C
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            cout << C[i][j] << " ";
        }
        cout << endl;
    }
    
    return 0;
}