#include <iostream>
#include <vector>
using namespace std;

// Naive matrix multiplication
void matrixmult(vector<vector<int>>& A, vector<vector<int>>& B, vector<vector<int>>& C, int n, int m, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            C[i][j] = 0; // Initialize result element
            for (int k = 0; k < m; k++) {
                C[i][j] += A[i][k] * B[k][j]; // Standard multiplication
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
    
    // Perform matrix multiplication
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