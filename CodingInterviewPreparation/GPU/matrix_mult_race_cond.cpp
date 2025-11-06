#include <iostream>
#include <thread>
#include <vector>
using namespace std;

// Shared result matrix
vector<vector<int>> C(2, vector<int>(2, 0));

void compute_partial(int i, int j, int k, vector<vector<int>>& A, vector<vector<int>>& B) {
    // Compute A[i][k] * B[k][j] and add to C[i][j]
    C[i][j] += A[i][k] * B[k][j]; // Race condition here
}

int main() {
    // Initialize matrices
    vector<vector<int>> A = {{1, 2}, {3, 4}};
    vector<vector<int>> B = {{5, 6}, {7, 8}};
    
    // Create two threads to compute parts of C[0][0]
    thread t1(compute_partial, 0, 0, 0, ref(A), ref(B)); // Computes A[0][0]*B[0][0] = 1*5 = 5
    thread t2(compute_partial, 0, 0, 1, ref(A), ref(B)); // Computes A[0][1]*B[1][0] = 2*7 = 14
    
    t1.join();
    t2.join();
    
    cout << "C[0][0] = " << C[0][0] << endl; // Expected: 19
    return 0;
}
// Fixed race condition version:
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
using namespace std;

// Shared result matrix
vector<vector<int>> C(2, vector<int>(2, 0));
mutex mtx; // Mutex for synchronization

void compute_partial(int i, int j, int k, vector<vector<int>>& A, vector<vector<int>>& B) {
    int result = A[i][k] * B[k][j]; // Compute partial result
    mtx.lock(); // Lock to prevent concurrent access
    C[i][j] += result; // Safely update shared memory
    mtx.unlock(); // Unlock
}

int main() {
    // Initialize matrices
    vector<vector<int>> A = {{1, 2}, {3, 4}};
    vector<vector<int>> B = {{5, 6}, {7, 8}};
    
    // Create two threads to compute parts of C[0][0]
    thread t1(compute_partial, 0, 0, 0, ref(A), ref(B)); // Computes A[0][0]*B[0][0] = 5
    thread t2(compute_partial, 0, 0, 1, ref(A), ref(B)); // Computes A[0][1]*B[1][0] = 14
    
    t1.join();
    t2.join();
    
    cout << "C[0][0] = " << C[0][0] << endl; // Outputs: 19
    return 0;
}