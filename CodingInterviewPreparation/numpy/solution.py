#%% Arrays
"""
Problem: Convert a list to a NumPy array and reverse it
Description: Given a space-separated list of numbers, convert to NumPy array and reverse
Input: Space-separated float numbers
Output: Reversed NumPy array
"""

import numpy as np

if __name__ == '__main__':
    def arrays(arr):
        # Convert list to numpy array and reverse
        return np.array(arr[::-1], float)
    
    arr = input().strip().split(' ')
    result = arrays(arr)
    print(result)

#%% Alternative solution for Arrays
"""
Using NumPy's indexing for reversal
"""

import numpy as np

if __name__ == '__main__':
    arr = input().strip().split()
    # Convert to float array and reverse using slicing
    result = np.array(arr, dtype=float)[::-1]
    print(result)

#%% Shape and Reshape
"""
Problem: Reshape a 1D array into a 2D array
Description: Convert 1x9 array into 3x3 array
Input: 9 space-separated integers
Output: 3x3 NumPy array
"""

import numpy as np

if __name__ == '__main__':
    arr = np.array(input().split(), int)
    # Reshape to 3x3
    result = np.reshape(arr, (3, 3))
    print(result)

#%% Transpose and Flatten
"""
Problem: Transpose and flatten a matrix
Description: Given NxM matrix, print transpose and then flattened array
Input: N and M, then N rows of M integers each
Output: Transposed matrix, then flattened 1D array
"""

import numpy as np

if __name__ == '__main__':
    n, m = map(int, input().split())
    
    # Read the matrix
    matrix = []
    for _ in range(n):
        row = list(map(int, input().split()))
        matrix.append(row)
    
    arr = np.array(matrix)
    
    # Transpose
    print(np.transpose(arr))
    
    # Flatten
    print(arr.flatten())

#%% Concatenate
"""
Problem: Concatenate two arrays along axis 0
Description: Given two NxP and MxP arrays, concatenate along axis 0
Input: N, M, P then N+M rows of P integers
Output: Concatenated array
"""

import numpy as np

if __name__ == '__main__':
    n, m, p = map(int, input().split())
    
    # Read first array (N rows)
    arr1 = []
    for _ in range(n):
        arr1.append(list(map(int, input().split())))
    
    # Read second array (M rows)
    arr2 = []
    for _ in range(m):
        arr2.append(list(map(int, input().split())))

    # Concatenate along axis 0 (vertically)
    result = np.concatenate((np.array(arr1), np.array(arr2)), axis=0)
    print(result)

#%% Zeros and Ones
"""
Problem: Create arrays of zeros and ones
Description: Create array of zeros and ones with given shape
Input: Space-separated integers representing shape
Output: Array of zeros, then array of ones
"""

import numpy as np

if __name__ == '__main__':
    shape = tuple(map(int, input().split()))
    
    # Create zeros array
    print(np.zeros(shape, dtype=int))
    
    # Create ones array
    print(np.ones(shape, dtype=int))

#%% Eye and Identity
"""
Problem: Create identity matrix using eye and identity
Description: Print identity matrix of size NxM using np.eye
Input: N (rows) and M (columns)
Output: Identity matrix
"""

import numpy as np

if __name__ == '__main__':
    n, m = map(int, input().split())
    
    # Create identity matrix with N rows and M columns
    print(np.eye(n, m))

#%% Alternative for Eye and Identity
"""
Using np.identity (only for square matrices)
"""

import numpy as np

if __name__ == '__main__':
    n, m = map(int, input().split())
    
    # np.eye can create non-square matrices
    result = np.eye(n, m, dtype=int)
    print(result)

#%% Array Mathematics
"""
Problem: Perform element-wise operations on two arrays
Description: Given two NxM arrays, perform add, subtract, multiply, 
            floor divide, mod, and power operations
Input: N and M, then 2N rows (N for each array)
Output: 6 arrays showing results of each operation
"""

import numpy as np

if __name__ == '__main__':
    n, m = map(int, input().split())
    
    # Read first array
    a = np.array([input().split() for _ in range(n)], int)
    
    # Read second array
    b = np.array([input().split() for _ in range(n)], int)
    
    # Perform operations
    print(np.add(a, b))
    print(np.subtract(a, b))
    print(np.multiply(a, b))
    print(np.floor_divide(a, b))
    print(np.mod(a, b))
    print(np.power(a, b))

#%% Floor, Ceil and Rint
"""
Problem: Apply floor, ceil, and rint functions
Description: Apply numpy floor, ceil, and rint to array elements
Input: Array of float numbers
Output: Floor, ceil, and rint results
"""

import numpy as np

if __name__ == '__main__':
    np.set_printoptions(legacy='1.13')
    
    arr = np.array(input().split(), float)
    
    # Floor - largest integer not greater than x
    print(np.floor(arr))
    
    # Ceil - smallest integer not less than x
    print(np.+(arr))
    
    # Rint - round to nearest integer
    print(np.rint(arr))

#%% Sum and Prod
"""
Problem: Compute sum along axis, then product
Description: Sum array along axis 0, then compute product of result
Input: N and M, then NxM array
Output: Product of sums along axis 0
"""

import numpy as np

if __name__ == '__main__':
    n, m = map(int, input().split())

    # Read array
    arr = np.array([input().split() for _ in range(n)], int)
    
    # Sum along axis 0 (column-wise sum)
    sum_result = np.sum(arr, axis=0)

    # Product of the sum result
    print(np.prod(sum_result))

#%% Min and Max
"""
Problem: Find min along axis, then max of result
Description: Find minimum along axis 1, then maximum of that result
Input: N and M, then NxM array
Output: Maximum of row-wise minimums
"""

import numpy as np

if __name__ == '__main__':
    n, m = map(int, input().split())
    
    # Read array
    arr = np.array([input().split() for _ in range(n)], int)
    
    # Min along axis 1 (row-wise minimum)
    min_result = np.min(arr, axis=1)
    
    # Max of the min result
    print(np.max(min_result))

#%% Mean, Var, and Std
"""
Problem: Calculate mean, variance, and standard deviation
Description: Compute mean along axis 1, var along axis 0, and std of entire array
Input: N and M, then NxM array
Output: Mean (axis=1), Variance (axis=0), Std (entire array)
"""

import numpy as np

if __name__ == '__main__':
    n, m = map(int, input().split())
    
    # Read array
    arr = np.array([input().split() for _ in range(n)], int)
    
    # Mean along axis 1 (row-wise mean)
    print(np.mean(arr, axis=1))
    
    # Variance along axis 0 (column-wise variance)
    print(np.var(arr, axis=0))
    
    # Standard deviation of entire array
    print(round(np.std(arr), 11))

#%% Dot and Cross
"""
Problem: Compute dot product of two matrices
Description: Perform matrix multiplication (dot product)
Input: N, then two NxN matrices
Output: Dot product result
"""

import numpy as np

if __name__ == '__main__':
    n = int(input())
    
    # Read first matrix
    a = np.array([input().split() for _ in range(n)], int)
    
    # Read second matrix
    b = np.array([input().split() for _ in range(n)], int)
    
    # Dot product (matrix multiplication)
    print(np.dot(a, b))

#%% Inner and Outer
"""
Problem: Compute inner and outer products
Description: Calculate inner product and outer product of two arrays
Input: Two arrays A and B
Output: Inner product, then outer product
"""

import numpy as np

if __name__ == '__main__':
    a = np.array(input().split(), int)
    b = np.array(input().split(), int)
    
    # Inner product (dot product for 1D arrays)
    print(np.inner(a, b))
    
    # Outer product
    print(np.outer(a, b))

#%% Polynomials
"""
Problem: Evaluate polynomial at a given point
Description: Given coefficients and a value x, evaluate polynomial at x
Input: Coefficients array, value x
Output: Result of polynomial evaluation
"""

import numpy as np

if __name__ == '__main__':
    coefficients = np.array(input().split(), float)
    x = float(input())
    
    # Evaluate polynomial
    result = np.polyval(coefficients, x)
    print(result)

#%% Linear Algebra
"""
Problem: Compute determinant of a matrix
Description: Calculate the determinant of an NxN matrix
Input: N, then NxN matrix
Output: Determinant (rounded to 2 decimal places)
"""

import numpy as np

if __name__ == '__main__':
    n = int(input())
    
    # Read matrix
    matrix = np.array([input().split() for _ in range(n)], float)
    
    # Compute determinant
    det = np.linalg.det(matrix)
    
    # Round to 2 decimal places
    print(round(det, 2))

#%% NumPy Comprehensive Reference
"""
NUMPY CHEAT SHEET:

=== Array Creation ===
np.array([1, 2, 3])              # From list
np.zeros((3, 4))                 # 3x4 array of zeros
np.ones((2, 3))                  # 2x3 array of ones
np.full((2, 2), 7)               # 2x2 array filled with 7
np.eye(3)                        # 3x3 identity matrix
np.eye(3, 4)                     # 3x4 identity matrix
np.arange(0, 10, 2)              # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)             # 5 evenly spaced values from 0 to 1
np.random.rand(3, 4)             # 3x4 random values [0, 1)
np.random.randn(3, 4)            # 3x4 random normal distribution
np.random.randint(0, 10, (3,4))  # 3x4 random integers [0, 10)

=== Array Attributes ===
arr.shape                        # Dimensions (rows, cols)
arr.size                         # Total number of elements
arr.ndim                         # Number of dimensions
arr.dtype                        # Data type of elements

=== Reshaping ===
arr.reshape(3, 4)                # Reshape to 3x4
arr.flatten()                    # Flatten to 1D
arr.ravel()                      # Flatten (returns view if possible)
arr.T                            # Transpose
np.transpose(arr)                # Transpose
arr.resize(3, 4)                 # Resize in-place

=== Indexing & Slicing ===
arr[0]                           # First row
arr[:, 0]                        # First column
arr[0, 1]                        # Element at row 0, col 1
arr[1:3, :]                      # Rows 1-2, all columns
arr[[0, 2], :]                   # Rows 0 and 2
arr[arr > 5]                     # Boolean indexing

=== Array Operations (Element-wise) ===
np.add(a, b)        or  a + b    # Addition
np.subtract(a, b)   or  a - b    # Subtraction
np.multiply(a, b)   or  a * b    # Multiplication
np.divide(a, b)     or  a / b    # Division
np.floor_divide(a, b) or a // b  # Floor division
np.mod(a, b)        or  a % b    # Modulo
np.power(a, b)      or  a ** b   # Power
np.sqrt(a)                       # Square root
np.exp(a)                        # Exponential
np.log(a)                        # Natural log

=== Mathematical Functions ===
np.floor(arr)                    # Round down
np.ceil(arr)                     # Round up
np.rint(arr)                     # Round to nearest integer
np.round(arr, decimals=2)        # Round to decimals
np.abs(arr)                      # Absolute value
np.sign(arr)                     # Sign (-1, 0, 1)

=== Aggregation Functions ===
np.sum(arr)                      # Sum all elements
np.sum(arr, axis=0)              # Sum along axis 0 (columns)
np.sum(arr, axis=1)              # Sum along axis 1 (rows)
np.prod(arr)                     # Product
np.mean(arr)                     # Mean
np.median(arr)                   # Median
np.std(arr)                      # Standard deviation
np.var(arr)                      # Variance
np.min(arr)                      # Minimum
np.max(arr)                      # Maximum
np.argmin(arr)                   # Index of minimum
np.argmax(arr)                   # Index of maximum

=== Linear Algebra ===
np.dot(a, b)         or  a @ b   # Matrix multiplication / dot product
np.inner(a, b)                   # Inner product
np.outer(a, b)                   # Outer product
np.cross(a, b)                   # Cross product
np.linalg.det(arr)               # Determinant
np.linalg.inv(arr)               # Inverse
np.linalg.eig(arr)               # Eigenvalues and eigenvectors
np.linalg.solve(A, b)            # Solve Ax = b
np.linalg.norm(arr)              # Norm

=== Array Manipulation ===
np.concatenate((a, b), axis=0)   # Concatenate arrays
np.vstack((a, b))                # Vertical stack
np.hstack((a, b))                # Horizontal stack
np.split(arr, 3)                 # Split into 3 parts
np.vsplit(arr, 2)                # Vertical split
np.hsplit(arr, 2)                # Horizontal split

=== Broadcasting ===
# Arrays with different shapes can be operated together
a = np.array([[1, 2, 3]])        # Shape (1, 3)
b = np.array([[1], [2], [3]])    # Shape (3, 1)
c = a + b                        # Shape (3, 3) - broadcasting

=== Polynomial Operations ===
np.poly1d([1, 2, 3])             # Create polynomial x² + 2x + 3
np.polyval(coeffs, x)            # Evaluate polynomial at x
np.roots([1, 2, 3])              # Find roots
np.polyfit(x, y, degree)         # Fit polynomial

=== Boolean Operations ===
arr > 5                          # Boolean array
np.where(arr > 5, 1, 0)          # Conditional
np.logical_and(a, b)             # Element-wise AND
np.logical_or(a, b)              # Element-wise OR
np.logical_not(a)                # Element-wise NOT
np.all(arr > 0)                  # True if all elements match
np.any(arr > 0)                  # True if any element matches

=== Axis Explanation ===
For 2D array:
- axis=0: operations along columns (↓)
- axis=1: operations along rows (→)
- axis=None: operations on entire array

Example:
arr = [[1, 2, 3],
       [4, 5, 6]]

np.sum(arr, axis=0) → [5, 7, 9]   # Column sums
np.sum(arr, axis=1) → [6, 15]     # Row sums
np.sum(arr)         → 21           # Total sum
"""

#%% NumPy Performance Tips
"""
Performance and Best Practices:

1. Vectorization - avoid loops
   Bad:  result = [x**2 for x in arr]
   Good: result = arr ** 2

2. In-place operations save memory
   arr += 1      # Better than arr = arr + 1

3. Use views instead of copies when possible
   view = arr[1:3]    # Creates view
   copy = arr[1:3].copy()  # Creates copy

4. Pre-allocate arrays
   result = np.zeros((1000, 1000))
   # Fill result...

5. Use appropriate data types
   arr = np.array([1, 2, 3], dtype=np.int32)  # Saves memory

6. Broadcasting is faster than loops
   arr + 5  # Much faster than looping

7. Use built-in NumPy functions
   np.sum(arr)  # Faster than sum(arr)
"""

#%% Common NumPy Patterns
"""
Practical patterns and examples
"""

import numpy as np

def numpy_patterns():
    print("=== Pattern 1: Normalization ===")
    arr = np.array([1, 2, 3, 4, 5])
    normalized = (arr - np.mean(arr)) / np.std(arr)
    print(f"Original: {arr}")
    print(f"Normalized: {normalized}")
    
    print("\n=== Pattern 2: Finding Indices ===")
    arr = np.array([10, 20, 30, 40, 50])
    indices = np.where(arr > 25)
    print(f"Indices where arr > 25: {indices}")
    
    print("\n=== Pattern 3: Matrix Operations ===")
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    print(f"A @ B (matrix mult):\n{A @ B}")
    print(f"A * B (element-wise):\n{A * B}")
    
    print("\n=== Pattern 4: Statistical Analysis ===")
    data = np.random.randn(100)
    print(f"Mean: {np.mean(data):.3f}")
    print(f"Std: {np.std(data):.3f}")
    print(f"Median: {np.median(data):.3f}")
    
    print("\n=== Pattern 5: Filtering ===")
    arr = np.array([1, 2, 3, 4, 5, 6])
    even = arr[arr % 2 == 0]
    print(f"Even numbers: {even}")

# Uncomment to run
# numpy_patterns()
# ```

# **Key Concepts Covered:**

# 1. **Array Creation**: zeros, ones, eye, arange, linspace
# 2. **Reshaping**: reshape, flatten, transpose
# 3. **Array Operations**: Element-wise operations, broadcasting
# 4. **Aggregations**: sum, mean, var, std, min, max
# 5. **Linear Algebra**: dot product, determinant, matrix operations
# 6. **Axis Operations**: Understanding axis=0 (columns) and axis=1 (rows)

# **Important Distinctions:**

# | Operation | Description | Example |
# |-----------|-------------|---------|
# | `*` | Element-wise multiplication | `[1,2] * [3,4]` = `[3,8]` |
# | `@` or `np.dot()` | Matrix multiplication | `[[1,2]] @ [[3],[4]]` = `[[11]]` |
# | `np.inner()` | Inner product | Sum of element-wise products |
# | `np.outer()` | Outer product | All pairs multiplication |

# **Axis Understanding:**
# ```
# Array: [[1, 2, 3],
#         [4, 5, 6]]

# axis=0 (↓): Column operations → [5, 7, 9]
# axis=1 (→): Row operations → [6, 15]
# axis=None: Entire array → 21