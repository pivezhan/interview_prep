#%% itertools.product()
"""
Problem: Compute cartesian product of two lists
Description: Given two lists A and B, print all possible combinations (a, b) 
            where a is from A and b is from B
Input: List A (space-separated integers), List B (space-separated integers)
Output: All tuples from cartesian product, space-separated
"""

from itertools import product

if __name__ == '__main__':
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    result = list(product(A, B))
    
    # Print each tuple separated by space
    print(*result)

#%% itertools.permutations()
"""
Problem: Print all permutations of a string
Description: Given a string and size k, print all permutations of length k
            in lexicographically sorted order
Input: String S and integer k (space-separated)
Output: All permutations of length k, sorted, one per line
"""

from itertools import permutations

if __name__ == '__main__':
    S, k = input().split()
    k = int(k)
    
    # Generate all permutations of length k
    perms = permutations(S, k)
    
    # Sort and print
    sorted_perms = sorted(perms)
    
    for perm in sorted_perms:
        print(''.join(perm))

#%% Alternative solution for permutations (one-liner style)
"""
More concise version using sorted string
"""

from itertools import permutations

if __name__ == '__main__':
    S, k = input().split()
    
    # Sort string first, then generate permutations
    for perm in permutations(sorted(S), int(k)):
        print(''.join(perm))

#%% itertools.combinations()
"""
Problem: Print all combinations of a string
Description: Given a string and size k, print all combinations from size 1 to k
            in lexicographically sorted order
Input: String S and integer k (space-separated)
Output: All combinations grouped by size, sorted, one per line
"""

from itertools import combinations

if __name__ == '__main__':
    S, k = input().split()
    k = int(k)
    
    # Sort the string first
    S = sorted(S)
    
    # Generate combinations for each size from 1 to k
    for size in range(1, k + 1):
        combs = combinations(S, size)
        for comb in combs:
            print(''.join(comb))

#%% itertools.combinations_with_replacement()
"""
Problem: Print all combinations with replacement
Description: Given a string and size k, print all combinations of length k
            where elements can be repeated, in lexicographically sorted order
Input: String S and integer k (space-separated)
Output: All combinations with replacement, sorted, one per line
"""

from itertools import combinations_with_replacement

if __name__ == '__main__':
    S, k = input().split()
    k = int(k)
    
    # Sort the string first
    S = sorted(S)
    
    # Generate combinations with replacement
    combs = combinations_with_replacement(S, k)
    
    for comb in combs:
        print(''.join(comb))

#%% Compress the String!
"""
Problem: Group consecutive identical characters
Description: Use groupby to compress a string by grouping consecutive characters
            and output (count, int_value) tuples
Input: String of digits
Output: Space-separated tuples of (count, integer_value)
"""

from itertools import groupby

if __name__ == '__main__':
    S = input().strip()
    
    result = []
    
    # Group consecutive characters
    for key, group in groupby(S):
        count = len(list(group))
        result.append((count, int(key)))
    
    # Print tuples separated by space
    print(*result)

#%% Alternative solution for Compress the String
"""
More concise version using generator expression
"""

from itertools import groupby

if __name__ == '__main__':
    S = input().strip()
    
    result = [(len(list(group)), int(key)) for key, group in groupby(S)]
    print(*result)

#%% Iterables and Iterators
"""
Problem: Find probability of selecting at least one 'a'
Description: Given a list and k indices to choose, calculate probability 
            that at least one selected index contains letter 'a'
Input: N (list size), list of letters, K (number of indices to select)
Output: Probability with at least 3 decimal places
"""

from itertools import combinations

if __name__ == '__main__':
    N = int(input())
    letters = input().split()
    K = int(input())
    
    # Generate all possible combinations of K indices
    all_combinations = list(combinations(range(N), K))
    
    # Count combinations that contain at least one 'a'
    combinations_with_a = 0
    
    for combo in all_combinations:
        # Check if any index in this combination has 'a'
        if any(letters[i] == 'a' for i in combo):
            combinations_with_a += 1
    
    # Calculate probability
    probability = combinations_with_a / len(all_combinations)
    print(probability)

#%% Alternative solution for Iterables and Iterators (more efficient)
"""
Using combination indices directly with letters
"""

from itertools import combinations

if __name__ == '__main__':
    N = int(input())
    letters = input().split()
    K = int(input())
    
    # Get all combinations of indices
    all_combos = list(combinations(range(N), K))
    
    # Count combos with at least one 'a'
    count_with_a = sum(1 for combo in all_combos 
                       if any(letters[i] == 'a' for i in combo))
    
    # Calculate and print probability
    print(count_with_a / len(all_combos))

#%% Maximize It!
"""
Problem: Maximize sum of squared elements modulo M
Description: Given K lists, select one element from each list to maximize
            the sum of their squares modulo M
Input: K and M, then K lists of varying sizes
Output: Maximum value of (sum of squares) % M
"""

from itertools import product

if __name__ == '__main__':
    K, M = map(int, input().split())
    
    lists = []
    for _ in range(K):
        line = list(map(int, input().split()))
        N = line[0]  # First element is the count
        elements = line[1:]  # Rest are the actual elements
        lists.append(elements)
    
    max_value = 0
    
    # Generate all combinations using product
    for combination in product(*lists):
        # Calculate sum of squares mod M
        current_sum = sum(x**2 for x in combination) % M
        max_value = max(max_value, current_sum)
    
    print(max_value)

#%% Alternative solution for Maximize It! (optimized)
"""
More efficient version calculating directly without intermediate variables
"""

from itertools import product

if __name__ == '__main__':
    K, M = map(int, input().split())
    
    # Read all lists
    lists = []
    for _ in range(K):
        nums = list(map(int, input().split()))
        lists.append(nums[1:])  # Skip first element (count)
    
    # Find maximum using generator expression
    max_value = max(sum(x**2 for x in combo) % M for combo in product(*lists))
    
    print(max_value)

#%% Comprehensive itertools demonstration
"""
Demonstration: All itertools functions used in HackerRank problems
This example shows the behavior of each function
"""

from itertools import (product, permutations, combinations, 
                       combinations_with_replacement, groupby)

def demonstrate_itertools():
    print("=== itertools.product ===")
    # Cartesian product of input iterables
    A = [1, 2]
    B = [3, 4]
    print(f"product({A}, {B}):")
    print(list(product(A, B)))
    # Output: [(1, 3), (1, 4), (2, 3), (2, 4)]
    
    print("\n=== itertools.permutations ===")
    # All permutations of given length
    S = "ABC"
    print(f"permutations('{S}', 2):")
    print(list(permutations(S, 2)))
    # Output: [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]
    
    print("\n=== itertools.combinations ===")
    # Combinations without replacement (no repeating elements)
    print(f"combinations('{S}', 2):")
    print(list(combinations(S, 2)))
    # Output: [('A', 'B'), ('A', 'C'), ('B', 'C')]
    
    print("\n=== itertools.combinations_with_replacement ===")
    # Combinations with replacement (elements can repeat)
    print(f"combinations_with_replacement('{S}', 2):")
    print(list(combinations_with_replacement(S, 2)))
    # Output: [('A', 'A'), ('A', 'B'), ('A', 'C'), ('B', 'B'), ('B', 'C'), ('C', 'C')]
    
    print("\n=== itertools.groupby ===")
    # Group consecutive equal elements
    data = "AAABBBCC"
    print(f"groupby('{data}'):")
    result = [(key, len(list(group))) for key, group in groupby(data)]
    print(result)
    # Output: [('A', 3), ('B', 3), ('C', 2)]
    
    print("\n=== Differences Summary ===")
    print("permutations: Order matters, no replacement")
    print("combinations: Order doesn't matter, no replacement")
    print("combinations_with_replacement: Order doesn't matter, with replacement")
    print("product: Cartesian product of multiple iterables")

# Uncomment to run demonstration
# demonstrate_itertools()

#%% Key differences cheat sheet
"""
ITERTOOLS CHEAT SHEET:

1. product(A, B):
   - Cartesian product: all pairs (a, b) where a∈A, b∈B
   - Example: product([1,2], [3,4]) → (1,3), (1,4), (2,3), (2,4)

2. permutations(iterable, r):
   - All r-length permutations (order matters)
   - No repetition of elements
   - Example: permutations('AB', 2) → AB, BA

3. combinations(iterable, r):
   - All r-length combinations (order doesn't matter)
   - No repetition of elements
   - Example: combinations('ABC', 2) → AB, AC, BC

4. combinations_with_replacement(iterable, r):
   - All r-length combinations (order doesn't matter)
   - Elements CAN be repeated
   - Example: combinations_with_replacement('AB', 2) → AA, AB, BB

5. groupby(iterable, key=None):
   - Groups consecutive equal elements
   - Returns iterator of (key, group) tuples
   - Example: groupby('AAABBC') → (A,3), (B,2), (C,1)

Complexity Notes:
- product: O(n^k) where n is length, k is number of iterables
- permutations: O(n!/(n-r)!)
- combinations: O(n!/(r!(n-r)!))
- groupby: O(n)
"""