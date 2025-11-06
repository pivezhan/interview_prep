#%% List Comprehensions
"""
Problem: Generate all possible coordinates (i, j, k) where i+j+k != n
Description: Given dimensions x, y, z and integer n, generate all coordinates
            where the sum is not equal to n using list comprehension
Input: Four integers x, y, z, n on separate lines
Output: List of all valid coordinates
"""

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    
    # List comprehension to generate all valid coordinates
    result = [[i, j, k] for i in range(x + 1) 
              for j in range(y + 1) 
              for k in range(z + 1) 
              if i + j + k != n]
    
    print(result)

#%% Find the Runner-Up Score!
"""
Problem: Find the second highest score
Description: Given a list of scores, find the runner-up (second highest) score
Input: n (number of scores), array of n scores
Output: Runner-up score
"""

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    
    # Convert to set to remove duplicates, then sort
    scores = sorted(set(arr), reverse=True)
    
    # Second element is the runner-up
    print(scores[1])

#%% Alternative solution for Runner-Up Score
"""
More efficient approach without sorting entire list
"""

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    
    # Remove duplicates and find max
    unique_scores = set(arr)
    max_score = max(unique_scores)
    unique_scores.remove(max_score)
    
    # Runner-up is the max of remaining scores
    print(max(unique_scores))

#%% Nested Lists
"""
Problem: Find students with second lowest grade
Description: Given student names and grades, find all students with 
            the second lowest grade, sorted alphabetically
Input: Number of students n, then n pairs of (name, grade)
Output: Names of students with second lowest grade, sorted, one per line
"""

if __name__ == '__main__':
    students = []
    
    for _ in range(int(input())):
        name = input()
        score = float(input())
        students.append([name, score])
    
    # Get unique grades and sort them
    grades = sorted(set([score for name, score in students]))
    
    # Second lowest grade
    second_lowest = grades[1]
    
    # Find all students with second lowest grade
    result = sorted([name for name, score in students if score == second_lowest])
    
    # Print each name on a new line
    for name in result:
        print(name)

#%% Alternative solution for Nested Lists (more concise)
"""
Using list comprehension and lambda functions
"""

if __name__ == '__main__':
    n = int(input())
    students = [[input(), float(input())] for _ in range(n)]
    
    # Find second lowest grade
    second_lowest = sorted(set([score for name, score in students]))[1]
    
    # Print names with second lowest grade, sorted
    for name in sorted([name for name, score in students if score == second_lowest]):
        print(name)

#%% Finding the percentage
"""
Problem: Calculate average marks for a student
Description: Given student records with marks in three subjects,
            calculate average for a queried student
Input: n (number of students), n lines of student data, query name
Output: Average marks with 2 decimal places
"""

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    
    query_name = input()
    
    # Calculate average
    marks = student_marks[query_name]
    average = sum(marks) / len(marks)
    
    # Print with 2 decimal places
    print(f"{average:.2f}")

#%% Lists
"""
Problem: Perform various list operations
Description: Execute insert, print, remove, append, sort, pop, reverse commands
Input: Number of commands N, then N command lines
Output: Print the list when 'print' command is encountered
"""

if __name__ == '__main__':
    n = int(input())
    lst = []
    
    for _ in range(n):
        command = input().split()
        cmd = command[0]
        
        if cmd == 'insert':
            lst.insert(int(command[1]), int(command[2]))
        elif cmd == 'print':
            print(lst)
        elif cmd == 'remove':
            lst.remove(int(command[1]))
        elif cmd == 'append':
            lst.append(int(command[1]))
        elif cmd == 'sort':
            lst.sort()
        elif cmd == 'pop':
            lst.pop()
        elif cmd == 'reverse':
            lst.reverse()

#%% Tuples
"""
Problem: Calculate hash of a tuple
Description: Given n integers, create a tuple and return its hash value
Input: n (number of integers), tuple of n space-separated integers
Output: Hash value of the tuple
"""

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    
    # Convert to tuple and calculate hash
    t = tuple(integer_list)
    print(hash(t))

#%% List Comprehensions - Advanced Examples
"""
Additional examples demonstrating list comprehension techniques
"""

def list_comprehension_examples():
    # Basic list comprehension
    squares = [x**2 for x in range(10)]
    print(f"Squares: {squares}")
    
    # With condition
    even_squares = [x**2 for x in range(10) if x % 2 == 0]
    print(f"Even squares: {even_squares}")
    
    # Nested list comprehension
    matrix = [[i*j for j in range(3)] for i in range(3)]
    print(f"Matrix: {matrix}")
    
    # Flattening a 2D list
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    flat = [num for row in matrix for num in row]
    print(f"Flattened: {flat}")
    
    # Using multiple conditions
    result = [x for x in range(20) if x % 2 == 0 if x % 3 == 0]
    print(f"Divisible by 2 and 3: {result}")
    
    # If-else in list comprehension
    labels = ["even" if x % 2 == 0 else "odd" for x in range(10)]
    print(f"Labels: {labels}")

# Uncomment to run examples
# list_comprehension_examples()

#%% Data Type Operations Cheat Sheet
"""
PYTHON BASIC DATA TYPES OPERATIONS:

=== LISTS (Mutable, Ordered) ===
Creation:
  lst = [1, 2, 3]
  lst = list(range(5))
  lst = [x**2 for x in range(5)]  # List comprehension

Common Operations:
  lst.append(x)         # Add to end - O(1)
  lst.insert(i, x)      # Insert at position - O(n)
  lst.remove(x)         # Remove first occurrence - O(n)
  lst.pop()             # Remove and return last - O(1)
  lst.pop(i)            # Remove at index - O(n)
  lst.sort()            # Sort in place - O(n log n)
  lst.reverse()         # Reverse in place - O(n)
  lst.index(x)          # Find first index - O(n)
  lst.count(x)          # Count occurrences - O(n)
  lst.extend(lst2)      # Extend list - O(k)
  
Access:
  lst[i]                # Get element - O(1)
  lst[start:end]        # Slice - O(k)
  len(lst)              # Length - O(1)
  min(lst), max(lst)    # Min/Max - O(n)
  sum(lst)              # Sum - O(n)

=== TUPLES (Immutable, Ordered) ===
Creation:
  tup = (1, 2, 3)
  tup = 1, 2, 3         # Parentheses optional
  tup = tuple([1, 2, 3])

Operations:
  tup[i]                # Access element - O(1)
  tup.count(x)          # Count occurrences - O(n)
  tup.index(x)          # Find index - O(n)
  hash(tup)             # Hash value (if all elements hashable)
  
Use Cases:
  - Immutable data
  - Dictionary keys
  - Function return multiple values
  - Faster than lists for fixed data

=== SETS (Mutable, Unordered, Unique) ===
Creation:
  s = {1, 2, 3}
  s = set([1, 2, 2, 3]) # Removes duplicates

Operations:
  s.add(x)              # Add element - O(1)
  s.remove(x)           # Remove (raises error if not found) - O(1)
  s.discard(x)          # Remove (no error) - O(1)
  s.pop()               # Remove arbitrary element - O(1)
  s.clear()             # Remove all - O(n)
  
Set Operations:
  s1.union(s2)          # s1 | s2
  s1.intersection(s2)   # s1 & s2
  s1.difference(s2)     # s1 - s2
  s1.symmetric_difference(s2)  # s1 ^ s2

=== DICTIONARIES (Mutable, Key-Value Pairs) ===
Creation:
  d = {'a': 1, 'b': 2}
  d = dict(a=1, b=2)

Operations:
  d[key]                # Get value - O(1)
  d[key] = value        # Set value - O(1)
  d.get(key, default)   # Safe get - O(1)
  d.pop(key)            # Remove and return - O(1)
  d.keys()              # All keys
  d.values()            # All values
  d.items()             # All (key, value) pairs
  key in d              # Check existence - O(1)

=== COMMON PATTERNS ===

Finding Maximum/Minimum:
  max_val = max(lst)
  min_val = min(lst)
  max_with_key = max(lst, key=lambda x: x[1])  # By second element

Sorting:
  sorted_lst = sorted(lst)                     # Returns new list
  lst.sort()                                   # In-place
  sorted_desc = sorted(lst, reverse=True)
  sorted_by_key = sorted(lst, key=lambda x: x[1])

Filtering:
  filtered = [x for x in lst if condition]
  filtered = filter(lambda x: condition, lst)

Mapping:
  mapped = [f(x) for x in lst]
  mapped = list(map(f, lst))

Removing Duplicates:
  unique = list(set(lst))                      # Unordered
  unique = list(dict.fromkeys(lst))            # Preserves order (Python 3.7+)

Counting:
  from collections import Counter
  counts = Counter(lst)

Finding Second Max:
  sorted_unique = sorted(set(lst), reverse=True)
  second_max = sorted_unique[1]

Flattening:
  flat = [item for sublist in nested_list for item in sublist]
"""

#%% Performance Comparison
"""
Time Complexity Comparison:

Operation          | List    | Tuple   | Set     | Dict
-------------------|---------|---------|---------|--------
Access by index    | O(1)    | O(1)    | N/A     | N/A
Access by key      | N/A     | N/A     | N/A     | O(1)
Search (in)        | O(n)    | O(n)    | O(1)    | O(1)
Insert at end      | O(1)    | N/A     | O(1)    | O(1)
Insert in middle   | O(n)    | N/A     | N/A     | N/A
Delete             | O(n)    | N/A     | O(1)    | O(1)
Iterate            | O(n)    | O(n)    | O(n)    | O(n)

Memory Usage (ascending):
Tuple < List < Set < Dict

Choose:
- List: When you need order and mutability
- Tuple: When you need immutability and hashability
- Set: When you need unique elements and fast lookup
- Dict: When you need key-value mapping with fast lookup
"""

#%% Common Interview Patterns
"""
Pattern demonstrations for common problems
"""

def common_patterns():
    # Pattern 1: Two Pointers
    def remove_duplicates_sorted(lst):
        if not lst:
            return 0
        i = 0
        for j in range(1, len(lst)):
            if lst[j] != lst[i]:
                i += 1
                lst[i] = lst[j]
        return i + 1
    
    # Pattern 2: Sliding Window
    def max_sum_subarray(arr, k):
        max_sum = sum(arr[:k])
        window_sum = max_sum
        for i in range(k, len(arr)):
            window_sum = window_sum - arr[i-k] + arr[i]
            max_sum = max(max_sum, window_sum)
        return max_sum
    
    # Pattern 3: Hash Map for Counting
    def first_non_repeating(s):
        count = {}
        for char in s:
            count[char] = count.get(char, 0) + 1
        for char in s:
            if count[char] == 1:
                return char
        return None
    
    # Pattern 4: Set for Lookup
    def has_pair_with_sum(arr, target):
        seen = set()
        for num in arr:
            if target - num in seen:
                return True
            seen.add(num)
        return False
    
    # Demonstrations
    print("Pattern demonstrations:")
    print(f"Remove duplicates: {remove_duplicates_sorted([1,1,2,2,3,4,4])}")
    print(f"Max sum subarray: {max_sum_subarray([1,2,3,4,5], 3)}")
    print(f"First non-repeating: {first_non_repeating('aabbcde')}")
    print(f"Has pair with sum: {has_pair_with_sum([1,2,3,4], 7)}")

# Uncomment to run
# common_patterns()