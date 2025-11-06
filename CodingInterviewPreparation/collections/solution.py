#%% DefaultDict Tutorial
"""
Problem: Use defaultdict to track word positions
Description: Given two groups of words, for each word in group B, 
            print the positions where it appears in group A (or -1 if not found)
Input: n (size of group A), m (size of group B), n words of group A, m words of group B
Output: For each word in group B, print space-separated positions (1-indexed) or -1
"""

from collections import defaultdict

if __name__ == '__main__':
    n, m = map(int, input().split())
    
    # Create defaultdict to store positions of words in group A
    d = defaultdict(list)
    
    # Read group A and store positions (1-indexed)
    for i in range(1, n + 1):
        word = input().strip()
        d[word].append(i)
    
    # Read group B and print positions
    for _ in range(m):
        word = input().strip()
        if word in d:
            print(' '.join(map(str, d[word])))
        else:
            print(-1)

#%% collections.Counter()
"""
Problem: Find the amount earned from selling shoes
Description: Given shoe sizes in stock and customer requests, 
            calculate total money earned when matching sizes are available
Input: Number of shoes X, shoe sizes, number of customers N, 
       then N lines with size and price customer is willing to pay
Output: Total money earned
"""

from collections import Counter

if __name__ == '__main__':
    X = int(input())
    shoe_sizes = Counter(map(int, input().split()))
    N = int(input())
    
    total_earned = 0
    
    for _ in range(N):
        size, price = map(int, input().split())
        
        # Check if shoe size is available
        if shoe_sizes[size] > 0:
            total_earned += price
            shoe_sizes[size] -= 1
    
    print(total_earned)

#%% Collections.namedtuple()
"""
Problem: Calculate average marks using namedtuple
Description: Given student records with ID, MARKS, NAME, CLASS, 
            calculate the average of MARKS column
Input: Number of students N, column names, N rows of student data
Output: Average marks
"""

from collections import namedtuple

if __name__ == '__main__':
    n = int(input())
    columns = input().split()
    
    # Create namedtuple class
    Student = namedtuple('Student', columns)
    
    total_marks = 0
    
    for _ in range(n):
        values = input().split()
        student = Student(*values)
        total_marks += int(student.MARKS)
    
    average = total_marks / n
    print(average)

#%% Collections.OrderedDict()
"""
Problem: Calculate net price for each item using OrderedDict
Description: Keep track of items in order of first appearance, 
            summing quantities for repeated items
Input: Number of items N, then N lines with item_name and net_price
Output: Each item with its total net price in order of first appearance
"""

from collections import OrderedDict

if __name__ == '__main__':
    n = int(input())
    
    items = OrderedDict()
    
    for _ in range(n):
        line = input().rsplit(' ', 1)  # Split from right to handle spaces in item names
        item_name = line[0]
        net_price = int(line[1])
        
        if item_name in items:
            items[item_name] += net_price
        else:
            items[item_name] = net_price
    
    for item, price in items.items():
        print(item, price)

#%% Word Order
"""
Problem: Count word occurrences and maintain order
Description: Print number of distinct words and their occurrence counts in order of appearance
Input: Number of words n, then n words
Output: Number of distinct words on first line, 
        space-separated occurrence counts in order of first appearance
"""

from collections import OrderedDict

if __name__ == '__main__':
    n = int(input())
    
    word_count = OrderedDict()
    
    for _ in range(n):
        word = input().strip()
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    
    print(len(word_count))
    print(' '.join(map(str, word_count.values())))

#%% Collections.deque()
"""
Problem: Perform deque operations
Description: Execute append, appendleft, pop, popleft operations on a deque
Input: Number of operations N, then N lines with operation commands
Output: Final deque elements space-separated
"""

from collections import deque

if __name__ == '__main__':
    n = int(input())
    d = deque()
    
    for _ in range(n):
        command = input().split()
        operation = command[0]
        
        if operation == 'append':
            d.append(int(command[1]))
        elif operation == 'appendleft':
            d.appendleft(int(command[1]))
        elif operation == 'pop':
            d.pop()
        elif operation == 'popleft':
            d.popleft()
    
    print(' '.join(map(str, d)))

#%% Company Logo
"""
Problem: Find three most common characters in a string
Description: Print the three most common characters with their counts, 
            sorted by frequency (descending) then alphabetically
Input: String s
Output: Three lines with character and count, most common first
"""

from collections import Counter

if __name__ == '__main__':
    s = input().strip()

    # Count character frequencies
    char_count = Counter(s)

    # Sort by count (descending) then by character (ascending)
    sorted_chars = sorted(char_count.items(), key=lambda x: (-x[1], x[0]))

    # Print top 3
    for i in range(3):
        print(sorted_chars[i][0], sorted_chars[i][1])

#%% Alternative solution for Company Logo using most_common
"""
Alternative approach using Counter's most_common method with custom sorting
"""

from collections import Counter

if __name__ == '__main__':
    s = input().strip()
    
    # Count and sort
    char_count = Counter(sorted(s)).most_common()
    
    # Sort by frequency (descending) then alphabetically
    char_count.sort(key=lambda x: (-x[1], x[0]))
    
    # Print top 3
    for i in range(3):
        print(char_count[i][0], char_count[i][1])

#%% Piling Up!
"""
Problem: Check if blocks can be stacked in non-increasing order
Description: Given side lengths of blocks, determine if you can pick from either end
            and create a pile where each block's side length >= block above it
Input: Number of test cases T, for each test case: number of blocks n, then side lengths
Output: "Yes" if possible, "No" otherwise for each test case
"""

from collections import deque

if __name__ == '__main__':
    t = int(input())
    
    for _ in range(t):
        n = int(input())
        blocks = deque(map(int, input().split()))
        
        can_pile = True
        last_block = float('inf')  # Start with infinity (no block on top yet)

        while blocks:
            # Choose the larger of the two ends
            if blocks[0] >= blocks[-1]:
                current = blocks.popleft()
            else:
                current = blocks.pop()
            
            # Check if current block can be placed (must be <= last block placed)
            if current > last_block:
                can_pile = False
                break
            
            last_block = current
        
        print("Yes" if can_pile else "No")

#%% Alternative solution for Piling Up! (more efficient)
"""
Alternative approach without using deque - checking if blocks form valid pile
"""

if __name__ == '__main__':
    t = int(input())
    
    for _ in range(t):
        n = int(input())
        blocks = list(map(int, input().split()))
        
        left = 0
        right = n - 1
        last_block = float('inf')
        can_pile = True
        
        while left <= right:
            # Choose larger from left or right end
            if blocks[left] >= blocks[right]:
                current = blocks[left]
                left += 1
            else:
                current = blocks[right]
                right -= 1
            
            # Check if valid placement
            if current > last_block:
                can_pile = False
                break
            
            last_block = current
        
        print("Yes" if can_pile else "No")

#%% Comprehensive example demonstrating all collections
"""
Demonstration: Using multiple collection types together
This example shows how different collections can work together
"""

from collections import defaultdict, Counter, OrderedDict, namedtuple, deque

def demonstrate_collections():
    # defaultdict - automatic default values
    word_positions = defaultdict(list)
    word_positions['apple'].append(1)
    word_positions['banana'].append(2)
    
    # Counter - counting elements
    fruits = ['apple', 'banana', 'apple', 'cherry', 'banana', 'banana']
    fruit_count = Counter(fruits)
    print("Most common:", fruit_count.most_common(2))
    
    # OrderedDict - maintains insertion order (Python 3.7+ dicts also maintain order)
    ordered_items = OrderedDict()
    ordered_items['first'] = 1
    ordered_items['second'] = 2
    
    # namedtuple - immutable tuple with named fields
    Person = namedtuple('Person', ['name', 'age', 'city'])
    person = Person('Alice', 30, 'NYC')
    print(f"Person: {person.name}, Age: {person.age}")
    
    # deque - double-ended queue for efficient append/pop from both ends
    d = deque([1, 2, 3])
    d.appendleft(0)  # [0, 1, 2, 3]
    d.append(4)       # [0, 1, 2, 3, 4]
    d.popleft()       # [1, 2, 3, 4]
    
    print("Collections demonstration complete!")

# Uncomment to run demonstration
# demonstrate_collections()