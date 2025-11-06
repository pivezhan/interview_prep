#%% No Idea!
"""
Problem: Compute your happiness based on sets A and B
Description: Given an array of integers and two sets A and B, calculate happiness:
            - Add 1 for each element in A
            - Subtract 1 for each element in B
Input: n (array size), m (set sizes), array, set A, set B
Output: Total happiness value
"""

if __name__ == '__main__':
    n, m = map(int, input().split())
    array = list(map(int, input().split()))
    A = set(map(int, input().split()))
    B = set(map(int, input().split()))
    
    happiness = 0
    for num in array:
        if num in A:
            happiness += 1
        elif num in B:
            happiness -= 1
    
    print(happiness)

#%% Introduction to Sets
"""
Problem: Calculate average of distinct heights
Description: Given a list of student heights, find the average of unique heights
Input: Number of students n, then n space-separated heights
Output: Average of distinct heights
"""

def average(array):
    distinct_heights = set(array)
    return sum(distinct_heights) / len(distinct_heights)

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)

#%% Symmetric Difference
"""
Problem: Find symmetric difference of two sets
Description: Output elements that are in either set A or B but not in both
Input: Size of set A, elements of A, size of set B, elements of B
Output: Symmetric difference in ascending order, one per line
"""

if __name__ == '__main__':
    m = int(input())
    set_m = set(map(int, input().split()))
    n = int(input())
    set_n = set(map(int, input().split()))
    
    symmetric_diff = set_m.symmetric_difference(set_n)
    result = sorted(symmetric_diff)
    
    for num in result:
        print(num)

#%% Set .add()
"""
Problem: Count distinct countries
Description: Given a number of country names, count how many distinct countries there are
Input: Number of entries n, then n country names
Output: Number of distinct countries
"""

if __name__ == '__main__':
    n = int(input())
    countries = set()
    
    for _ in range(n):
        country = input().strip()
        countries.add(country)
    
    print(len(countries))

#%% Set .discard(), .remove() & .pop()
"""
Problem: Perform various set operations
Description: Execute commands like pop, remove, and discard on a set
Input: Set size n, set elements, number of commands, then commands
Output: Sum of remaining elements in the set
"""

if __name__ == '__main__':
    # Enter your code here. Read input from STDIN. Print output to STDOUT
    # Enter your code here. Read input from STDIN. Print output to STDOUT

    n = int(input())
    s = set(map(int, input().split()))
    num_commands = int(input())

    for _ in range(num_commands):
        cmd = input().split()
        if cmd[0] == 'pop':
            s.pop()
        elif cmd[0] == 'remove':
            try:
                s.remove(int(cmd[1]))
            except KeyError:
                pass
        elif cmd[0] == 'discard':
            s.discard(int(cmd[1]))

    print(sum(s))
#%% Set .union() Operation
"""
Problem: Find number of students subscribed to at least one newspaper
Description: Count distinct student IDs from two newspaper subscriptions
Input: English newspaper subscriber count, roll numbers, French newspaper count, roll numbers
Output: Total number of distinct students
"""

if __name__ == '__main__':
    n = int(input())
    english_subscribers = set(map(int, input().split()))
    b = int(input())
    french_subscribers = set(map(int, input().split()))
    
    all_subscribers = english_subscribers.union(french_subscribers)
    print(len(all_subscribers))

#%% Set .intersection() Operation
"""
Problem: Find number of students subscribed to both newspapers
Description: Count student IDs that appear in both English and French subscriptions
Input: English subscriber count, roll numbers, French subscriber count, roll numbers
Output: Number of students subscribed to both
"""

if __name__ == '__main__':
    n = int(input())
    english_subscribers = set(map(int, input().split()))
    b = int(input())
    french_subscribers = set(map(int, input().split()))
    
    both_subscribers = english_subscribers.intersection(french_subscribers)
    print(len(both_subscribers))

#%% Set .difference() Operation
"""
Problem: Find students subscribed to English but not French newspaper
Description: Count student IDs in English set but not in French set
Input: English subscriber count, roll numbers, French subscriber count, roll numbers
Output: Number of students subscribed only to English
"""

if __name__ == '__main__':
    n = int(input())
    english_subscribers = set(map(int, input().split()))
    b = int(input())
    french_subscribers = set(map(int, input().split()))
    
    only_english = english_subscribers.difference(french_subscribers)
    print(len(only_english))

#%% Set .symmetric_difference() Operation
"""
Problem: Find students subscribed to only one newspaper
Description: Count students subscribed to either English or French but not both
Input: English subscriber count, roll numbers, French subscriber count, roll numbers
Output: Number of students subscribed to exactly one newspaper
"""

if __name__ == '__main__':
    n = int(input())
    english_subscribers = set(map(int, input().split()))
    b = int(input())
    french_subscribers = set(map(int, input().split()))
    
    only_one = english_subscribers.symmetric_difference(french_subscribers)
    print(len(only_one))

#%% Set Mutations
"""
Problem: Perform mutation operations on a set
Description: Apply operations like update, intersection_update, difference_update, symmetric_difference_update
Input: Set size and elements, number of operations, then operation details
Output: Sum of elements in the final set
"""

if __name__ == '__main__':
    n = int(input())
    A = set(map(int, input().split()))
    num_operations = int(input())
    
    for _ in range(num_operations):
        operation_input = input().split()
        operation = operation_input[0]
        length = int(input())
        other_set = set(map(int, input().split()))

        if operation == 'update':
            A.update(other_set)
        elif operation == 'intersection_update':
            A.intersection_update(other_set)
        elif operation == 'difference_update':
            A.difference_update(other_set)
        elif operation == 'symmetric_difference_update':
            A.symmetric_difference_update(other_set)
    
    print(sum(A))

#%% The Captain's Room
"""
Problem: Find the captain's room number
Description: Given room numbers where each family has K members with same room number,
            find the captain's unique room number (appears only once)
Input: K (group size), list of room numbers
Output: Captain's room number
"""

if __name__ == '__main__':
    K = int(input())
    room_numbers = list(map(int, input().split()))
    
    # Method 1: Using sets and math
    unique_rooms = set(room_numbers)
    captain_room = (sum(unique_rooms) * K - sum(room_numbers)) // (K - 1)
    print(captain_room)

    # Method 2: Using Counter (alternative)
    # from collections import Counter
    # room_count = Counter(room_numbers)
    # for room, count in room_count.items():
    #     if count == 1:
    #         print(room)
    #         break

#%% Check Subset
"""
Problem: Check if set A is a subset of set B
Description: For multiple test cases, determine if A is a subset of B
Input: Number of test cases, then for each test case: size and elements of A, size and elements of B
Output: True or False for each test case
"""

if __name__ == '__main__':
    t = int(input())
    
    for _ in range(t):
        n = int(input())
        A = set(map(int, input().split()))
        m = int(input())
        B = set(map(int, input().split()))
        
        print(A.issubset(B))

#%% Check Strict Superset
"""
Problem: Check if set A is a strict superset of all other sets
Description: Determine if A is a strict superset (contains all elements + more) of all given sets
Input: Set A elements, number of other sets n, then n sets
Output: True if A is strict superset of all, False otherwise
"""

if __name__ == '__main__':
    A = set(map(int, input().split()))
    n = int(input())
    
    is_strict_superset = True

    for _ in range(n):
        other_set = set(map(int, input().split()))
        
        # A must be a strict superset: A > other_set (not just A >= other_set)
        if not (A > other_set):
            is_strict_superset = False
            break
    
    print(is_strict_superset)
    
# Enter your code here. Read input from STDIN. Print output to STDOUT
if __name__ == '__main__':
    inp = set(map(int, input().split()))
    t = int(input())
    is_strict_superset = True
    for _ in range(t):
        A = set(map(int, input().split()))
        if A.issubset(inp) and len(inp) > len(A):
            pass
        else:
            is_strict_superset = False
            
    print(is_strict_superset)
            
#%% Alternative solution for Check Strict Superset using all()
"""
Alternative approach using all() function for more Pythonic code
"""

if __name__ == '__main__':
    A = set(map(int, input().split()))
    n = int(input())
    
    other_sets = [set(map(int, input().split())) for _ in range(n)]
    
    # Check if A is strict superset of all other sets
    result = all(A > other_set for other_set in other_sets)
    print(result)