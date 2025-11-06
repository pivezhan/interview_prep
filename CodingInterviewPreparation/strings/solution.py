#%% sWAP cASE
"""
Problem: Swap the letter cases of a given string.
Description: Convert uppercase letters to lowercase and vice versa.
Input: A string
Output: String with swapped cases
"""

def swap_case(s):
    result = ""
    for char in s:
        if char.isupper():
            result += char.lower()
        elif char.islower():
            result += char.upper()
        else:
            result += char
    return result

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)

#%% String Split and Join
"""
Problem: Split a string on spaces and join with hyphens
Description: Take a string, split it by spaces, and join with '-'
Input: A string with words separated by spaces
Output: String with words joined by hyphens
"""

def split_and_join(line):
    line = line.split(" ")
    line = "-".join(line)
    return line

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

#%% What's Your Name?
"""
Problem: Print a personalized greeting
Description: Given first and last name, print "Hello firstname lastname! You just delved into python."
Input: First name and last name on separate lines
Output: Formatted greeting message
"""

def print_full_name(first, last):
    print(f"Hello {first} {last}! You just delved into python.")

if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)

#%% Mutations
"""
Problem: Mutate a string at a given position
Description: Change a character at position i to character c
Input: String s, position i, and character c
Output: Modified string
"""

def mutate_string(string, position, character):
    string = string[:position] + character + string[position+1:]
    return string

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)

#%% Find a string
"""
Problem: Find the number of times a substring occurs in a string
Description: Count overlapping occurrences of substring in string
Input: String and substring
Output: Count of occurrences
"""

def count_substring(string, sub_string):
    count = 0
    for i in range(len(string) - len(sub_string) + 1):
        if string[i:i+len(sub_string)] == sub_string:
            count += 1
    return count

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    count = count_substring(string, sub_string)
    print(count)

#%% String Validators
"""
Problem: Check if string has alphanumeric, alphabetic, digits, lowercase, uppercase characters
Description: Print True/False for each validation check
Input: A string
Output: Five lines with True/False for each check
"""

if __name__ == '__main__':
    s = input()
    
    print(any(c.isalnum() for c in s))
    print(any(c.isalpha() for c in s))
    print(any(c.isdigit() for c in s))
    print(any(c.islower() for c in s))
    print(any(c.isupper() for c in s))

#%% Text Alignment
"""
Problem: Display HackerRank logo using text alignment
Description: Use string formatting methods (ljust, center, rjust) to create a logo
Input: Thickness value (odd number)
Output: Formatted HackerRank logo
"""

thickness = int(input())
c = 'H'

# Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

# Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

# Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

# Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

# Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

#%% Text Wrap
"""
Problem: Wrap a string to a specified width
Description: Use textwrap module to wrap text
Input: String and width
Output: Wrapped text
"""

import textwrap

def wrap(string, max_width):
    return textwrap.fill(string, max_width)

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)

#%% Designer Door Mat
"""
Problem: Design a door mat pattern
Description: Create a door mat with specific pattern using .|. symbols
Input: N (height) and M (width) where N is odd and M = N * 3
Output: Door mat design with WELCOME in center
"""

n, m = map(int, input().split())

# Top half
for i in range(n//2):
    pattern = ".|." * (2*i + 1)
    print(pattern.center(m, '-'))

# Middle line with WELCOME
print("WELCOME".center(m, '-'))

# Bottom half
for i in range(n//2 - 1, -1, -1):
    pattern = ".|." * (2*i + 1)
    print(pattern.center(m, '-'))

#%% String Formatting
"""
Problem: Print decimal, octal, hexadecimal, and binary values
Description: For each number from 1 to n, print its decimal, octal, hex, and binary representations
Input: Integer n
Output: Formatted table with 4 columns (decimal, octal, hex, binary)
"""

def print_formatted(number):
    width = len(bin(number)[2:])
    
    for i in range(1, number + 1):
        decimal = str(i)
        octal = oct(i)[2:]
        hexadecimal = hex(i)[2:].upper()
        binary = bin(i)[2:]
        
        print(f"{decimal.rjust(width)} {octal.rjust(width)} {hexadecimal.rjust(width)} {binary.rjust(width)}")

if __name__ == '__main__':
    n = int(input())
    print_formatted(n)

def decconvert(number, base):
    if number == 0:
        return '0'
    
    result = []
    while number > 0:
        result.append(str(number % base))
        number //= base  # integer division
    return ''.join(result[::-1])  # reverse to correct order


def print_formatted(number):
    # Width = binary length of 'number' (for alignment)
    width = len(decconvert(number, 2))
    
    for num in range(1, number + 1):
        decimal = decconvert(num, 10)
        octal   = decconvert(num, 8)
        hexa    = decconvert(num, 16).upper()
        binary  = decconvert(num, 2)
        
        # Right-align with width
        print(decimal.rjust(width),
              octal.rjust(width),
              hexa.rjust(width),
              binary.rjust(width))

if __name__ == '__main__':
    n = int(input())
    print_formatted(n)

#%% Alphabet Rangoli
"""
Problem: Print an alphabet rangoli pattern
Description: Create a rangoli pattern using lowercase letters
Input: Size n (determines which letters to use, e.g., n=3 uses a,b,c)
Output: Rangoli pattern
"""

def print_rangoli(size):
    import string
    alpha = string.ascii_lowercase
    
    lines = []
    for i in range(size):
        left = '-'.join(alpha[size-1:i:-1])
        middle = alpha[i]
        right = '-'.join(alpha[i+1:size])
        
        line = left + '-' + middle
        if right:
            line = line + '-' + right
        
        lines.append(line)
    
    width = len(lines[-1])
    
    # Print top half including middle
    for line in lines:
        print(line.center(width, '-'))
    
    # Print bottom half (reverse, excluding middle)
    for line in reversed(lines[:-1]):
        print(line.center(width, '-'))

if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)

def print_rangoli(size):
    import string
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    # alphabet = string.ascii_lowercase
    
    lines = []
    max_len = 0
    for i in range(size):
        if size > 1:
            left = '-'.join(alphabet[size-1:i:-1])
            middle = alphabet[i]
            right = '-'.join(alphabet[i+1:size])
            
            line = left + '-' + middle + '-' + right
            
            lines.append(line)
            if len(line) > max_len:
                max_len = len(line)
        else:
            lines = alphabet[i]
            
    

    width = max_len

    # Print bottom half (reverse, excluding middle)
    for line in reversed(lines[1:]):
        print(line.center(width,'-'))
    
    # Print top half including middle
    for line in lines:
        print(line.center(width,'-'))


if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)
    
#%% Capitalize!
"""
Problem: Capitalize the first letter of each word
Description: Capitalize first letter of each word in a full name, including after spaces
Input: Full name string
Output: Capitalized full name
"""

def solve(s):
    result = ""
    capitalize_next = True
    
    for char in s:
        if capitalize_next and char.isalpha():
            result += char.upper()
            capitalize_next = False
        else:
            result += char
            if char == ' ':
                capitalize_next = True
    
    return result

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s = input()
    result = solve(s)
    fptr.write(result + '\n')
    fptr.close()

# Alternative simpler solution for local testing:
def solve_simple(s):
    return ' '.join(word.capitalize() for word in s.split(' '))

#%% The Minion Game
"""
Problem: Play the minion game between Stuart (consonants) and Kevin (vowels)
Description: Count substrings starting with consonants vs vowels
Input: String s
Output: Winner name and score, or "Draw"
"""

def minion_game(string):
    vowels = 'AEIOU'
    stuart_score = 0
    kevin_score = 0
    length = len(string)
    
    for i in range(length):
        if string[i] in vowels:
            kevin_score += (length - i)
        else:
            stuart_score += (length - i)
    
    if stuart_score > kevin_score:
        print(f"Stuart {stuart_score}")
    elif kevin_score > stuart_score:
        print(f"Kevin {kevin_score}")
    else:
        print("Draw")

if __name__ == '__main__':
    s = input()
    minion_game(s)

#%% Merge the Tools!
"""
Problem: Split string into chunks and remove consecutive duplicates
Description: Divide string into substrings of length k, remove duplicate consecutive chars
Input: String s and integer k
Output: Modified substrings on separate lines
"""

def merge_the_tools(string, k):
    for i in range(0, len(string), k):
        substring = string[i:i+k]
        
        # Remove consecutive duplicates
        result = ""
        seen = set()
        for char in substring:
            if char not in seen:
                result += char
                seen.add(char)
        
        print(result)

if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)


# 125. Valid Palindrome
# Easy
# Topics
# premium lock icon
# Companies
# A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

# Given a string s, return true if it is a palindrome, or false otherwise.

 

# Example 1:

# Input: s = "A man, a plan, a canal: Panama"
# Output: true
# Explanation: "amanaplanacanalpanama" is a palindrome.
# Example 2:

# Input: s = "race a car"
# Output: false
# Explanation: "raceacar" is not a palindrome.
# Example 3:

# Input: s = " "
# Output: true
# Explanation: s is an empty string "" after removing non-alphanumeric characters.
# Since an empty string reads the same forward and backward, it is a palindrome.
 

# Constraints:

# 1 <= s.length <= 2 * 105
# s consists only of printable ASCII characters.

class Solution:
    def isPalindrome(self, s: str) -> bool:
        # Clean string: lowercase + alphanumeric only
        cleaned = ''.join(filter(str.isalnum, s.lower()))
        # Compare cleaned string to its reverse
        return cleaned == cleaned[::-1]

#%%
# 20. Valid Parentheses
# Easy
# Topics
# premium lock icon
# Companies
# Given a string s containing just the characters '(', ')', '{', '}', '[' and
# ']', determine if the input string is valid.
# An input string is valid if:
# Open brackets must be closed by the same type of brackets.
# Open brackets must be closed in the correct order.


from collections import deque
from typing import Deque

class Solution:
    def isValid(self, s: str) -> bool:
        """
        Return True if s contains only valid parentheses pairs, False otherwise.
        """
        # Mapping of closing -> opening brackets
        pairs: dict[str, str] = {')': '(', '}': '{', ']': '['}
        stack: Deque[str] = deque()

        for char in s:
            if char in pairs.values():          # opening bracket
                stack.append(char)
            elif char in pairs:                 # closing bracket
                if not stack or stack[-1] != pairs[char]:
                    return False
                stack.pop()
            else:
                # ignore any non-bracket characters (problem guarantees only (){}[])
                pass

        return len(stack) == 0
    
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        mapping = {')': '(', ']': '[', '}': '{'}
        
        for char in s:
            if char in mapping.values():  # Opening bracket
                stack.append(char)
            elif char in mapping.keys():  # Closing bracket
                if not stack or stack.pop() != mapping[char]:
                    return False
        return not stack  # Stack must be empty for valid parentheses