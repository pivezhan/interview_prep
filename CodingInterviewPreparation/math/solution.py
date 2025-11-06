#%% Polar Coordinates
"""
Problem: Convert complex number to polar coordinates
Description: Given a complex number, output its polar coordinates (r, φ)
            where r is the modulus and φ is the phase angle in radians
Input: Complex number in the form a+bj or a-bj
Output: Two lines - modulus r and phase angle φ
"""

import cmath

if __name__ == '__main__':
    z = complex(input())
    
    # Convert to polar coordinates
    r = abs(z)  # Modulus
    phi = cmath.phase(z)  # Phase angle in radians
    
    print(r)
    print(phi)

#%% Alternative solution for Polar Coordinates (using math)
"""
Using math module instead of cmath
"""

import math

if __name__ == '__main__':
    z = complex(input())
    
    # Extract real and imaginary parts
    real = z.real
    imag = z.imag
    
    # Calculate modulus and phase
    r = math.sqrt(real**2 + imag**2)
    phi = math.atan2(imag, real)
    
    print(r)
    print(phi)

#%% Find Angle MBC
"""
Problem: Find angle MBC in a right triangle
Description: Given AB and BC sides of a right triangle (right angle at B),
            find angle MBC where M is the midpoint of hypotenuse AC
Input: Two integers AB and BC (in centimeters)
Output: Angle MBC in degrees with ° symbol
"""

import math

if __name__ == '__main__':
    AB = int(input())
    BC = int(input())
    
    # In a right triangle with right angle at B
    # M is midpoint of hypotenuse AC
    # Angle MBC can be calculated using: tan(angle) = AB/BC
    
    # Calculate angle in radians
    angle_radians = math.atan2(AB, BC)
    
    # Convert to degrees
    angle_degrees = math.degrees(angle_radians)
    
    # Round to nearest integer
    result = round(angle_degrees)
    
    print(f"{result}°")

#%% Alternative solution for Find Angle MBC
"""
More detailed geometric approach
"""

import math

if __name__ == '__main__':
    AB = int(input())
    BC = int(input())
    
    # Using atan2 for better handling of edge cases
    angle = math.atan2(AB, BC)
    
    # Convert radians to degrees and round
    angle_deg = round(math.degrees(angle))
    
    # Print with degree symbol
    print(str(angle_deg) + chr(176))

#%% Triangle Quest 2
"""
Problem: Print palindromic triangle pattern using one line
Description: Print a pattern of palindromic numbers without string manipulation
Input: Integer n (number of rows)
Output: Palindromic triangle pattern
Pattern for n=5:
1
121
12321
1234321
123454321
"""

# Mathematical approach - each row i prints: (10^i - 1)^2 // 9^2
for i in range(1, int(input()) + 1):
    print(((10**i - 1) // 9) ** 2)

#%% Explanation for Triangle Quest 2
"""
Mathematical Pattern Explanation:

Row 1: 1 = 1^2
Row 2: 121 = 11^2
Row 3: 12321 = 111^2
Row 4: 1234321 = 1111^2
Row 5: 123454321 = 11111^2

Pattern: Row i = (10^i - 1)^2 / 81
- 10^1 - 1 = 9; 9/9 = 1; 1^2 = 1
- 10^2 - 1 = 99; 99/9 = 11; 11^2 = 121
- 10^3 - 1 = 999; 999/9 = 111; 111^2 = 12321

General formula: ((10^i - 1) // 9)^2
"""

#%% Mod Divmod
"""
Problem: Demonstrate divmod function
Description: Given two integers a and b, print:
            - Integer division (a // b)
            - Modulo (a % b)
            - Result of divmod(a, b)
Input: Two integers a and b on separate lines
Output: Three lines with division, modulo, and divmod result
"""

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    
    # Integer division
    print(a // b)
    
    # Modulo
    print(a % b)
    
    # divmod returns tuple (quotient, remainder)
    print(divmod(a, b))

#%% Power - Mod Power
"""
Problem: Calculate power and modular power
Description: Given integers a, b, m compute:
            - a^b
            - (a^b) % m
Input: Three integers a, b, m on separate lines
Output: Two lines with power and modular power
"""

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    m = int(input())
    
    # Calculate power
    print(pow(a, b))
    
    # Calculate modular power (more efficient than (a**b) % m)
    print(pow(a, b, m))

#%% Integers Come In All Sizes
"""
Problem: Demonstrate Python's arbitrary precision integers
Description: Given four integers a, b, c, d, compute a^b + c^d
Input: Four integers a, b, c, d on separate lines
Output: Result of a^b + c^d
"""

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    c = int(input())
    d = int(input())
    
    # Python handles arbitrarily large integers automatically
    result = pow(a, b) + pow(c, d)
    print(result)

#%% Alternative for Integers Come In All Sizes
"""
Using ** operator instead of pow
"""

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    c = int(input())
    d = int(input())
    
    result = a**b + c**d
    print(result)

#%% Triangle Quest
"""
Problem: Print number triangle pattern in one line
Description: Print a specific triangle pattern without string manipulation
Input: Integer n (number of rows)
Output: Triangle pattern
Pattern for n=5:
1
22
333
4444
55555
"""

for i in range(1, int(input()) + 1):
    print(i * ((10**i - 1) // 9))

#%% Explanation for Triangle Quest
"""
Pattern Explanation:

Row 1: 1 = 1 × 1
Row 2: 22 = 2 × 11
Row 3: 333 = 3 × 111
Row 4: 4444 = 4 × 1111
Row 5: 55555 = 5 × 11111

Repunits (numbers with all 1s):
- 1 = (10^1 - 1) / 9
- 11 = (10^2 - 1) / 9
- 111 = (10^3 - 1) / 9
- 1111 = (10^4 - 1) / 9

Formula for row i: i × ((10^i - 1) // 9)
"""

#%% Mathematical Concepts Cheat Sheet
"""
PYTHON MATH MODULE & CONCEPTS:

=== Basic Arithmetic ===
+, -, *, /          # Addition, subtraction, multiplication, division
//                  # Floor division (integer division)
%                   # Modulo (remainder)
**                  # Exponentiation
divmod(a, b)        # Returns (a//b, a%b)

=== Power Operations ===
pow(a, b)           # a^b
pow(a, b, m)        # (a^b) % m - efficient modular exponentiation
a**b                # Same as pow(a, b)

=== Math Module Functions ===
import math

# Trigonometry
math.sin(x)         # Sine (x in radians)
math.cos(x)         # Cosine
math.tan(x)         # Tangent
math.asin(x)        # Arcsine
math.acos(x)        # Arccosine
math.atan(x)        # Arctangent
math.atan2(y, x)    # atan(y/x) with correct quadrant

# Conversions
math.degrees(x)     # Radians to degrees
math.radians(x)     # Degrees to radians

# Rounding
math.ceil(x)        # Ceiling (round up)
math.floor(x)       # Floor (round down)
round(x, n)         # Round to n decimal places

# Logarithms
math.log(x)         # Natural log (ln)
math.log10(x)       # Base-10 log
math.log(x, base)   # Log with custom base

# Square root
math.sqrt(x)        # Square root
x**0.5              # Alternative

# Constants
math.pi             # π ≈ 3.14159
math.e              # e ≈ 2.71828

# Other
math.factorial(n)   # n!
math.gcd(a, b)      # Greatest common divisor
math.lcm(a, b)      # Least common multiple (Python 3.9+)

=== Complex Numbers (cmath) ===
import cmath

z = complex(a, b)   # Create complex number a+bj
z.real              # Real part
z.imag              # Imaginary part
abs(z)              # Modulus |z| = √(a²+b²)
cmath.phase(z)      # Phase angle φ = atan2(b, a)
cmath.polar(z)      # Returns (r, φ)
cmath.rect(r, phi)  # Polar to rectangular

=== Number Theory ===
# Modular Arithmetic
(a + b) % m = ((a % m) + (b % m)) % m
(a * b) % m = ((a % m) * (b % m)) % m
(a^b) % m = pow(a, b, m)  # Efficient for large b

# GCD and LCM
import math
gcd = math.gcd(a, b)
lcm = (a * b) // gcd

# Prime checking (simple)
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

=== Useful Patterns ===

# Repunit (all 1s): (10^n - 1) / 9
# Examples: 1, 11, 111, 1111
repunit = (10**n - 1) // 9

# Palindromic numbers via squaring
# 11^2 = 121, 111^2 = 12321, 1111^2 = 1234321
palindrome = ((10**n - 1) // 9) ** 2

# Sum of first n natural numbers: n(n+1)/2
sum_n = n * (n + 1) // 2

# Sum of squares: n(n+1)(2n+1)/6
sum_squares = n * (n + 1) * (2 * n + 1) // 6

# Geometric series sum: (r^n - 1) / (r - 1)
geo_sum = (r**n - 1) // (r - 1)

=== Precision & Large Numbers ===
# Python integers have arbitrary precision
big_num = 10**1000  # No overflow!

# Floating point precision
from decimal import Decimal
x = Decimal('0.1')  # Exact decimal arithmetic

# Fractions
from fractions import Fraction
f = Fraction(1, 3)  # Exact rational numbers
"""

#%% Common Math Problem Patterns
"""
Demonstrations of common mathematical patterns in programming
"""

def math_patterns():
    import math
    
    print("=== Pattern 1: Digital Root ===")
    # Sum digits repeatedly until single digit
    def digital_root(n):
        return 1 + (n - 1) % 9 if n > 0 else 0
    print(f"Digital root of 38: {digital_root(38)}")  # 3+8=11, 1+1=2
    
    print("\n=== Pattern 2: Palindrome Check ===")
    def is_palindrome_number(n):
        return str(n) == str(n)[::-1]
    print(f"Is 12321 palindrome? {is_palindrome_number(12321)}")
    
    print("\n=== Pattern 3: Sum of Divisors ===")
    def sum_of_divisors(n):
        result = 0
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                result += i
                if i != n // i and i != 1:
                    result += n // i
        return result
    print(f"Sum of divisors of 12: {sum_of_divisors(12)}")
    
    print("\n=== Pattern 4: Fibonacci ===")
    def fibonacci(n):
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    print(f"10th Fibonacci: {fibonacci(10)}")
    
    print("\n=== Pattern 5: Prime Factorization ===")
    def prime_factors(n):
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors
    print(f"Prime factors of 60: {prime_factors(60)}")
    
    print("\n=== Pattern 6: Fast Power (Exponentiation by Squaring) ===")
    def fast_power(base, exp, mod=None):
        result = 1
        base = base % mod if mod else base
        while exp > 0:
            if exp % 2 == 1:
                result = (result * base) % mod if mod else result * base
            exp //= 2
            base = (base * base) % mod if mod else base * base
        return result
    print(f"2^10 mod 1000: {fast_power(2, 10, 1000)}")

# Uncomment to run
# math_patterns()

#%% Trigonometry Quick Reference
"""
Common trigonometric identities and conversions
"""

import math

def trig_reference():
    # Angle conversions
    angle_deg = 45
    angle_rad = math.radians(angle_deg)
    print(f"{angle_deg}° = {angle_rad} radians")
    
    # Basic trig functions
    print(f"\nFor 45°:")
    print(f"sin(45°) = {math.sin(angle_rad):.4f}")
    print(f"cos(45°) = {math.cos(angle_rad):.4f}")
    print(f"tan(45°) = {math.tan(angle_rad):.4f}")
    
    # Inverse trig
    x = 0.5
    print(f"\nInverse functions for {x}:")
    print(f"arcsin({x}) = {math.degrees(math.asin(x)):.2f}°")
    print(f"arccos({x}) = {math.degrees(math.acos(x)):.2f}°")
    print(f"arctan({x}) = {math.degrees(math.atan(x)):.2f}°")
    
    # Pythagorean theorem
    a, b = 3, 4
    c = math.sqrt(a**2 + b**2)
    print(f"\nPythagorean: {a}² + {b}² = {c}²")
    
    # Distance between points
    x1, y1 = 0, 0
    x2, y2 = 3, 4
    dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    print(f"Distance from ({x1},{y1}) to ({x2},{y2}) = {dist}")

# Uncomment to run
# trig_reference()