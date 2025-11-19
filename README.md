# 🎯 Interview Preparation Repository

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C++-17-00599C.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive repository tracking my structured preparation for technical interviews, including coding challenges, GPU/CUDA systems programming, machine learning, and certification materials.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Coding Interview Preparation](#1-coding-interview-preparation)
  - [Data Structures & Algorithms](#data-structures--algorithms)
  - [Programming Languages](#programming-languages)
  - [GPU & Systems Programming](#gpu--systems-programming)
- [Certificates & Courses](#2-certificates--courses)
- [Progress Tracking](#3-progress-tracking)
- [How to Use This Repository](#how-to-use-this-repository)
- [Resources](#resources)

---

## 🎓 Overview

This repository serves as a centralized hub for my technical interview preparation journey, covering:

- ✅ **Algorithm & Data Structure Mastery** - LeetCode-style problems organized by topic
- ✅ **System-Level Programming** - CUDA, GPU optimization, parallel computing
- ✅ **Machine Learning & Deep Learning** - PyTorch, NumPy implementations
- ✅ **Industry Certifications** - Coursera, Meta, Algorithmic Toolbox courses
- ✅ **NVIDIA Interview Preparation** - Specialized GPU and systems track

**🎯 Goal**: Systematic preparation for senior software engineering and ML engineering roles at top tech companies.

---

## 📁 Repository Structure

```
interview_prep/
├── CodingInterviewPreparation/     # Core DSA and programming practice
│   ├── arrays/                     # Array manipulation problems
│   ├── binary_tree/                # Binary tree algorithms
│   ├── collections/                # Python collections module
│   ├── cpp/                        # C++ solutions
│   ├── datatypes/                  # Data type problems
│   ├── dates/                      # Date/time handling
│   ├── dynamicprogramming/         # DP patterns and solutions
│   ├── GPU/                        # CUDA and GPU programming
│   ├── graphs/                     # Graph algorithms
│   ├── itertools/                  # Python itertools
│   ├── linkedlists/                # Linked list problems
│   ├── math/                       # Mathematical problems
│   ├── numpy/                      # NumPy-based solutions
│   ├── pytorch/                    # PyTorch implementations
│   ├── search/                     # Search algorithms
│   ├── sets/                       # Set operations
│   ├── sort/                       # Sorting algorithms
│   ├── strings/                    # String manipulation
│   ├── nvidia_schedule.md          # NVIDIA interview timeline
│   └── README.md                   # Detailed coding prep guide
│
└── certificates/                   # Course materials and certificates
    ├── Algorithmic-Toolbox_San-Diego/
    ├── Coding-Interview-Preparation_META/
    └── certificates/
```

---

## 1. 💻 Coding Interview Preparation

### Data Structures & Algorithms

#### 📊 Arrays & Sequences
**Location**: [`CodingInterviewPreparation/arrays`](./CodingInterviewPreparation/arrays)

| Problem | Difficulty | Key Concepts |
|---------|-----------|--------------|
| [121. Best Time to Buy and Sell Stock](./CodingInterviewPreparation/arrays/121_Best_Time_to_Buy.py) | Easy | Kadane's Algorithm, Greedy |
| [53. Maximum Subarray](./CodingInterviewPreparation/arrays/53_Maximum_Subarray.py) | Medium | Dynamic Programming, Kadane's |
| [300. Longest Increasing Subsequence](./CodingInterviewPreparation/arrays/300_Longest_Increasing_Subsequence.py) | Medium | DP, Binary Search |

**Key Patterns**:
- Two Pointers
- Sliding Window
- Kadane's Algorithm
- Prefix Sums

---

#### 🌳 Trees & Graphs
**Locations**: 
- [`CodingInterviewPreparation/binary_tree`](./CodingInterviewPreparation/binary_tree)
- [`CodingInterviewPreparation/graphs`](./CodingInterviewPreparation/graphs)

| Problem | Type | Key Concepts |
|---------|------|--------------|
| [236. Lowest Common Ancestor](./CodingInterviewPreparation/binary_tree/236_Lowest_Common_Ancestor.py) | Binary Tree | DFS, Recursion |
| [Serialize/Deserialize Binary Tree](./CodingInterviewPreparation/binary_tree/serialize_deserialize.py) | Binary Tree | BFS, String Encoding |
| [133. Clone Graph](./CodingInterviewPreparation/graphs/133_Clone_Graph.py) | Graph | DFS/BFS, HashMap |
| [98. Validate Binary Search Tree](./CodingInterviewPreparation/graphs/98_validate_binary.py) | BST | In-order Traversal |
| [Find Number of Islands](./CodingInterviewPreparation/graphs/fnd_islands.py) | Graph | DFS/BFS, Union-Find |

**Key Patterns**:
- DFS (Depth-First Search)
- BFS (Breadth-First Search)
- Topological Sort
- Union-Find (Disjoint Set)
- Shortest Path (Dijkstra, Bellman-Ford)

---

#### 🔗 Linked Lists
**Location**: [`CodingInterviewPreparation/linkedlists`](./CodingInterviewPreparation/linkedlists)

| Problem | Difficulty | Key Concepts |
|---------|-----------|--------------|
| [2. Add Two Numbers](./CodingInterviewPreparation/linkedlists/2_add_two_numbers.py) | Medium | Math, Carry Handling |
| [Merge Two Sorted Lists](./CodingInterviewPreparation/linkedlists/merge_two_sorted_list.py) | Easy | Two Pointers, Merge |

**Key Patterns**:
- Fast & Slow Pointers (Floyd's Cycle Detection)
- Reversal
- Merge Techniques

---

#### 🎯 Dynamic Programming
**Location**: [`CodingInterviewPreparation/dynamicprogramming`](./CodingInterviewPreparation/dynamicprogramming)

| Problem | Pattern | Complexity |
|---------|---------|------------|
| [322. Coin Change](./CodingInterviewPreparation/dynamicprogramming/322_Coin_Change.py) | Unbounded Knapsack | O(n × amount) |
| [0/1 Knapsack](./CodingInterviewPreparation/dynamicprogramming/knapsack.py) | Classic DP | O(n × W) |

**Common DP Patterns**:
1. 0/1 Knapsack
2. Unbounded Knapsack
3. LCS (Longest Common Subsequence)
4. LIS (Longest Increasing Subsequence)
5. Edit Distance
6. Matrix Chain Multiplication

---

#### 🔍 Search & Sort Algorithms
**Locations**: 
- [`CodingInterviewPreparation/search`](./CodingInterviewPreparation/search)
- [`CodingInterviewPreparation/sort`](./CodingInterviewPreparation/sort)

**Search Algorithms**:
- [Binary Search](./CodingInterviewPreparation/search/binary.py)
- [Linear Search](./CodingInterviewPreparation/search/linear.py)

**Sorting Algorithms**:
- [Quick Sort](./CodingInterviewPreparation/sort/quicksort.py) | [C++ Version](./CodingInterviewPreparation/sort/quicksort.cpp)
- [Merge Sort](./CodingInterviewPreparation/sort/marge.py)
- [Insertion Sort](./CodingInterviewPreparation/sort/insert.py)
- [Selection Sort](./CodingInterviewPreparation/sort/selection.py)

**Time Complexities**:
| Algorithm | Best | Average | Worst | Space |
|-----------|------|---------|-------|-------|
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) |
| Insertion | O(n) | O(n²) | O(n²) | O(1) |

---

### Programming Languages

#### 🐍 Python Standard Library
**Locations**: 
- [`collections`](./CodingInterviewPreparation/collections) - Deque, LRU Cache, OrderedDict
- [`itertools`](./CodingInterviewPreparation/itertools) - Combinatorics, permutations
- [`sets`](./CodingInterviewPreparation/sets) - Set operations
- [`datatypes`](./CodingInterviewPreparation/datatypes) - Type handling
- [`dates`](./CodingInterviewPreparation/dates) - DateTime manipulation
- [`strings`](./CodingInterviewPreparation/strings) - String algorithms
- [`math`](./CodingInterviewPreparation/math) - Mathematical operations

**Key Modules Covered**:
```python
# Collections
from collections import deque, OrderedDict, Counter, defaultdict
from functools import lru_cache

# Itertools
from itertools import combinations, permutations, product, chain

# Example: LRU Cache implementation
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

#### 🔧 C++
**Location**: [`CodingInterviewPreparation/cpp`](./CodingInterviewPreparation/cpp)

Focus areas:
- STL (Standard Template Library)
- Pointers and Memory Management
- Templates and Generic Programming
- Move Semantics

---

### GPU & Systems Programming

#### 🚀 CUDA & GPU Programming
**Location**: [`CodingInterviewPreparation/GPU`](./CodingInterviewPreparation/GPU)

| Topic | File | Concepts |
|-------|------|----------|
| Matrix Multiplication | [matrix_mult.cpp](./CodingInterviewPreparation/GPU/matrix_mult.cpp) | Basic CUDA kernels |
| Cache Optimization | [matrix_mult_cache.cpp](./CodingInterviewPreparation/GPU/matrix_mult_cache.cpp) | Shared memory, tiling |
| Race Conditions | [matrix_mult_race_cond.cpp](./CodingInterviewPreparation/GPU/matrix_mult_race_cond.cpp) | Synchronization, atomics |
| Warp Reduction | [warpsreduction.pdf](./CodingInterviewPreparation/GPU/warpsreduction.pdf) | Parallel reduction patterns |

**Key CUDA Concepts**:
- Thread hierarchy (Grid → Block → Warp → Thread)
- Memory hierarchy (Global, Shared, Local, Registers)
- Occupancy optimization
- Bank conflicts and coalescing
- Warp divergence
- Async operations and streams

**NVIDIA Interview Focus** 📅:
See detailed schedule: [`nvidia_schedule.md`](./CodingInterviewPreparation/nvidia_schedule.md)

---

#### 🤖 Machine Learning & Deep Learning
**Locations**:
- [`numpy`](./CodingInterviewPreparation/numpy) - NumPy implementations
- [`pytorch`](./CodingInterviewPreparation/pytorch) - PyTorch tutorials

**PyTorch Coverage**:
- [Tensor Operations](./CodingInterviewPreparation/pytorch/tensor_tutorial.py)
- [Autograd Tutorial](./CodingInterviewPreparation/pytorch/autograd_tutorial.py)
- [Data Loading](./CodingInterviewPreparation/pytorch/data_tutorial.py)
- [Neural Networks](./CodingInterviewPreparation/pytorch/neural_networks_tutorial.py)
- [CIFAR-10 Classification](./CodingInterviewPreparation/pytorch/cifar10_tutorial.py)
- [Multi-Layer Perceptron](./CodingInterviewPreparation/pytorch/mlp.py)
- [Data Parallelism](./CodingInterviewPreparation/pytorch/copy_of_data_parallel_tutorial.py)

---

## 2. 🎓 Certificates & Courses

### Completed Certifications

#### 📘 Algorithmic Toolbox (UC San Diego)
**Location**: [`certificates/Algorithmic-Toolbox_San-Diego`](./certificates/Algorithmic-Toolbox_San-Diego)

**Course Structure**:
- Week 1: Programming Challenges
- Week 2: Algorithmic Warm-up
- Week 3: Greedy Algorithms
- Week 4: Divide and Conquer
- Week 5: Dynamic Programming 1
- Week 6: Dynamic Programming 2

**Certificate**: [View PDF](./certificates/certificates/Algorithmic%20Toolbox.pdf)

---

#### 🏢 Coding Interview Preparation (META)
**Location**: [`certificates/Coding-Interview-Preparation_META`](./certificates/Coding-Interview-Preparation_META)

**Course Structure**:
- **Week 1**: Introduction to Coding Interview
- **Week 2**: Data Structures Deep Dive
- **Week 3**: Algorithms and Problem Solving
- **Week 4**: Mock Interviews and Best Practices

**Certificate**: [View PDF](./certificates/certificates/Coding%20Interview%20Preparation.pdf)

---

#### ☕ Java Programming: Solving Problems with Software
**Certificate**: [View PDF](./certificates/certificates/Java%20Programming:%20Solving%20Problems%20with.pdf)

**Topics Covered**:
- Object-Oriented Programming
- Data Structures in Java
- Algorithm Implementation
- Software Design Patterns

---

#### 💼 Software Developer Career Guide
**Certificate**: [View PDF](./certificates/certificates/Software%20Developer%20Career%20Guide%20and%20Interview%20Preparation.pdf)

**Focus Areas**:
- Resume building
- Technical interview strategies
- System design fundamentals
- Behavioral interview preparation

---

## 3. 📊 Progress Tracking

### 📈 Statistics (Updated Regularly)

```
┌─────────────────────────────────────────────┐
│  PROGRESS DASHBOARD                         │
├─────────────────────────────────────────────┤
│  Total Problems Solved:         150+        │
│  LeetCode Profile:              [Link]      │
│  Easy:                          45          │
│  Medium:                        82          │
│  Hard:                          23          │
│                                             │
│  Topics Mastered:                           │
│  ✅ Arrays & Strings                        │
│  ✅ Hash Tables                             │
│  ✅ Linked Lists                            │
│  ✅ Trees & Graphs                          │
│  ✅ Dynamic Programming                     │
│  ✅ Binary Search                           │
│  🔄 System Design (In Progress)             │
│  🔄 CUDA Optimization (In Progress)         │
│                                             │
│  Certifications Completed:      4           │
│  Mock Interviews:               12          │
│                                             │
│  Last Updated:   November 19, 2025          │
└─────────────────────────────────────────────┘
```

### 🎯 Current Focus Areas
- [ ] Advanced Dynamic Programming patterns
- [ ] System Design case studies
- [ ] CUDA kernel optimization
- [ ] LLM fine-tuning and deployment
- [ ] Distributed systems concepts

### 📅 Weekly Goals
**This Week**:
- [ ] Solve 15 medium-level problems
- [ ] Complete 2 mock interviews
- [ ] Review GPU optimization techniques
- [ ] Study system design patterns

---

## 🚀 How to Use This Repository

### For Interview Preparation
1. **Start with Fundamentals**: Begin with arrays, strings, and basic data structures
2. **Follow the Topics**: Work through each topic folder systematically
3. **Practice Patterns**: Focus on recognizing problem patterns rather than memorizing solutions
4. **Time Yourself**: Practice with time constraints (45-60 minutes per problem)
5. **Review Regularly**: Revisit solved problems weekly

### For GPU/CUDA Preparation
1. Read [notes.md](./CodingInterviewPreparation/GPU/notes.md) for concept overview
2. Study example implementations in order:
   - Basic matrix multiplication
   - Cache-optimized version
   - Race condition handling
3. Review [warpsreduction.pdf](./CodingInterviewPreparation/GPU/warpsreduction.pdf) for advanced patterns

### Running the Code
```bash
# Python solutions
cd CodingInterviewPreparation/arrays
python3 53_Maximum_Subarray.py

# C++ solutions
cd CodingInterviewPreparation/cpp
g++ -std=c++17 solutions.cpp -o solution
./solution

# CUDA examples
cd CodingInterviewPreparation/GPU
nvcc matrix_mult.cpp -o matrix_mult
./matrix_mult

# PyTorch tutorials
cd CodingInterviewPreparation/pytorch
python3 tensor_tutorial.py
```

---

## 📚 Resources

### Recommended Books
- **Cracking the Coding Interview** - Gayle Laakmann McDowell
- **Elements of Programming Interviews** - Adnan Aziz, Tsung-Hsien Lee, Amit Prakash
- **Algorithm Design Manual** - Steven Skiena
- **Programming Massively Parallel Processors** - David Kirk, Wen-mei Hwu

### Online Platforms
- [LeetCode](https://leetcode.com) - Practice problems
- [HackerRank](https://hackerrank.com) - Skill certification
- [Codeforces](https://codeforces.com) - Competitive programming
- [NVIDIA Developer](https://developer.nvidia.com) - CUDA documentation

### Video Resources
- [NeetCode](https://neetcode.io) - Problem patterns and solutions
- [Back To Back SWE](https://backtobackswe.com) - Algorithm explanations
- [TechLead](https://www.youtube.com/c/TechLead) - Interview tips

---

## 🤝 Contributing

This is a personal learning repository, but suggestions are welcome! Feel free to:
- Open an issue for corrections
- Suggest additional resources
- Share alternative solutions

---

## 📄 License

MIT License - Feel free to use this repository structure for your own interview prep!

---

## 📞 Contact

**Mohammad** | [GitHub Profile](https://github.com/yourusername)

*Last Updated: November 19, 2025*

---

<div align="center">
  
### ⭐ Star this repo if you find it helpful!

**Good luck with your interviews! 🚀**

</div>
