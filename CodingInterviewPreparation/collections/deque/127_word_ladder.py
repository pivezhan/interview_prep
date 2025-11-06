# 127. Word Ladder
# Solved
# Hard
# Topics
# premium lock icon
# Companies
# A transformation sequence from word beginWord to word endWord using a dictionary wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that:

# Every adjacent pair of words differs by a single letter.
# Every si for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList.
# sk == endWord
# Given two words, beginWord and endWord, and a dictionary wordList, return the number of words in the shortest transformation sequence from beginWord to endWord, or 0 if no such sequence exists.

 

# Example 1:

# Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
# Output: 5
# Explanation: One shortest transformation sequence is "hit" -> "hot" -> "dot" -> "dog" -> cog", which is 5 words long.
# Example 2:

# Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
# Output: 0
# Explanation: The endWord "cog" is not in wordList, therefore there is no valid transformation sequence.
 

# Constraints:

# 1 <= beginWord.length <= 10
# endWord.length == beginWord.length
# 1 <= wordList.length <= 5000
# wordList[i].length == beginWord.length
# beginWord, endWord, and wordList[i] consist of lowercase English letters.
# beginWord != endWord
# All the words in wordList are unique.

from collections import deque


class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordset = set(wordList)
        visited = set([beginWord])
        queue = deque([(beginWord,1)])
        if endWord not in wordset:
            return 0

        # Step 2: BFS initialization
        queue = deque([(beginWord, 1)])
        visited = set([beginWord])

        while queue:
            word, steps = queue.popleft()
            if word == endWord:
                return steps
            
            # Generate all 1-letter variations
            for i in range(len(word)):
                for char in 'abcdefghijklmnopqrstuvwxyz':
                    new_word = word[:i] + char + word[i+1:]
                    if new_word in wordset and new_word not in visited:
                        visited.add(new_word)
                        queue.append((new_word, steps + 1))
        return 0  # If no path found


        
        