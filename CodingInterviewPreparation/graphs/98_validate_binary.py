# 98. Validate Binary Search Tree
# Medium
# Topics
# premium lock icon
# Companies
# Given the root of a binary tree, determine if it is a valid binary search tree (BST).

# A valid BST is defined as follows:

# The left subtree of a node contains only nodes with keys strictly less than the node's key.
# The right subtree of a node contains only nodes with keys strictly greater than the node's key.
# Both the left and right subtrees must also be binary search trees.
 

# Example 1:


# Input: root = [2,1,3]
# Output: true
# Example 2:


# Input: root = [5,1,4,null,null,3,6]
# Output: false
# Explanation: The root node's value is 5 but its right child's value is 4.
 

# Constraints:

# The number of nodes in the tree is in the range [1, 104].
# -231 <= Node.val <= 231 - 1


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from typing import Optional

class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def validate(node: Optional[TreeNode], min_val: float, max_val: float) -> bool:
            if not node:
                return True
            
            # Check if current node's value is within the valid range
            if node.val <= min_val or node.val >= max_val:
                return False
            
            # Recursively validate left and right subtrees
            # Left subtree: values must be < node.val
            # Right subtree: values must be > node.val
            return validate(node.left, min_val, node.val) and validate(node.right, node.val, max_val)
        
        return validate(root, float('-inf'), float('inf'))