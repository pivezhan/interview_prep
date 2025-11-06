# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        # Helper function to perform DFS (Depth-First Search) in pre-order
        def dfs(node):
            # Base case: if node is None, represent it as "null"
            if not node:
                return "null,"
            
            # Pre-order traversal: Root → Left → Right
            # 1. Convert current node value to string
            # 2. Recursively serialize left subtree
            # 3. Recursively serialize right subtree
            # 4. Join with commas
            return str(node.val) + "," + dfs(node.left) + dfs(node.right)
        
        # Call DFS from root and remove the trailing comma at the end
        return dfs(root)[:-1]

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        # Convert the string into a list of values and create an iterator
        # This allows us to consume values one by one using next()
        vals = iter(data.split(","))

        # Helper function to reconstruct tree using pre-order logic
        def dfs():
            # Get the next value from the iterator
            val = next(vals)
            
            # If value is "null", return None (no node)
            if val == "null":
                return None
            
            # Otherwise, create a new TreeNode with this value
            node = TreeNode(int(val))
            
            # Recursively build left subtree (next values in pre-order)
            node.left = dfs()
            
            # Recursively build right subtree
            node.right = dfs()
            
            # Return the constructed node
            return node
        
        # Start reconstruction from the root
        return dfs()


# ========================================
# Example Usage with Comments
# ========================================

"""
# Build the tree:
#       1
#      / \
#     2   3
#        / \
#       4   5

root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.right.left = TreeNode(4)
root.right.right = TreeNode(5)

# Create codec object
codec = Codec()

# Serialize: tree → string
serialized = codec.serialize(root)
print(serialized)
# Output: "1,2,null,null,3,4,null,null,5,null,null"

# Deserialize: string → tree
deserialized_root = codec.deserialize(serialized)

# Verify: serialize again and compare
print(codec.serialize(deserialized_root) == serialized)  # True
"""