from collections import deque
from typing import List

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        """
        Returns the maximum value in each sliding window of size `k` in the array `nums`.
        
        Example:
            nums = [1,3,-1,-3,5,3,6,7], k = 3
            Output: [3, 3, 5, 5, 6, 7]
        
        Time Complexity:  O(n)  -> Each element is added and removed at most once
        Space Complexity: O(k)  -> Deque stores at most k indices
        """
        
        # Edge case: if array is empty, return empty list
        if not nums or k == 0:
            return []
        
        # This deque will store indices of elements in the current window
        # It maintains indices in decreasing order of their values:
        #   dq[0]   -> index of the largest element in current window
        #   dq[-1]  -> index of the smallest useful element
        dq = deque()
        
        # Final result list to store maximum of each window
        result = []
        
        # Iterate through each element in the array
        for i in range(len(nums)):
            current_value = nums[i]
            
            # === STEP 1: Remove indices that are outside the current window ===
            # The window is from index (i - k + 1) to i
            # So any index <= i - k is no longer in the current window
            if dq and dq[0] <= i - k:
                dq.popleft()  # Remove the index that is out of window
            
            # === STEP 2: Remove indices of elements smaller than current ===
            # These smaller elements at the back cannot be maximum as long as
            # current element (nums[i]) exists in the window
            # So we remove them to keep deque monotonic decreasing
            while dq and nums[dq[-1]] < current_value:
                dq.pop()  # Remove from back
            
            # === STEP 3: Add current index to the deque ===
            dq.append(i)
            
            # === STEP 4: If window size is reached, add maximum to result ===
            # Window is fully formed when i >= k-1 (first full window ends at i = k-1)
            if i >= k - 1:
                # dq[0] always holds the index of the maximum in current window
                max_in_window = nums[dq[0]]
                result.append(max_in_window)
        
        return result


# ========================================
# EXAMPLE WALKTHROUGH WITH COMMENTS
# ========================================

"""
Let's trace through: nums = [1,3,-1,-3,5,3,6,7], k = 3

i = 0: num = 1
    dq = [0]                          # add 1
    i < 2 → no result yet

i = 1: num = 3
    remove smaller: 1 < 3 → pop 0
    dq = [1]                          # add 3
    i < 2 → no result

i = 2: num = -1
    dq = [1,2]                        # add -1 (3 is still larger)
    i == 2 → window [1,3,-1] → max = nums[1] = 3
    result = [3]

i = 3: num = -3
    dq = [1,2,3]                      # add -3
    i == 3 → window [3,-1,-3] → max = nums[1] = 3
    result = [3,3]

i = 4: num = 5
    remove smaller: -3 < 5 → pop 3
                    -1 < 5 → pop 2
                     3 < 5 → pop 1
    dq = [4]                          # add 5
    i == 4 → window [-1,-3,5] → max = 5
    result = [3,3,5]

i = 5: num = 3
    dq = [4,5]                        # add 3 (5 is larger)
    i == 5 → window [-3,5,3] → max = 5
    result = [3,3,5,5]

i = 6: num = 6
    remove smaller: 3 < 6 → pop 5
                    5 < 6 → pop 4
    dq = [6]                          # add 6
    i == 6 → window [5,3,6] → max = 6
    result = [3,3,5,5,6]

i = 7: num = 7
    remove smaller: 6 < 7 → pop 6
    dq = [7]                          # add 7
    i == 7 → window [3,6,7] → max = 7
    result = [3,3,5,5,6,7]

Final Output: [3,3,5,5,6,7]
"""

# ========================================
# TEST THE SOLUTION
# ========================================

if __name__ == "__main__":
    sol = Solution()
    
    test_cases = [
        ([1,3,-1,-3,5,3,6,7], 3, [3,3,5,5,6,7]),
        ([1], 1, [1]),
        ([1,2,3,4,5], 3, [3,4,5]),
        ([9,8,7,6,5,4,3,2,1], 3, [9,8,7,6,5,4,3]),
        ([], 0, []),
    ]
    
    for i, (nums, k, expected) in enumerate(test_cases):
        result = sol.maxSlidingWindow(nums, k)
        status = "PASS" if result == expected else "FAIL"
        print(f"Test {i+1}: {status}")
        print(f"   Input: nums={nums}, k={k}")
        print(f"   Got:    {result}")
        print(f"   Want:   {expected}\n")