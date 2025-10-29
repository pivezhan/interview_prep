# Given an integer array nums, return the length of the longest strictly increasing subsequence.

 

# Example 1:

# Input: nums = [10,9,2,5,3,7,101,18]
# Output: 4
# Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.
# Example 2:

# Input: nums = [0,1,0,3,2,3]
# Output: 4
# Example 3:

# Input: nums = [7,7,7,7,7,7,7]
# Output: 1
 

# Constraints:

# 1 <= nums.length <= 2500
# -104 <= nums[i] <= 104
 

# Follow up: Can you come up with an algorithm that runs in O(n log(n)) time complexity?

class Solution:
	def lengthOfLIS(self, nums: List[int]) -> int:
		n = len(nums)
		dp = [1] * n  # dp[i] = length of LIS ending at index i
		
		for i in range(n):
			for j in range(i):
				if nums[i] > nums[j]:
					dp[i] = max(dp[i], dp[j] + 1)
		
		return max(dp)

	# o(nlog(n)) algorithm
	def lengthOfLIS_optimized(self, nums: List[int]) -> int:
		from bisect import bisect_left

		sub = []
		for num in nums:
			pos = bisect_left(sub, num)
			if pos == len(sub):
				sub.append(num)
			else:
				sub[pos] = num
		return len(sub)

	def bisect_left(a, x):
		lo, hi = 0, len(a)
		while lo < hi:
			mid = (lo + hi) // 2
			if a[mid] < x:
				lo = mid + 1
			else:
				hi = mid
		return lo
			