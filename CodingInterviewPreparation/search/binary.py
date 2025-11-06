#%%
def binary_search(arr, target):
	"""
	Perform binary search on a sorted array to find the index of the target value.

	:param arr: List of sorted elements
	:param target: The value to search for
	:return: Index of target in arr if found, otherwise -1
	"""
	left, right = 0, len(arr) - 1

	while left <= right:
		mid = left + (right - left) // 2

		# Check if target is present at mid
		if arr[mid] == target:
			return mid
		# If target is greater, ignore left half
		elif arr[mid] < target:
			left = mid + 1
		# If target is smaller, ignore right half
		else:
			right = mid - 1

	# Target was not found in the array
	return -1
# Example usage
array = [1, 2, 3, 4, 5, 6, 7, 8, 9]
target_value = 5
result = binary_search(array, target_value)
if result != -1:
	print(f"Element found at index: {result}")