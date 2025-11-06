#%%
def linear_search(arr, target):
	"""
	Perform a linear search for the target in the given array.

	Parameters:
	arr (list): The list to search through.
	target: The value to search for.

	Returns:
	int: The index of the target if found, otherwise -1.
	"""
	for index in range(len(arr)):
		if arr[index] == target:
			return index
	return -1
# Example usage
array = [5, 3, 8, 4, 2]
target_value = 4
result = linear_search(array, target_value)
if result != -1:
	print(f"Element found at index: {result}")
else:
	print("Element not found in the array.")
