#%%
### Insertion Sort Implementation
def insertion_sort(arr):
	"""Sorts an array using the insertion sort algorithm.

	Args:
		arr (list): A list of comparable elements.

	Returns:
		list: A new list containing the sorted elements.
	"""
	sorted_arr = arr.copy()
	for i in range(1, len(sorted_arr)):
		key = sorted_arr[i]
		j = i - 1
		while j >= 0 and key < sorted_arr[j]:
			sorted_arr[j + 1] = sorted_arr[j]
			j -= 1
		sorted_arr[j + 1] = key
	return sorted_arr
# Example usage
array = [12, 11, 13, 5, 6]
print("Original array:", array)
sorted_array = insertion_sort(array)
print("Sorted array:", sorted_array)
