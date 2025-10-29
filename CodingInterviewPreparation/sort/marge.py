def merged_sort(arr):
	if len(arr) <= 1:
		return arr

	mid = len(arr) // 2
	left_half = merged_sort(arr[:mid])
	right_half = merged_sort(arr[mid:])

	return merge(left_half, right_half)
def merge(left, right):
	sorted_array = []
	i = j = 0

	while i < len(left) and j < len(right):
		if left[i] < right[j]:
			sorted_array.append(left[i])
			i += 1
		else:
			sorted_array.append(right[j])
			j += 1

	# Append any remaining elements from both halves
	sorted_array.extend(left[i:])
	sorted_array.extend(right[j:])

	return sorted_array
# Example usage
array = [38, 27, 43, 3, 9, 82, 10]
print("Original array:", array)
sorted_array = merged_sort(array)
print("Sorted array:", sorted_array)
