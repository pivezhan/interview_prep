# from numpy import np


def quicksort(arr):
	"""Sorts an array using the quicksort algorithm.

	Args:
		arr (list): The list of elements to be sorted.

	Returns:
		list: A new sorted list.
	"""
	if len(arr) <= 1:
		return arr
	else:
		pivot = arr[len(arr) // 2]
		left = [x for x in arr if x < pivot]
		middle = [x for x in arr if x == pivot]
		right = [x for x in arr if x > pivot]
		return quicksort(left) + middle + quicksort(right)
def __init__():
	array = [3,6,8,10,1,2,1]
	print("Original array:", array)
	sorted_array = quicksort(array)
	print("Sorted array:", sorted_array)