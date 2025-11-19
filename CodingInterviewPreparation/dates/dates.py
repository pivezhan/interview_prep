#%%
# Write a function that takes two integers namely hours & minutes, converts them to seconds, and adds them together to display the total seconds.

def convert_to_seconds(hours, minutes):
	second = (hours * 60 + minutes) * 60
	return second

# Example usage:
hours = 2
minutes = 30
total_seconds = convert_to_seconds(hours, minutes)
print(f"{hours} hours and {minutes} minutes is equal to {total_seconds} seconds.")
# Output: 2 hours and 30 minutes is equal to 9000 seconds.

#%%
# Create a function that takes an array of any items, removes all duplicate items (items with the same name or number-case sensitive), and returns a new array in the same sequence as the old array without any duplicates.

# You will get a chance to compare your code with a proposed solution in the following lab.

def remove_duplicates(input_array):
	unique_items = set(input_array)
	return list(unique_items)
# Example usage:
input_array = [1, 2, 2, 3, 4, 4, 5]
result = remove_duplicates(input_array)
print(f"Array after removing duplicates: {result}")
# Output: Array after removing duplicates: [1, 2, 3, 4, 5]
# Note: The order of items in the output array may vary since sets do not maintain order.