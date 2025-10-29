# # Enter your code here. Read input from STDIN. Print output to STDOUT
# collections.OrderedDict
# An OrderedDict is a dictionary that remembers the order of the keys that were inserted first. If a new entry overwrites an existing entry, the original insertion position is left unchanged.

# Example

# Code

# >>> from collections import OrderedDict
# >>> 
# >>> ordinary_dictionary = {}
# >>> ordinary_dictionary['a'] = 1
# >>> ordinary_dictionary['b'] = 2
# >>> ordinary_dictionary['c'] = 3
# >>> ordinary_dictionary['d'] = 4
# >>> ordinary_dictionary['e'] = 5
# >>> 
# >>> print ordinary_dictionary
# {'a': 1, 'c': 3, 'b': 2, 'e': 5, 'd': 4}
# >>> 
# >>> ordered_dictionary = OrderedDict()
# >>> ordered_dictionary['a'] = 1
# >>> ordered_dictionary['b'] = 2
# >>> ordered_dictionary['c'] = 3
# >>> ordered_dictionary['d'] = 4
# >>> ordered_dictionary['e'] = 5
# >>> 
# >>> print ordered_dictionary
# OrderedDict([('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5)])
# Task

# You are the manager of a supermarket.
# You have a list of  items together with their prices that consumers bought on a particular day.
# Your task is to print each item_name and net_price in order of its first occurrence.

# item_name = Name of the item.
# net_price = Quantity of the item sold multiplied by the price of each item.

# Input Format

# The first line contains the number of items, .
# The next  lines contains the item's name and price, separated by a space.

# Constraints


# Output Format

# Print the item_name and net_price in order of its first occurrence.

# Sample Input

# 9
# BANANA FRIES 12
# POTATO CHIPS 30
# APPLE JUICE 10
# CANDY 5
# APPLE JUICE 10
# CANDY 5
# CANDY 5
# CANDY 5
# POTATO CHIPS 30
# Sample Output

# BANANA FRIES 12
# POTATO CHIPS 60
# APPLE JUICE 20
# CANDY 20
# Explanation

# BANANA FRIES: Quantity bought: , Price: 
# Net Price: 
# POTATO CHIPS: Quantity bought: , Price: 
# Net Price: 
# APPLE JUICE: Quantity bought: , Price: 
# Net Price: 
# CANDY: Quantity bought: , Price: 
# Net Price: 


from collections import OrderedDict

if __name__ == '__main__':
    n = int(input())
    ordinary_items = OrderedDict()
    
    for i in range(n): 
        get_inp = input().split()
        # print(get_inp)
        if len(get_inp) == 3:
            
            name1, name2, count = get_inp
            item_name = name1 + " " + name2
        else:
            item_name, count = get_inp
        if item_name in ordinary_items: 
            ordinary_items[item_name] = int(count) + ordinary_items[item_name]
        else:
            ordinary_items[item_name] = int(count)
    for item_name, count in ordinary_items.items():
        print(item_name, count)
