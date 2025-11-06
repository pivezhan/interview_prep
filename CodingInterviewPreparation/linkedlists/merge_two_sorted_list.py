# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        # Edge cases
        if not list1:
            return list2
        if not list2:
            return list1
        
        # Choose head of merged list
        if list1.val <= list2.val:
            head = list1
            list1 = list1.next
        else:
            head = list2
            list2 = list2.next
        
        # Current pointer to build the result
        current = head
        
        # Merge while both lists have nodes
        while list1 and list2:
            if list1.val <= list2.val:
                current.next = list1
                list1 = list1.next
            else:
                current.next = list2
                list2 = list2.next
            current = current.next
        
        # Attach remaining nodes
        current.next = list1 if list1 else list2
        
        return head
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
	def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
		# Create a dummy node to simplify edge cases
		dummy = ListNode(0)
		current = dummy
		
		# Merge while both lists have nodes
		while list1 and list2:
			if list1.val <= list2.val:
				current.next = list1
				list1 = list1.next
			else:
				current.next = list2
				list2 = list2.next
			current = current.next
		
		# Attach remaining nodes
		current.next = list1 if list1 else list2
		
		return dummy.next  # Return head of merged list, skipping dummy
