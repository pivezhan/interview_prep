
# You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

# You may assume the two numbers do not contain any leading zero, except the number 0 itself.

 

# Example 1:


# Input: l1 = [2,4,3], l2 = [5,6,4]
# Output: [7,0,8]
# Explanation: 342 + 465 = 807.
# Example 2:

# Input: l1 = [0], l2 = [0]
# Output: [0]
# Example 3:

# Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
# Output: [8,9,9,9,0,0,0,1]


class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)  # Dummy head for the result linked list
        current = dummy  # Pointer to build the result list
        carry = 0  # Carry for addition
        
        # Process both lists until both are exhausted
        while l1 or l2 or carry:
            # Get values, use 0 if list is exhausted
            x1 = l1.val if l1 else 0
            x2 = l2.val if l2 else 0

            # Compute sum and new carry
            total = x1 + x2 + carry
            carry = total // 10
            digit = total % 10
            
            # Create new node with the computed digit
            current.next = ListNode(digit)
            current = current.next
            
            # Move to next nodes if available
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None

        return dummy.next  # Return head of result list


        