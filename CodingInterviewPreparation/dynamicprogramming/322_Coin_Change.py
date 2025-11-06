# 322. Coin Change
# Medium
# Topics
# premium lock icon
# Companies
# You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

# Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

# You may assume that you have an infinite number of each kind of coin.

 

# Example 1:

# Input: coins = [1,2,5], amount = 11
# Output: 3
# Explanation: 11 = 5 + 5 + 1
# Example 2:

# Input: coins = [2], amount = 3
# Output: -1
# Example 3:

# Input: coins = [1], amount = 0
# Output: 0
 

# Constraints:

# 1 <= coins.length <= 12
# 1 <= coins[i] <= 231 - 1
# 0 <= amount <= 104
from typing import List

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # Initialize DP array with amount + 1 (impossible value) as a sentinel
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0  # Base case: 0 amount requires 0 coins

        # For each amount from 1 to target amount

        for i in range(1, amount + 1):
            # Try each coin denomination
            for coin in coins:
                if i >= coin:

                    # i:1,coin:1: dp[1] = min(dp[1],dp[0]+1)=1
                    # i:2,coin:1: dp[2] = min(dp[2],dp[1]+1)=2
                    # i:3,coin:1: dp[3] = min(dp[3],dp[2]+1)=3
                    # i:4,coin:1: dp[4] = min(dp[4],dp[3]+1)=4
                    # i:5,coin:1: dp[5] = min(dp[5],dp[4]+1)=5
                    dp[i] = min(dp[i], dp[i - coin] + 1)
                    print(f"i:{i},coin:{coin},dp[{i}]=min(dp[{i}],dp[{i-coin}]+1)={dp[i]}")

        # Return -1 if no solution is found, otherwise return dp[amount]
        return dp[amount] if dp[amount] != float('inf') else -1


# Input
# coins =
# [1,2,5]
# amount =
# 11
# i:1,coin:1,dp[1]=min(dp[1],dp[0]+1)=1
# i:2,coin:1,dp[2]=min(dp[2],dp[1]+1)=2
# i:2,coin:2,dp[2]=min(dp[2],dp[0]+1)=1
# i:3,coin:1,dp[3]=min(dp[3],dp[2]+1)=2
# i:3,coin:2,dp[3]=min(dp[3],dp[1]+1)=2
# i:4,coin:1,dp[4]=min(dp[4],dp[3]+1)=3
# i:4,coin:2,dp[4]=min(dp[4],dp[2]+1)=2
# i:5,coin:1,dp[5]=min(dp[5],dp[4]+1)=3
# i:5,coin:2,dp[5]=min(dp[5],dp[3]+1)=3
# i:5,coin:5,dp[5]=min(dp[5],dp[0]+1)=1
# i:6,coin:1,dp[6]=min(dp[6],dp[5]+1)=2
# i:6,coin:2,dp[6]=min(dp[6],dp[4]+1)=2
# i:6,coin:5,dp[6]=min(dp[6],dp[1]+1)=2
# i:7,coin:1,dp[7]=min(dp[7],dp[6]+1)=3
# i:7,coin:2,dp[7]=min(dp[7],dp[5]+1)=2
# i:7,coin:5,dp[7]=min(dp[7],dp[2]+1)=2
# i:8,coin:1,dp[8]=min(dp[8],dp[7]+1)=3
# i:8,coin:2,dp[8]=min(dp[8],dp[6]+1)=3
# i:8,coin:5,dp[8]=min(dp[8],dp[3]+1)=3
# i:9,coin:1,dp[9]=min(dp[9],dp[8]+1)=4
# i:9,coin:2,dp[9]=min(dp[9],dp[7]+1)=3
# i:9,coin:5,dp[9]=min(dp[9],dp[4]+1)=3
# i:10,coin:1,dp[10]=min(dp[10],dp[9]+1)=4
# i:10,coin:2,dp[10]=min(dp[10],dp[8]+1)=4
# i:10,coin:5,dp[10]=min(dp[10],dp[5]+1)=2
# i:11,coin:1,dp[11]=min(dp[11],dp[10]+1)=3
# i:11,coin:2,dp[11]=min(dp[11],dp[9]+1)=3
# i:11,coin:5,dp[11]=min(dp[11],dp[6]+1)=3

# from typing import List

# class Solution:
#     def coinChange(self, coins: List[int], amount: int) -> int:
#         # Initialize DP array with amount + 1 (impossible value) as a sentinel
#         dp = [float('inf')] * (amount + 1)
#         dp[0] = 0  # Base case: 0 amount requires 0 coins
#         # Array to track which coin was used for each amount
#         used_coins = [None] * (amount + 1)

#         # For each amount from 1 to target amount
#         for i in range(1, amount + 1):
#             # Try each coin denomination
#             for coin in coins:
#                 if i >= coin:
#                     # Check if using this coin gives a better solution
#                     if dp[i - coin] + 1 < dp[i]:
#                         dp[i] = dp[i - coin] + 1
#                         used_coins[i] = coin  # Store the coin used
#                     print(f"i:{i},coin:{coin},dp[{i}]=min(dp[{i}],dp[{i-coin}]+1)={dp[i]},used_coin={used_coins[i]}")

#         # If no solution is found, return -1
#         if dp[amount] == float('inf'):
#             return -1

#         # Reconstruct the list of coins used
#         coin_list = []
#         current_amount = amount
#         while current_amount > 0:
#             coin = used_coins[current_amount]
#             coin_list.append(coin)
#             current_amount -= coin

#         print(f"Coins used: {coin_list}")
#         return dp[amount]