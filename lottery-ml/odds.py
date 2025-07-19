import math

# calculate permutations
def permutations(n, r):
    return math.factorial(n) / math.factorial(n - r)


n = 80
r = 20

# calculate permutations
result = permutations(n, r)

print(result)