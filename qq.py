# import math
# def df(x, y):
#     return 4*(x**2+y**2)*math.exp(x**2-y**2)

# def f(x,y):
#     return math.exp(x**2-y**2)

# print(df(4/3, 4/3) - 9 * f(4/3, 1) - 9 * f(1, 4/3))
# print(df(5/3, 4/3) - 9 * f(5/3, 1) - 9 * f(6/3, 4/3))
# print(df(4/3, 5/3) - 9 * f(1, 5/3) - 9 * f(4/3, 2))
# print(df(5/3, 5/3) - 9 * f(2, 5/3) - 9 * f(5/3, 2))


import math
def df(x, y):
    return  -math.atan(x/y)

def mu1(x, y):
    return 0

def mu2(x, y):
    return 0

def mu3(x, y):
    return (math.sin(math.pi*x))**2

def mu4(x, y):
    return (math.cosh((x-1)*(x-2)))-1

# Node (4/3, 4/3): influenced by left boundary (mu1 at x=1) and bottom boundary (mu3 at y=1)
print(df(4/3, 4/3) - 9 * mu1(1, 4/3) - 9 * mu3(4/3, 1))

# Node (5/3, 4/3): influenced by right boundary (mu2 at x=2) and bottom boundary (mu3 at y=1)
print(df(5/3, 4/3) - 9 * mu2(2, 4/3) - 9 * mu3(5/3, 1))

# Node (4/3, 5/3): influenced by left boundary (mu1 at x=1) and top boundary (mu4 at y=2)
print(df(4/3, 5/3) - 9 * mu1(1, 5/3) - 9 * mu4(4/3, 2))

# Node (5/3, 5/3): influenced by right boundary (mu2 at x=2) and top boundary (mu4 at y=2)
print(df(5/3, 5/3) - 9 * mu2(2, 5/3) - 9 * mu4(5/3, 2))