import random

a=[1,2,3]
b=[2,3,4,5]
c=random.sample(a,2)

d=list(set(a)-set(c))
print(a,b,c,d)
