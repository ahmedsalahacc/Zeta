from sys import path
path.insert(0, '../')

from Typhoon import *
from random import uniform

x = Matrix3( 0.9955864, -0.0868292,  0.0356144,
   0.0874380,  0.9960430, -0.0159048,
  -0.0340925,  0.0189486,  0.9992390)
#for i in range(9):
#   x.data[i] = uniform(0,1)

#x.data[0] = uniform(0,1)
#x.data[1] = 0
#x.data[2] = uniform(0,1)
#x.data[3] = 0
#x.data[4] = 1
#x.data[5] = 0
#x.data[6] = uniform(0,1)
#x.data[7] = 0
#x.data[8] = uniform(0,1)



print(x)
y = Vector(1,2,5)
y.normalize()
print(y)

print(x*y)



