from sys import path
path.insert(0, '../') 

from Typhoon import *

x = Matrix4()
q = Quaternion(1,-0.1,0.3,0.5)
p = Vector(1,2,3)

x.setOrientationAndPos(q,p)
print(x)

q2 = x.transformQuaternion()

print(q2)