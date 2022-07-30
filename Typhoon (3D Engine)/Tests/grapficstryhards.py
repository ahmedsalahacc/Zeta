
from vpython import *

scene.width = scene.height = 600
L = 50
d = L-2
R= L/100
scene.center = vec(d/2,0,d/2)
scene.range = 0.5*L

xaxis = cylinder(pos=vec(0,0,0), axis=vec(0,0,d), radius=R, color=color.yellow)
yaxis = cylinder(pos=vec(0,0,0), axis=vec(d,0,0), radius=R, color=color.yellow)
zaxis = cylinder(pos=vec(0,0,0), axis=vec(0,d,0), radius=R, color=color.yellow)
floor = box (pos=vector( d/2, -0.5, d/2), size=vector(d, 0.5, d),  color = color.blue)

pos = vector(10,3,10)
offsetSailRod = vector(-4,1.5,0)
offsetSail = vector(-4,4.5,0)

body = box(length=10, height=3, width=4)
body.pos = pos

rod = cylinder(axis=vector(0,6.5,0), radius=0.3)
rod.pos = pos+offsetSailRod

sail = pyramid(color = color.white,size = vector(10,0.3,5))
sail.axis = vector(0,1,0)
sail.size = vector(10,0.3,5)
sail.pos = pos+offsetSail




