from sys import path
path.insert(0, '../') 

import Typhoon.Pfgen
from Typhoon import Particle
from Typhoon.Core import Vector
#from Typhoon.Pfgen import ForceRegistry
from vpython import *

x = Particle.Particle()
x.setPosition(10,100,5)
x.setMass(3)
x.setDamping(1)
#y = Particle.Particle()
#y.setPosition(-50,-50,-30)
#y.setMass(5)
#y.setDamping(1)

y = Vector.Vector(0,100,0)
z= Vector.Vector(50,100,0)

registery = Typhoon.Pfgen.ForceRegistery.ParticleForceRegistery()

pfax = Typhoon.Pfgen.AnchordBungee.ParticleAnchordBungee(y,1,100)
pfgx = Typhoon.Pfgen.Gravity.ParticleGravity(Vector.Vector(0,-9.8,0))
pfdx = Typhoon.Pfgen.Drag.ParticleDrag(2,3)
pfasx = Typhoon.Pfgen.AnchordSpring.ParticleAnchordSpring(y,4,50)
pfbx = Typhoon.Pfgen.Bouyancy.ParticleBouyancy(1,2,0,1000)
#pfy = Bungee.ParticleBungee(y, 1, 2)
registery.add(x,pfax)
registery.add(x,pfgx)
registery.add(x,pfdx)
registery.add(x,pfasx)
registery.add(x,pfbx)
#registery.add(y,pfx)

ballx = sphere (color = color.red, radius = 1)
bally = sphere (color = color.green, radius = 1)

def display():
    scene.width = scene.height = 600
    L = 50
    scene.center = vec(0,0,0)
    scene.range = 3*L
    d = L-2
    R= L/100
    xaxis = cylinder(pos=vec(0,0,0), axis=vec(d,0,0), radius=R, color=color.yellow)
    yaxis = cylinder(pos=vec(0,0,0), axis=vec(0,d,0), radius=R, color=color.yellow)
    zaxis = cylinder(pos=vec(0,0,0), axis=vec(0,0,d), radius=R, color=color.yellow)
    k = 1.02
    h = 0.05*L
    text(pos=xaxis.pos+k*xaxis.axis, text='x', height=h)
    text(pos=yaxis.pos+k*yaxis.axis, text='y', height=h)
    text(pos=zaxis.pos+k*zaxis.axis, text='z', height=h)

    floor = box (pos=vector( 0, 0, 0), size=vector(200, 0.1,200 ),  color = color.blue)

def update():
    while True:
        rate(60)
        for register in registery.registerations:
            registery.updateForces(1/60)
            x.integrate(1/60)
            #y.integrate(1/60)

            ballx.pos = vector(x.getPosition().x,x.getPosition().y,x.getPosition().z)
            bally.pos = vector(y.x,y.y,y.z)


display()
update()
