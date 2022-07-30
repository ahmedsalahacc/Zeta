from sys import path
path.insert(0, '../') 

from vpython import *
from Typhoon import *

class ParticleGraphics:
        def __init__(self):
            self.particle = Particle()
            self.ball = sphere (color = color.green, radius = 0.2)

class RodGraphics:
        def __init__(self):
            self.rod = ParticleCable()
            self.shape = cylinder(radius = 0.1,color=color.blue)

x = ParticleGraphics()
y = ParticleGraphics()
freq = 60
x.particle.setPosition(0,6,0)
y.particle.setPosition(1,5,0)
x.particle.setMass(1)
y.particle.setMass(1)
x.ball.color = color.yellow
x.particle.setAcceleration(0,-9.81,0)
#y.particle.setAcceleration(0,-3.1,0)

rod = RodGraphics()
rod.rod.particles[0] = x.particle
rod.rod.particles[1] = y.particle
rod.rod.maxLength = 3

contacts = [ParticleContact()]*6
contactGround = GroundContacts([y.particle,x.particle])
resolver = ParticleContactResolver(5)


def update():
    scene.width = scene.height = 600
    scene.center = vec(0,0,0)
    scene.range = 20
    floor = box (pos=vector( 0, -0.01/2-0.2, 0), size=vector(50, 0.01, 50),  color = color.red)

    while True:
        rate(60)
        rod.rod.addContact(contacts[0],5)
        contactGround.addContact(contacts,5,1)
        resolver.resolveContacts(contacts,1/freq)
       
        position = x.particle.getPosition()
        x.ball.pos = vec(position.x,position.y,position.z)
        
        position = y.particle.getPosition()
        y.ball.pos = vec(position.x,position.y,position.z)
        scene.center = vec(position.x,position.y,position.z)

        p0 = rod.rod.particles[0].getPosition()
        p1 = rod.rod.particles[1].getPosition()
        rod.shape.pos = vec(p0.x,p0.y,p0.z)
        rod.shape.axis = vec(p1.x-p0.x,p1.y-p0.y,p1.z-p0.z)
        x.particle.integrate(1/freq)
        y.particle.integrate(1/freq)
        x.particle.position.z+=0.05
        x.particle.position.x+=0.01
        x.particle.position.y+=0.3

update()

