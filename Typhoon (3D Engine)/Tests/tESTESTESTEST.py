from sys import path
path.insert(0, '../../')   

from Typhoon import *
from random import uniform
from vpython import *


class Bone(CollisionBox):
    def __init__(self):
        super().__init__()
        self.body = RigidBody()
        self.shape = box()
        self.sphere = sphere()

    def display(self):
        #Body position and axis
        pos = self.body.transformMatrix.getAxisVector(3)
        axis = self.body.transformMatrix.getAxisVector(0)
        up = self.body.transformMatrix.getAxisVector(1)

        #Use graphics library
        pos = pos.toVPyhton()
        axis = axis.toVPyhton()
        up = up.toVPyhton()


        self.sphere.opacity = 0.4
        self.sphere.pos = pos
        self.sphere.radius = self.getCollisionSphere().radius

        self.shape.pos = pos
        self.shape.axis = axis
        self.shape.up = up
        self.shape.size = self.halfSize.toVPyhton()
    
     #We use a sphere to collide bone on bone to allow some limited
     #interpenetration.
    def getCollisionSphere(self):   
        sphere = CollisionSphere()
        sphere.body = self.body;
        sphere.radius = self.halfSize.x #- self.halfSize.x*0.4;
        if self.halfSize.y < sphere.radius: sphere.radius = self.halfSize.y;
        if self.halfSize.z < sphere.radius: sphere.radius = self.halfSize.z;
        sphere.calculateInternals();
        return sphere

    #Sets the bone to a specific location.
    def setState(self, position: "Vector", extents: "Vector"):
        self.body.setPosition(position.x, position.y, position.z)
        self.halfSize = extents

        mass = self.halfSize.x * self.halfSize.y * self.halfSize.z * 8.0
        self.body.setMass(mass)

        tensor = Matrix3()
        tensor.setBlockInertiaTensor(self.halfSize, mass)
        self.body.setInertiaTensor(tensor)

        self.body.setLinearDamping(0.95)
        self.body.setAngularDamping(0.8)
        self.body.clearAccumulators()
        self.body.setAcceleration(0,-9.81,0)

        self.body.setCanSleep(False)
        self.body.setAwake()

        self.body.calculateDerivedData();
        self.calculateInternals();

duration = 1/30
MAX_CONTACT = 256
contacts = []
cData = CollisionData()
cData.contactArray = contacts   
resolver = ContactResolver(MAX_CONTACT*8,MAX_CONTACT*8)

#Setup bones and joints
bone = Bone()
bone.setState(Vector(0, 10, -0.5), Vector(3, 3, 3))
bone.body.orientation.setComponent(1,0.3,0.1,0.5)

bone2 = Bone()
bone2.setState(Vector(0, 15, -0.5), Vector(1, 1, 1))
bone2.body.orientation.setComponent(1,0.3,0.1,0.5)

plane = CollisionPlane()
plane.direction = Vector(0,1,0)
plane.offset = 0;


cData.reset(MAX_CONTACT)
cData.friction = 0.9
cData.restitution = 0.1
cData.tolerance = 0.1

def update(duration):
    bone.body.integrate(duration)
    bone.calculateInternals()

    bone2.body.integrate(duration)
    bone2.calculateInternals()

    contacts.clear()
    #cData.reset(MAX_CONTACT)

    CollisionDetector.boxAndHalfSpace(bone, plane, cData)
    CollisionDetector.boxAndHalfSpace(bone2, plane, cData)

    #CollisionDetector.sphereAndSphere(bone.getCollisionSphere(), bone2.getCollisionSphere(), cData)
    CollisionDetector.sphereAndSphere(bone2.getCollisionSphere(), bone.getCollisionSphere(), cData)
    
    resolver.resolveContacts(cData.contactArray, cData.contactCount,duration)

def graphicsSetup():
        scene.width = scene.height = 600
        L = 50
        d = L-2
        R= L/100
        scene.center = vec(0,0,0)
        scene.camera.pos = vector(1,4,1)
        scene.range = 0.2*L

        zaxis = cylinder(pos=vec(0,0,0), axis=vec(0,0,d), radius=R, color=color.yellow)
        xaxis = cylinder(pos=vec(0,0,0), axis=vec(d,0,0), radius=R, color=color.yellow)
        #yaxis = cylinder(pos=vec(0,0,0), axis=vec(0,d,0), radius=R, color=color.yellow)
        floor = box (pos=vector( d/2, -0.5, d/2), size=vector(d, 0.5, d),  color = color.blue)

def run():
        graphicsSetup()
        while(True):
            rate(30)
            
            bone.display()
            bone2.display()
            update(1/30)
            print(bone2.body.position)


run()