from sys import path
path.insert(0, '../../')   
from Typhoon import *
from random import uniform
from vpython import *

drawGraphics = True 


class Drone(CollisionBox):
    def __init__(self):
        super().__init__()
        self.body = RigidBody()
        if drawGraphics:
            self.box = box()
            self.motors = [cylinder(),cylinder(),cylinder(),cylinder()]

    def display(self):
        #Body position and axis
        pos = self.body.transformMatrix.getAxisVector(3)
        axis = self.body.transformMatrix.getAxisVector(0)
        up = self.body.transformMatrix.getAxisVector(1)
        #Use graphics library
        pos = pos.toVPython()
        axis = axis.toVPython()
        up = up.toVPython()
        #Draw drone
        self.box.pos = pos
        self.box.axis = axis
        self.box.up = up
        self.box.size = (self.halfSize*2).toVPython()
        self.box.color = color.red
        #Draw 4 motors
        self.motors[0].axis = up
        self.motors[0].pos = self.body.getPointInWorldSpace(Vector(self.halfSize.x,2*self.halfSize.y/3,self.halfSize.z)).toVPython()
        self.motors[0].radius = (self.halfSize.x+self.halfSize.z)/4
        self.motors[0].color = color.white
        self.motors[1].axis = up
        self.motors[1].pos = self.body.getPointInWorldSpace(Vector(self.halfSize.x,2*self.halfSize.y/3,-self.halfSize.z)).toVPython()
        self.motors[1].radius = (self.halfSize.x+self.halfSize.z)/4
        self.motors[1].color = color.white
        self.motors[2].axis = up
        self.motors[2].pos = self.body.getPointInWorldSpace(Vector(-self.halfSize.x,2*self.halfSize.y/3,self.halfSize.z)).toVPython()
        self.motors[2].radius = (self.halfSize.x+self.halfSize.z)/4
        self.motors[2].color = color.white
        self.motors[3].axis = up
        self.motors[3].pos = self.body.getPointInWorldSpace(Vector(-self.halfSize.x,2*self.halfSize.y/3,-self.halfSize.z)).toVPython()
        self.motors[3].radius = (self.halfSize.x+self.halfSize.z)/4
        self.motors[3].color = color.white

    #Sets the Drone to a specific location, with specific parameters
    def setState(self, position: "Vector", extents: "Vector"):
        #Set position and size
        self.body.setPosition(position.x, position.y, position.z)
        self.halfSize = extents.copy()
        #Mass and inertia
        mass = self.halfSize.x * self.halfSize.y * self.halfSize.z * 8.0
        self.body.setMass(mass)
        tensor = Matrix3()
        tensor.setBlockInertiaTensor(self.halfSize, mass)
        self.body.setInertiaTensor(tensor)
        #Damping 
        self.body.setLinearDamping(0.95)
        self.body.setAngularDamping(0.8)
        self.body.clearAccumulators()
        #Acceleration
        self.body.setAcceleration(0,-9.81 ,0)
        #Body cant go to sleep 
        self.body.setCanSleep(False)
        self.body.setAwake()
        #Calculate derieved data of both Drone and it's collision box
        self.body.calculateDerivedData();
        self.calculateInternals();

        

class DroneDemo():
    FREQ = 30
    
    def __init__(self):
        self.MAX_CONTACT = 32
        #Array of contacts
        self.contacts = []
        #Data of collision detector
        self.cData = CollisionData()
        self.cData.contactArray = self.contacts
        self.cData.friction = 0.9
        self.cData.restitution = 0.6
        self.cData.tolerance = 0.1
        #Contact resolver
        self.resolver = ContactResolver(self.MAX_CONTACT*6,self.MAX_CONTACT*6)
        #Drone initialization
        self.drone = Drone()
        #Forces initialization 
        self.reset(Vector(0,5,0), Vector(3,0.4,2))
        #Add forces

        
        
        #A sphere which the drone follows and try to stay at its center
        self.goalSphere = sphere()
        self.goalSphere.pos = vector(0,10,0)
        self.goalSphere.color = color.blue
        self.goalSphere.opacity = 0.3



    def addForces(self, force):
        self.drone.body.addForceAtBodyPoint(force[0], Vector(self.drone.halfSize.x,self.drone.halfSize.y,self.drone.halfSize.z))
        self.drone.body.addForceAtBodyPoint(force[1], Vector(self.drone.halfSize.x,self.drone.halfSize.y,-self.drone.halfSize.z))
        self.drone.body.addForceAtBodyPoint(force[2], Vector(-self.drone.halfSize.x,self.drone.halfSize.y,self.drone.halfSize.z))
        self.drone.body.addForceAtBodyPoint(force[3], Vector(-self.drone.halfSize.x,self.drone.halfSize.y,-self.drone.halfSize.z))


    def reset(self, position, extent):
        self.drone.setState(position, extent)
        self.cData.reset(self.MAX_CONTACT)

    def updateObjects(self, duration: float):
        self.drone.body.integrate(duration)
        self.drone.calculateInternals()

    def generateContacts(self):
        #Ground
        plane = CollisionPlane()
        plane.direction = Vector(0,1,0)
        plane.offset = 0;
        #Contact data reset
        self.cData.reset(self.MAX_CONTACT)
        #Drone collision with ground
        CollisionDetector.boxAndHalfSpace(self.drone, plane, self.cData)

    def update(self,duration):
        self.addForces([Vector(3, 100,5),Vector(3, 100,5),Vector(3, 100,5),Vector(3, 100,5)] )
        self.updateObjects(duration)
        self.generateContacts()
        self.resolver.resolveContacts(self.cData.contactArray, self.cData.contactCount,duration)
    
    def graphicsSetup(self):
        scene.width = scene.height = 600
        L = 50
        d = L-2
        R= L/100
        scene.center = vec(0,0,0)
        scene.range = 0.2*L
        scene.bind('keydown', self.keyInput)
        #zaxis = cylinder(pos=vec(0,0,0), axis=vec(0,0,d), radius=R, color=color.yellow)
        #xaxis = cylinder(pos=vec(0,0,0), axis=vec(d,0,0), radius=R, color=color.yellow)
        #yaxis = cylinder(pos=vec(0,0,0), axis=vec(0,d,0), radius=R, color=color.yellow)
        floor = box (pos=vector( 0, -0.25, 0), size=vector(d, 0.5, d),  color = color.green)

    def display(self):
        self.drone.display()

    def run(self):
        if drawGraphics:
            self.graphicsSetup()
        while(True):
            rate(self.FREQ)
            self.update(1/self.FREQ)
            if drawGraphics:
                self.display()
    
    def keyInput(self, evt):
        if evt.key == 'w' or evt.key == 'W':
            self.goalSphere.pos.z -= 1
        elif evt.key == 's' or evt.key == 'S':
            self.goalSphere.pos.z += 1
        elif evt.key == 'a' or evt.key == 'A':
            self.goalSphere.pos.x -= 1
        elif evt.key == 'd' or evt.key == 'D':
            self.goalSphere.pos.x += 1
        elif evt.key == 'q' or evt.key == 'Q':
            self.goalSphere.pos.y -= 1
        elif evt.key == 'e' or evt.key == 'E':
            self.goalSphere.pos.y += 1
    

x = DroneDemo()
x.run()



