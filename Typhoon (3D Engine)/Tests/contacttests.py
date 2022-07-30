from sys import path
path.insert(0, '../') 

from Typhoon import *

#contact = Contact()
#contact.contactNormal = Vector(0.3,1,0.5)
#contact.calculateContactBasis()

#print(contact.contactToWorld)
#print(contact.contactNormal)

def initBody(body, position, velocity):
    body.setPosition(position.x, position.y, position.z)
    body.setVelocity(velocity.x, velocity.y, velocity.z)
    body.setOrientation(1,0.1,0.5,0.8)
    mass = 12
    body.setMass(mass)
    tensor = Matrix3()
    tensor.setBlockInertiaTensor(Vector(2,1,1), mass)
    body.setInertiaTensor(tensor)

    body.setLinearDamping(0.95)
    body.setAngularDamping(0.8)
    body.clearAccumulators()
    body.setAcceleration(0,0,0)

    body.setCanSleep(False)
    body.setAwake()

    body.calculateDerivedData();

body1 = RigidBody()
body2 = RigidBody()

initBody(body1, Vector(0,1,1),Vector(0,0,0))
initBody(body2, Vector(1,3,0),Vector(0,-9.81,0))
body2.setOrientation(1,0.2,0.9,0.8)

contact = Contact()
contact.setBodyData(body1,body2,1,0.1)
angularChange = [Vector(),Vector()]
linearChange = [None, None]
contact.contactNormal = Vector(2,1,8)
contact.contactNormal.normalize()
contact.contactPoint = Vector(0,2,0)
contact.calculateInternals(1/30)
contact.calculateContactBasis()

contact.applyPositionChange(linearChange,angularChange,0.1)

print(contact.relativeContactPosition[0],contact.relativeContactPosition[1])
print(linearChange[0],linearChange[1])
print(angularChange[0], angularChange[1])
print(contact.contactNormal)
