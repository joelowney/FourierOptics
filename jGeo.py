import numpy as np

def RotateToNormal(nvec):
    # Return rotation matrix that will rotate arbitrary nvec to [0 0 1]

    nvec = nvec / np.linalg.norm(nvec)

    alpha1 = nvec[0]
    beta1 = nvec[1]
    gamma1 = nvec[2]
    theta1 = np.arctan(alpha1/beta1)

    R2_1 = np.array([[np.cos(theta1), -np.sin(theta1)],[+np.sin(theta1), +np.cos(theta1)]])


    R3_1 = np.concatenate(   (np.concatenate((R2_1,np.array([[0,],[0]])),axis=1),np.array([[0,0,1],])),axis=0)

    s0 = np.array([[alpha1],[beta1],[gamma1]])
    s1 = np.matmul(R3_1, s0)

    alpha2 = s1[0]
    beta2 = s1[1]
    gamma2 = s1[2]
    theta2 = np.arctan(beta2[0]/gamma2[0])
    R2_2 = np.array([[np.cos(theta2), -np.sin(theta2)],[+np.sin(theta2), +np.cos(theta2)]])
    R3_2 = np.concatenate( ( np.array([[1,0,0,],]),np.concatenate((np.array([[0, ], [0]]), R2_2), axis=1)),axis=0)
    R3 = np.matmul(R3_2,R3_1)

    return R3

def FindSurfaceNormal(zfun, xq, yq, d=1e-5):
    # Assumption: input function is an anonymous function of two dimensions
    # e.g. z(x,y)
    # d represents a differential unit and should be much smaller than any differences in x,y,z that we would care about
    zq = zfun(xq,yq)


    v0 = np.array([xq, yq, zfun(xq,yq)])

    dx = d * (np.random.randint(2, size=1)-0.5)*2
    dy = d * (np.random.randint(2, size=1)-0.5)*2

    dx = dx[0]
    dy = dy[0]
    ### Find two nearby locations on surface
    v1m = np.array([xq-dx,yq,zfun(xq-dx,yq)])
    v2m = np.array([xq,yq-dy,zfun(xq,yq-dy)])
    v1p = np.array([xq+dx,yq,zfun(xq+dx,yq)])
    v2p = np.array([xq,yq+dy,zfun(xq,yq+dy)])
    ### Calculate the two vectors to those locations within the manifold
    dv1 = v1p - v1m
    dv2 = v2p - v2m
    ### Take cross product of those two vectors within the manifold to find normal direction.
    ### Then normalize to find the surface normal
    nhat = np.cross(dv1,dv2) / np.linalg.norm(np.cross(dv1,dv2))

    return nhat


