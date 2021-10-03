import numpy as np

def cosd(theta):
    return np.cos(theta*np.pi/180)

def sind(theta):
    return np.sin(theta*np.pi/180)

def tand(theta):
    return np.tan(theta*np.pi/180)

def secd(theta):
    return 1/np.cos(theta*np.pi/180)

def cscd(theta):
    return np.csc(theta*np.pi/180)

def cotd(theta):
    return 1/np.tan(theta*np.pi/180)

def CFC(FunctionIn,NormalizedEdgeCoordinates,DiffOrders):
    NOrders = DiffOrders.size
    LeftBoundariesNormalized = np.concatenate([np.array([0]), NormalizedEdgeCoordinates])
    RightBoundariesNormalized = np.concatenate([NormalizedEdgeCoordinates, np.array([1])])
    FourierCoefficients = np.zeros(NOrders) + 1J

    for aa in range(0, NOrders):
        if (DiffOrders[aa] == 0):
            myval = \
                sum(FunctionIn * RightBoundariesNormalized - FunctionIn * LeftBoundariesNormalized)
        else:
            myval = 1J / (2 * np.pi * DiffOrders[aa]) \
                    * sum(FunctionIn * ( \
                        +1 * np.exp(-2 * 1J * np.pi * DiffOrders[aa] * RightBoundariesNormalized) \
                        - 1 * np.exp(-2 * 1J * np.pi * DiffOrders[aa] * LeftBoundariesNormalized) \
                ))
        FourierCoefficients[aa] = myval


    return FourierCoefficients

def FlipExplodingModes(arr_in):
    """ If a mode is exploding exponentially (imag part negative) change its direction """
    for aa in range(0, arr_in.size):

        if (arr_in[aa].imag < 0):
            arr_in[aa] = -arr_in[aa]

    return arr_in

def ConcatAB(A,B):
    return np.concatenate((A,B),axis=1)

def ConcatAC(A,C):
    return np.concatenate((A,C),axis=0)

def ConcatABCD(A,B,C,D):
    return ConcatAC(ConcatAB(A,B),ConcatAB(C,D))


def FindZeroIndex(InputArray):
    ZeroIndex = 0
    for jj in range(0,np.size(InputArray)):
        if(InputArray[jj]==0):
            ZeroIndex = jj
    return ZeroIndex

def GenToeplitz(A,B):
    q = np.zeros((np.size(A),np.size(A)))+0j

    for aa in range(0,np.size(A)):
        for bb in range(0,np.size(A)):
            if(aa==bb):
                q[aa,bb] = A[0];
            else:
                if(aa>bb):
                    q[aa,bb] = A[aa-bb]
                else:
                    q[aa,bb] = B[bb-aa]

    return q;


def CatRow(arr_list):

    arr_out = np.concatenate(arr_list,0)

    return arr_out

def CatCol(arr_list):
    arr_out = np.concatenate(arr_list,axis=1)

def MatMulRow(arr_list):
    NCat = len(arr_list)

    arr_out = arr_list[0]
    for aa in range(1,NCat):
        arr_out = np.matmul(arr_out,arr_list[aa])

    return arr_out

def MatDivRight(a,b):
    # Solves a/b
    return (np.linalg.solve(b,a))
def MatDivLeft(a,b):
    return np.linalg.solve(a,b)

def StripSmallImaginaryComponent(b,thresh):

    c = b * 0
    for zz in range(0,len(b)):
        if(abs(b[zz].imag)<thresh):
            b[zz] = b[zz].real

    return b

def FindIndicesBySign(b):
    pos_list = []
    neg_list = []
    zer_list = []
    for aa in range (0,len(b)):
        if(b[aa].imag>0):
            pos_list.append(aa)
        elif(b[aa].imag==0):
            zer_list.append(aa)
        else:
            neg_list.append(aa)
    return np.array(pos_list),np.array(neg_list),np.array(zer_list)

def SortListByImag(arr_in):
    [plist, nlist, zlist] = FindIndicesBySign(arr_in)

    plist_sort = arr_in[plist[arr_in[plist].imag.argsort(axis=0)]]
    nlist_sort = arr_in[nlist[arr_in[nlist].imag.argsort(axis=0)]]
    zlist_sort = arr_in[zlist[arr_in[zlist].imag.argsort(axis=0)]]

    slistp = plist[arr_in[plist].imag.argsort(axis=0)]
    slistz = zlist[arr_in[zlist].real.argsort(axis=0)]
    slistn = nlist[arr_in[nlist].imag.argsort(axis=0)]

    slist = np.flipud(np.concatenate([slistn, slistz, slistp], axis=0))

    return slist,arr_in[slist]

def submat(M,ix,iy):
    return M[np.ix_(ix,iy)]

def reshape_to_vect(ar):
    if len(ar.shape) == 1:
      return ar.reshape(ar.shape[0],1)
    return ar


def jprint(M,tol=1e-14):
    if(len(M) == 2):
        M_temp = M
        for aa in range(0,M.shape[0]):
            for bb in range(0,M.shape[1]):
                if(abs(M[aa,bb].imag) < tol):
                   M_temp[aa,bb] = M[aa,bb].real
                if(abs(M[aa,bb].real) < tol):
                    M_temp[aa,bb] = 1j*M_temp[aa,bb].imag
        print(M_temp)
    else:
        M_temp = M
        for aa in range(0, M.shape[0]):
            if (abs(M[aa].imag) < tol):
               M_temp[aa] = M[aa].real
            if (abs(M[aa].real) < tol):
                M_temp[aa] = 1j*M_temp[aa].imag
        print(M_temp)

class jarray(np.ndarray):
    # def __init__(self,mat_in):

    pass