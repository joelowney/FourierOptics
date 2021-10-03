
import numpy as np
import jmath as jmath

from tabulate import tabulate

import matplotlib.pyplot as plt,  mpld3

from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection

# import h5py

""" 
Classes:
FMMobj
Grating
InputRay
"""


class jrcwa:
    def __init__(self):



        self.AzimuthAngle_deg = 0
        self.SlantAngle_deg = 30
        print("Initializing FMM object")
        self.n_reflection = 1.00
        self.n_transmission = 1.50 # 0.06+3.25j
        self.IncidentLight = InputRay(lambda0=0.525, nx_global_input=0.0001, ny_global_input=0.00001, \
                     Ex_global_input=0.0, Ey_global_input=1.0, n_reflection=self.n_reflection,\
                      AzimuthAngle_deg=self.AzimuthAngle_deg)

        self.Grating = Grating(self,\
            LayerThicknesses = np.array([0.1]),\
            layer_materials = [np.array([1.0,1.5])],\
            MaterialBoundaries = [np.array([0.5])],\
            UnitCellWidth=0.5,\
            AzimuthAngle_deg = self.AzimuthAngle_deg,\
            SlantAngle_deg = self.SlantAngle_deg \
            )

        self.M = 5

        self.HasComputed = False

        self.CalculateIndices()
        self.InitializeMatrices()
        self.LoopLayers()
        self.ConnectToInternalRegion()
        self.CalculateEfficiencies()


        """ Write file outputs """
        """
        fnameout = 'data123.h5'
        hf = h5py.File(fnameout, 'w')
        hf.create_dataset('nx_output', data=self.nx_global_output)
        hf.create_dataset('ny_output', data=self.ny_global_output)
        hf.create_dataset('Eff_ref', data=self.Eff_reflection)
        hf.close()
        """
        print("Computation complete!")
        # print(self.nx_global_output)
        # print(self.Eff_reflection)



    def __str__(self):
        if(self.HasComputed):

            nmax = np.fmax(self.n_reflection, self.n_transmission)

            nx_arr = np.array([])
            ny_arr = np.array([])
            r_arr = np.array([])
            t_arr = np.array([])

            for aa in range(0, len(self.nx_global_output)):
                mynx = (self.nx_global_output[aa])
                myny = self.ny_global_output[aa]
                myref = (self.Eff_reflection[aa])
                mytrn = (self.Eff_transmission[aa])
                if ((mynx > -nmax) and (mynx < +nmax)):
                    nx_arr = np.append(nx_arr, mynx)
                    ny_arr = np.append(ny_arr, myny)
                    r_arr = np.append(r_arr, myref)
                    t_arr = np.append(t_arr, mytrn)

            nx_row = ["nx"]
            ny_row = ["ny"]
            r_row = ["Ref"]
            t_row = ["Trn"]

            for aa in range(0, len(nx_arr)):
                nx_row.append(nx_arr[aa])
                ny_row.append(ny_arr[aa])
                r_row.append(r_arr[aa])
                t_row.append(t_arr[aa])

            m = [nx_arr, ny_arr, r_arr, t_arr, nx_row]
            m = [nx_row, ny_row, r_row, t_row]
            return tabulate(m, tablefmt="fancy_grid")

        else:
            return "Has not computed."

    def Compute(self):
        self.CalculateIndices()
        self.InitializeMatrices()
        self.LoopLayers()
        self.ConnectToInternalRegion()
        self.CalculateEfficiencies()
        self.HasComputed = True

    def CalculateIndices(self):
        M = self.M

        self.DiffOrders = np.arange(-M,M+1)
        self.NumberOfLayers = self.Grating.LayerThicknesses.size
        self.NumberOfDiffOrders = self.DiffOrders.size
        self.InputModeIndex = M

        self.DiffOrderIndices1 = np.arange(0,2*self.NumberOfDiffOrders)
        self.DiffOrderIndices2 = self.DiffOrderIndices1 + 2*self.NumberOfDiffOrders

    def InitializeMatrices(self):
        self.kx_m_local =  self.IncidentLight.kx_local_input + self.DiffOrders * \
                           2 * np.pi / self.Grating.UnitCellWidth

        self.X = np.diag(self.kx_m_local)
        self.I1 = np.eye(self.NumberOfDiffOrders)
        self.I2 = np.eye(self.NumberOfDiffOrders*2)
        self.I4 = np.eye(self.NumberOfDiffOrders*4)

        self.Z1 = np.zeros((self.NumberOfDiffOrders,self.NumberOfDiffOrders))
        self.Z2 = np.zeros((self.NumberOfDiffOrders*2,self.NumberOfDiffOrders*2))

        self.kz_m_transmission = \
            np.sqrt( self.IncidentLight.k0 **2*(1+0J) * self.n_transmission**2 \
                - self.kx_m_local**2 - self.IncidentLight.ky_local_input**2)

        self.kz_m_reflection = \
            np.sqrt(self.IncidentLight.k0**2*(1+0J) * self.n_reflection**2 \
                - self.kx_m_local**2 - self.IncidentLight.ky_local_input**2)

        self.kz_m_transmission = jmath.FlipExplodingModes(self.kz_m_transmission)
        self.kz_m_reflection = jmath.FlipExplodingModes(self.kz_m_reflection)

        self.Ajplus1 = np.eye(2*self.NumberOfDiffOrders)
        self.Bjplus1 = np.eye(2*self.NumberOfDiffOrders)

        Cjplus1_A = -np.diag(self.kx_m_local * self.IncidentLight.ky_local_input / \
                             (self.IncidentLight.k0 * self.kz_m_transmission))
        Cjplus1_B = -np.diag((self.IncidentLight.ky_local_input**2 + self.kz_m_transmission**2)\
                             /(self.IncidentLight.k0 * self.kz_m_transmission))
        Cjplus1_C = +np.diag((self.kx_m_local**2 + self.kz_m_transmission**2) / \
                             (self.IncidentLight.k0 * self.kz_m_transmission))
        Cjplus1_D = +np.diag(self.kx_m_local*self.IncidentLight.ky_local_input / \
                             (self.IncidentLight.k0 * self.kz_m_transmission))
        self.Cjplus1 = jmath.ConcatABCD(Cjplus1_A,Cjplus1_B,Cjplus1_C,Cjplus1_D)
        self.Djplus1 = -self.Cjplus1

        # Initialize S-Matrix
        self.Suu = self.I2
        self.Sud = self.Z2
        self.Sdu = self.Z2
        self.Sdd = self.I2

        self.Fpos = self.I2
        self.Fneg = self.I2



    def LoopLayers(self):
        for LayerIndex in range(0,self.NumberOfLayers):
            self.UpdateSMatrix(LayerIndex)

    def UpdateSMatrix(self,LayerIndex):
        MyLayer_MaterialIndices = self.Grating.layer_materials[LayerIndex]
        MyLayer_MaterialBoundaries = self.Grating.MaterialBoundaries[LayerIndex]
        MyLayer_EpsilonVector = MyLayer_MaterialIndices ** 2
        MyLayer_XiVector = 1 / MyLayer_MaterialIndices ** 2

        DoubleDiffOrders = np.arange(self.DiffOrders[0] - \
            self.DiffOrders[-1],(self.DiffOrders[-1]-self.DiffOrders[0])+1)

        self.InputIndex2 = self.M*2

        ### Calculate Fourier Coefficients
        FC_Epsilon = jmath.CFC(MyLayer_MaterialIndices ** 2, \
                               MyLayer_MaterialBoundaries,\
                               DoubleDiffOrders)
        FC_EPS_INV = jmath.CFC(1/MyLayer_MaterialIndices ** 2,\
                               MyLayer_MaterialBoundaries,\
                               DoubleDiffOrders)


        ### Construct Rx for x-fields, applying Laurent's rule step 2
        Rx = jmath.MatDivRight(\
            self.I1,\
            jmath.GenToeplitz(\
                FC_EPS_INV[self.InputIndex2:],\
                np.flipud(FC_EPS_INV[0:self.InputIndex2+1])))
        ### Calculate Ry
        Ry =  \
            jmath.GenToeplitz(\
                FC_Epsilon[self.InputIndex2:],\
                np.flipud(FC_Epsilon[0:self.InputIndex2+1])\
                )


        J = jmath.MatDivRight(\
            np.eye(self.NumberOfDiffOrders) , \
            (\
            Ry * jmath.cosd(self.Grating.SlantAngle_deg) ** 2 + \
            Rx * jmath.sind(self.Grating.SlantAngle_deg) ** 2 \
                )\
            )

        M11 = jmath.MatMulRow([self.X,J,Rx]) * jmath.sind(self.Grating.SlantAngle_deg)
        M12 = np.zeros((self.NumberOfDiffOrders,self.NumberOfDiffOrders))
        M13 = self.IncidentLight.ky_local_input / self.IncidentLight.k0 \
            * np.matmul(self.X,J) * jmath.cosd(self.Grating.SlantAngle_deg)
        M14 = 1/self.IncidentLight.k0 * (self.IncidentLight.k0**2 * self.I1 - \
            jmath.MatMulRow([self.X,J,self.X])) * jmath.cosd(self.Grating.SlantAngle_deg)

        M_row1 = np.concatenate([M11,M12,M13,M14],axis=1)


        M21 = self.IncidentLight.ky_local_input*(np.matmul(J,Rx) - self.I1)*jmath.sind(self.Grating.SlantAngle_deg)
        M22 = self.X * jmath.sind(self.Grating.SlantAngle_deg)
        M23 = 1/self.IncidentLight.k0 * (self.IncidentLight.ky_local_input**2 * J - self.IncidentLight.k0 **2 * self.I1) * jmath.cosd(self.Grating.SlantAngle_deg)
        M24 = -self.IncidentLight.ky_local_input/self.IncidentLight.k0 * np.matmul(J,self.X) * jmath.cosd(self.Grating.SlantAngle_deg)
        M_row2 = np.concatenate([M21,M22,M23,M24],axis=1)

        M31 = -(self.IncidentLight.ky_local_input / self.IncidentLight.k0) * self.X * jmath.cosd(self.Grating.SlantAngle_deg)
        M32 = +1 / self.IncidentLight.k0 * (np.matmul(self.X, self.X) - self.IncidentLight.k0 ** 2 * Ry) * jmath.cosd(\
                self.Grating.SlantAngle_deg)
        M33 = self.X * jmath.sind(self.Grating.SlantAngle_deg)
        M34 = np.zeros((self.NumberOfDiffOrders, self.NumberOfDiffOrders))

        M_row3 = np.concatenate([M31,M32,M33,M34],axis=1)

        M41 = self.IncidentLight.k0 * np.matmul(Rx,\
                (self.I1-np.matmul(J,Rx)*jmath.sind(self.Grating.SlantAngle_deg)**2)) * jmath.secd(self.Grating.SlantAngle_deg)\
              - (self.IncidentLight.ky_local_input**2/self.IncidentLight.k0)* self.I1*jmath.cosd(self.Grating.SlantAngle_deg)
        M42 = self.IncidentLight.ky_local_input/self.IncidentLight.k0     * self.X * jmath.cosd(self.Grating.SlantAngle_deg)
        M43 = self.IncidentLight.ky_local_input*(self.I1 - np.matmul(Rx,J)) * jmath.sind(self.Grating.SlantAngle_deg)
        M44 = jmath.MatMulRow([Rx,J,self.X]) * jmath.sind(self.Grating.SlantAngle_deg)

        M_row4 = np.concatenate([M41,M42,M43,M44],axis=1)

        M = np.concatenate([M_row1,M_row2,M_row3,M_row4],axis=0)

        ### Solve eigen-value problem
        [EigenValues, M] = np.linalg.eig(M)

        EigenValues = jmath.StripSmallImaginaryComponent(EigenValues,self.IncidentLight.k0 * 1e-12)

        [SortIndex,EigenValues_Sorted] = jmath.SortListByImag(EigenValues)
        EigenValues = EigenValues[SortIndex]

        M = M[:,SortIndex]

        self.Aj = M[np.ix_(self.DiffOrderIndices1,self.DiffOrderIndices1)]
        self.Bj = M[np.ix_(self.DiffOrderIndices1,self.DiffOrderIndices2)]
        self.Cj = M[np.ix_(self.DiffOrderIndices2,self.DiffOrderIndices1)]
        self.Dj = M[np.ix_(self.DiffOrderIndices2,self.DiffOrderIndices2)]

        gamma1 = EigenValues[self.DiffOrderIndices1]
        gamma2 = EigenValues[self.DiffOrderIndices2]

        ### Update Z
        my_A = self.Ajplus1 + jmath.MatMulRow([self.Bjplus1,self.Fneg,self.Sdu,self.Fpos])
        my_B = -self.Bj
        my_C = self.Cjplus1 + jmath.MatMulRow([self.Djplus1,self.Fneg,self.Sdu,self.Fpos])
        my_D = -self.Dj
        my_denom = jmath.ConcatABCD(my_A,my_B,my_C,my_D)
        Z = jmath.MatDivRight(self.I4,my_denom)

        ### Update S-matrix
        self.Sdu = \
            + np.matmul(jmath.submat(Z,self.DiffOrderIndices2,self.DiffOrderIndices1),self.Aj) \
            + np.matmul(jmath.submat(Z,self.DiffOrderIndices2,self.DiffOrderIndices2),self.Cj)
        self.Sud = \
            + self.Sud \
            - jmath.MatMulRow([\
                self.Suu,\
                self.Fpos,\
                + np.matmul(jmath.submat(Z,self.DiffOrderIndices1,self.DiffOrderIndices1),self.Bjplus1) \
                + np.matmul(jmath.submat(Z,self.DiffOrderIndices1,self.DiffOrderIndices2),self.Djplus1),\
                self.Fneg,\
                self.Sdd])
        self.Suu = jmath.MatMulRow([\
            self.Suu,\
            self.Fpos,\
            (
                + np.matmul(jmath.submat(Z,self.DiffOrderIndices1,self.DiffOrderIndices1),self.Aj) \
                + np.matmul(jmath.submat(Z,self.DiffOrderIndices1,self.DiffOrderIndices2),self.Cj)
            )])
        self.Sdd = - jmath.MatMulRow([\
                + np.matmul(jmath.submat(Z,self.DiffOrderIndices2,self.DiffOrderIndices1),self.Bjplus1) \
                + np.matmul(jmath.submat(Z,self.DiffOrderIndices2,self.DiffOrderIndices2),self.Djplus1),\
            self.Fneg,\
            self.Sdd])

        ### Shift matrices for next iteration:
        self.Ajplus1 = self.Aj
        self.Bjplus1 = self.Bj
        self.Cjplus1 = self.Cj
        self.Djplus1 = self.Dj

        self.Fpos = \
            np.diag(\
                np.exp( + 1j * gamma1 * self.Grating.LayerThicknesses[LayerIndex] * jmath.secd(self.Grating.SlantAngle_deg)))
        self.Fneg = \
            np.diag(\
                np.exp( - 1j * gamma2 * self.Grating.LayerThicknesses[LayerIndex] * jmath.secd(self.Grating.SlantAngle_deg)))

    def ConnectToInternalRegion(self):
        self.Aj = self.I2
        self.Bj = self.I2


        my_A = - np.diag(self.kx_m_local * self.IncidentLight.ky_local_input /(self.IncidentLight.k0 * self.kz_m_reflection))
        my_B = - np.diag((self.IncidentLight.ky_local_input**2 + self.kz_m_reflection**2) / (self.IncidentLight.k0*self.kz_m_reflection))
        my_C = + np.diag((self.kx_m_local**2 + self.kz_m_reflection ** 2) / (self.IncidentLight.k0 * self.kz_m_reflection))
        my_D = + np.diag(self.kx_m_local * self.IncidentLight.ky_local_input/(self.IncidentLight.k0 * self.kz_m_reflection))

        self.Cj = jmath.ConcatABCD(my_A,my_B,my_C,my_D)
        self.Dj = -self.Cj

        ### Update Z
        my_A = self.Ajplus1 + jmath.MatMulRow([self.Bjplus1,self.Fneg,self.Sdu,self.Fpos])
        my_B = -self.Bj
        my_C = self.Cjplus1 + jmath.MatMulRow([self.Djplus1,self.Fneg,self.Sdu,self.Fpos])
        my_D = -self.Dj
        my_denom = jmath.ConcatABCD(my_A,my_B,my_C,my_D)
        Z = jmath.MatDivRight(self.I4,my_denom)

        ### Update S-matrix
        self.Sdu = \
            + np.matmul(jmath.submat(Z,self.DiffOrderIndices2,self.DiffOrderIndices1),self.Aj) \
            + np.matmul(jmath.submat(Z,self.DiffOrderIndices2,self.DiffOrderIndices2),self.Cj)
        self.Sud = \
            + self.Sud \
            - jmath.MatMulRow([\
                self.Suu,\
                self.Fpos,\
                + np.matmul(jmath.submat(Z,self.DiffOrderIndices1,self.DiffOrderIndices1),self.Bjplus1) \
                + np.matmul(jmath.submat(Z,self.DiffOrderIndices1,self.DiffOrderIndices2),self.Djplus1),\
                self.Fneg,\
                self.Sdd])
        self.Suu = jmath.MatMulRow([\
            self.Suu,\
            self.Fpos,\
            (
                + np.matmul(jmath.submat(Z,self.DiffOrderIndices1,self.DiffOrderIndices1),self.Aj) \
                + np.matmul(jmath.submat(Z,self.DiffOrderIndices1,self.DiffOrderIndices2),self.Cj)
            )])
        self.Sdd = - jmath.MatMulRow([\
                + np.matmul(jmath.submat(Z,self.DiffOrderIndices2,self.DiffOrderIndices1),self.Bjplus1) \
                + np.matmul(jmath.submat(Z,self.DiffOrderIndices2,self.DiffOrderIndices2),self.Djplus1),\
            self.Fneg,\
            self.Sdd])

    def CalculateEfficiencies(self):
        SlantAnglePhaseTerm = \
            np.exp(-1j*np.sum(self.Grating.LayerThicknesses) * jmath.tand(self.Grating.SlantAngle_deg) * self.kx_m_local)

        Ez = \
            -(\
                + self.IncidentLight.sx_local * self.IncidentLight.Ex_local \
                + self.IncidentLight.sy_local * self.IncidentLight.Ey_local) \
            / self.IncidentLight.sz_local


        E_ref_ampl = \
            + self.Sdu[:,self.InputModeIndex] * self.IncidentLight.Ex_local \
            + self.Sdu[:,self.InputModeIndex+self.NumberOfDiffOrders] * self.IncidentLight.Ey_local
        E_trn_ampl = \
            + self.Suu[:,self.InputModeIndex] * self.IncidentLight.Ex_local \
            + self.Suu[:,self.InputModeIndex+self.NumberOfDiffOrders] * self.IncidentLight.Ey_local

        Ex_reflection_global = \
            + E_ref_ampl[0:self.NumberOfDiffOrders] * jmath.cosd(self.AzimuthAngle_deg) \
            - E_ref_ampl[self.NumberOfDiffOrders:] * jmath.sind(self.AzimuthAngle_deg)
        Ey_reflection_global = \
            + E_ref_ampl[0:self.NumberOfDiffOrders] * jmath.sind(self.AzimuthAngle_deg) \
            + E_ref_ampl[self.NumberOfDiffOrders:] * jmath.cosd(self.AzimuthAngle_deg)
        Ex_transmission_global = \
            + E_trn_ampl[0:self.NumberOfDiffOrders] * jmath.cosd(self.AzimuthAngle_deg) \
            - E_trn_ampl[self.NumberOfDiffOrders:] * jmath.sind(self.AzimuthAngle_deg)
        Ey_transmission_global = \
            + E_trn_ampl[0:self.NumberOfDiffOrders] * jmath.sind(self.AzimuthAngle_deg) \
            + E_trn_ampl[self.NumberOfDiffOrders:] * jmath.cosd(self.AzimuthAngle_deg)


        self.kx_global = self.kx_m_local * jmath.cosd(self.AzimuthAngle_deg) - self.IncidentLight.ky_local_input * jmath.sind(self.AzimuthAngle_deg)
        self.ky_global = self.kx_m_local * jmath.sind(self.AzimuthAngle_deg) + self.IncidentLight.ky_local_input * jmath.cosd(self.AzimuthAngle_deg)

        self.nx_global_output = self.kx_global / 2 / np.pi * self.IncidentLight.lambda0
        self.ny_global_output = self.ky_global / 2 / np.pi * self.IncidentLight.lambda0
        self.kz_reflection = - self.kz_m_reflection
        self.kz_transmission = + self.kz_m_transmission

        Ez_transmission_global = \
            -( self.kx_global * Ex_transmission_global + self.ky_global * Ey_transmission_global) / self.kz_m_transmission
        Ez_reflection_global = \
            +(self.kx_global*Ex_reflection_global + self.ky_global * Ey_reflection_global)/self.kz_m_reflection

        Ex_reflection_global = jmath.reshape_to_vect(Ex_reflection_global)
        Ey_reflection_global = jmath.reshape_to_vect(Ey_reflection_global)
        Ez_reflection_global = jmath.reshape_to_vect(Ez_reflection_global)
        Ex_transmission_global = jmath.reshape_to_vect(Ex_transmission_global)
        Ey_transmission_global = jmath.reshape_to_vect(Ey_transmission_global)
        Ez_transmission_global = jmath.reshape_to_vect(Ez_transmission_global)


        SlantAnglePhaseMatrix = (np.tile(SlantAnglePhaseTerm, (3, 1)))

        E_reflection = np.concatenate([Ex_reflection_global,Ey_reflection_global,Ez_reflection_global],axis=1)
        E_transmission = np.matmul(\
            np.concatenate([Ex_transmission_global,Ey_transmission_global,Ez_transmission_global],axis=1),\
            SlantAnglePhaseMatrix)

        self.Eff_reflection = \
                              jmath.reshape_to_vect((self.kz_m_reflection/self.IncidentLight.kz_local_input).real) * \
                              ( abs(Ex_reflection_global)**2 + abs(Ey_reflection_global)**2 + abs(Ez_reflection_global)**2) / \
                              (abs(self.IncidentLight.Ex_local)**2 + abs(self.IncidentLight.Ey_local)**2 + abs(Ez)**2)
        self.Eff_transmission = \
            jmath.reshape_to_vect((self.kz_m_transmission / self.IncidentLight.kz_local_input).real) * \
            (abs(Ex_transmission_global) ** 2 + abs(Ey_transmission_global) ** 2 + abs(Ez_transmission_global) ** 2) / \
            (abs(self.IncidentLight.Ex_local) ** 2 + abs(self.IncidentLight.Ey_local) ** 2 + abs(Ez) ** 2)


class Grating:
    def __init__(self,\
                 FMMobj,\
                 LayerThicknesses,\
                 layer_materials,\
                 MaterialBoundaries,\
                 UnitCellWidth, \
                 AzimuthAngle_deg, \
                 SlantAngle_deg \
                 ):
        self.LayerThicknesses = LayerThicknesses
        self.layer_materials = layer_materials
        self.MaterialBoundaries = MaterialBoundaries
        self.UnitCellWidth = UnitCellWidth

        self.SlantAngle_deg = SlantAngle_deg
        self.AzimuthAngle_deg = AzimuthAngle_deg

        self.FMMobj = FMMobj

    def DrawGrating(self):

        # fig = plt.figure()
        # ax = fig.gca()
        #
        # fig, ax = plt.subplots()

        GratHeight = sum(self.LayerThicknesses)
        nlayers = len(self.LayerThicknesses)
        GratWidth = self.UnitCellWidth

        # Draw substrate

        fig, ax = plt.subplots()

        colors = np.array([], ndmin=1)
        print(colors.shape)
        patches = []

        # GratHeight = 0.2
        ## Add substrate (n_reflection)
        xs = np.array([0,1,1,0,0],ndmin=2) * self.UnitCellWidth
        ys = np.array([1,1,0,0,1],ndmin=2) * GratHeight
        xy = np.concatenate((xs,ys),axis=0).transpose()
        polygon = Polygon(xy,True)
        patches.append(polygon)
        colors = np.concatenate((colors, np.array(self.FMMobj.n_reflection,ndmin=1)),axis=0)

        ## Add transmission region
        xs = np.array([0,1,1,0,0],ndmin=2) * self.UnitCellWidth
        ys = np.array([1,1,0,0,1],ndmin=2) * GratHeight + GratHeight*2
        xy = np.concatenate((xs,ys),axis=0).transpose()
        polygon = Polygon(xy,True)
        patches.append(polygon)
        colors = np.concatenate((colors, np.array(self.FMMobj.n_transmission,ndmin=1)),axis=0)


        print(self.MaterialBoundaries[1])
        # Layer = 0
        for Layer in range(0,len(self.MaterialBoundaries)):
            for aa in range (0,len(self.MaterialBoundaries[Layer])+1):
                print(self.LayerThicknesses)

                if(Layer==0): # If first layer
                    ys = np.array([1,1,0,0,1],ndmin=2) * self.LayerThicknesses[Layer] + GratHeight
                    print("First Layer")
                elif(Layer==len(self.MaterialBoundaries)-1): # If last layer
                    ys = np.array([1,1,0,0,1],ndmin=2) * self.LayerThicknesses[Layer] + GratHeight + \
                        sum(self.LayerThicknesses[0:Layer])
                    print("Last Layer")
                    print(sum(self.LayerThicknesses[0:Layer]))
                    print(ys)
                else: # If intermediate layer
                    ys = np.array([1,1,0,0,1],ndmin=2) * self.LayerThicknesses[Layer] + GratHeight + \
                        sum(self.LayerThicknesses[0:Layer-1])
                    print("Intermediate Layer")

                print(self.SlantAngle_deg)
                print(np.tan(self.SlantAngle_deg*np.pi/180))


                if(aa==0):
                    xs = np.array([0,1,1,0,0],ndmin=2) * (self.MaterialBoundaries[Layer][0]*self.UnitCellWidth)
                elif(aa==len(self.MaterialBoundaries[Layer])):
                    xs = np.array([0,1,1,0,0],ndmin=2) * ((1 - self.MaterialBoundaries[Layer][-1])*self.UnitCellWidth)+ \
                        self.MaterialBoundaries[Layer][-1]*self.UnitCellWidth
                else:
                    xs = np.array([0,1,1,0,0],ndmin=2) * (self.MaterialBoundaries[Layer][aa]*self.UnitCellWidth - self.MaterialBoundaries[Layer][aa-1]*self.UnitCellWidth) + \
                        self.MaterialBoundaries[Layer][aa-1]*self.UnitCellWidth

                # Apply x offset to bottom of grating due to slant angle
                xs = xs + np.tan(self.SlantAngle_deg * np.pi /180) * sum(self.LayerThicknesses[0:Layer])
                # Apply additional x offset to top of layer
                xs = xs + np.array([1,1,0,0,1],ndmin=2) * np.tan(self.SlantAngle_deg * np.pi / 180) * self.LayerThicknesses[Layer]
                print(xs)
                # ys = ys - 0.3
                xy = np.concatenate((xs, ys), axis=0).transpose()
                polygon = Polygon(xy, True)
                patches.append(polygon)
                colors = np.concatenate((colors, np.array(self.layer_materials[Layer][aa],ndmin=1)))

                xs = xs - self.UnitCellWidth
                xy = np.concatenate((xs, ys), axis=0).transpose()
                polygon = Polygon(xy, True)
                patches.append(polygon)
                colors = np.concatenate((colors, np.array(self.layer_materials[Layer][aa],ndmin=1)))

                xs = xs + 2 * self.UnitCellWidth
                xy = np.concatenate((xs, ys), axis=0).transpose()
                polygon = Polygon(xy, True)
                patches.append(polygon)
                colors = np.concatenate((colors, np.array(self.layer_materials[Layer][aa],ndmin=1)))

        xmin = +0.0
        xmax = self.UnitCellWidth
        ymin = 0
        ymax = +GratHeight*3
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        # colors = 10 * np.random.rand(len(patches))


        p = PatchCollection(patches, alpha=0.4)
        p.set_array(np.array(colors))
        ax.add_collection(p)
        fig.colorbar(p, ax=ax)

        # mpld3.show()
        # zz = mpld3.fig_to_html(fig)
        # print(zz)


        plt.show()


class InputRay:
    def __init__(\
            self,\
            lambda0,\
            nx_global_input,\
            ny_global_input,\
            Ex_global_input,\
            Ey_global_input,\
            n_reflection,\
            AzimuthAngle_deg):

        self.lambda0 = lambda0
        self.nx_global_input = nx_global_input
        self.ny_global_input = ny_global_input

        self.nz_global_input = 0
        self.Ex_global_input = Ex_global_input
        self.Ey_global_input = Ey_global_input

        self.n_reflection = n_reflection

        self.UpdateInputCosines()
        self.RotateInputsToLocalCoordinates(AzimuthAngle_deg)
        self.UpdateInputKVectors()
    def UpdateInputCosines(self):

        if self.nx_global_input**2 + self.ny_global_input**2 > self.n_reflection**2:
            print("Invalid input direction detected")
        else:
            self.nz_global_input = np.sqrt(self.n_reflection**2 - self.nx_global_input**2 - self.ny_global_input**2)
            self.sx_global_input = self.nx_global_input / self.n_reflection
            self.sy_global_input = self.ny_global_input/self.n_reflection
            self.sz_global_input = self.nz_global_input/self.n_reflection

    def RotateInputsToLocalCoordinates(self,AzimuthAngle_deg):
        self.sx_local = \
            + self.sx_global_input * jmath.cosd(AzimuthAngle_deg) \
            + self.sy_global_input * jmath.sind(AzimuthAngle_deg)
        self.sy_local = \
            - self.sx_global_input * jmath.sind(AzimuthAngle_deg) \
            + self.sy_global_input * jmath.cosd(AzimuthAngle_deg)
        self.sz_local = self.sz_global_input

        self.Ex_local = \
            + self.Ex_global_input * jmath.cosd(AzimuthAngle_deg) \
            + self.Ey_global_input * jmath.sind(AzimuthAngle_deg)

        self.Ey_local = \
            - self.Ex_global_input * jmath.sind(AzimuthAngle_deg) \
            + self.Ey_global_input * jmath.cosd(AzimuthAngle_deg)

    def UpdateInputKVectors(self):
        self.k0 = 2 * np.pi / self.lambda0
        self.k_reflection = self.k0 * self.n_reflection

        self.kx_local_input = self.sx_local * self.k_reflection
        self.ky_local_input = self.sy_local * self.k_reflection
        self.kz_local_input = self.sz_local * self.k_reflection



class ThinFilmObj(jrcwa):
    def __init__(self):
        print("a")



