# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 09:06:14 2021

@author: Veronica Reynolds

Perform calculations of the strain-dependent behavior of elastic transistors and simple circuits.

Classes:

    transistor
    inverter

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2

class transistor():

    """
    Define a stretchable transistor object.

    Instance Variables:
        flavor (string): Flavor of transistor: 'TFT', 'EDLT', or 'OECT'
        Ttype (string): Type of transistor: 'p-type' or 'n-type'
        W (float): Width of the semiconducting channel
            Arbitrary dimensional units, as long as it matches L
        L (float): Length of the semiconducting channel
            Arbitrary dimensional units, as long as it matches W
        d (float): Thickness of the semiconducting layer; only relevant for OECTs [m]
        C (float): For TFT and EDLT this is gate capacitance C_G [F/m^2]
                   For OECT this is volumetric capacitance [C*, F/m^3]
        mu (float): Mobility [m^2/Vs]
        V_T (float): Threshold voltage [V]
        V_DD (float): Supply voltage [V]
        V_resolution (int): Resolution of the voltage sweep
        deformMode (string): Deformation axis options:
            (1) uniaxial along channel length 'uniaxial-L'
            (2) uniaxial along channel width 'uniaxial-W'
            (3) biaxial 'biaxial-WL'
        er (List[float]): List of extension ratios over which to calculate
            Note: This must include the undeformed state (er = 1)

    Functions:
        calculateStrainDependence(self)
        calculateI_SD(self)
        plotIVvsDeformation(self, er_plot)
        calculateRelativeI_SD(self)
        plotRelativeI_SD(self)

    """

    def __init__(self, flavor, Ttype, W, L, d, C, mu, V_T, V_DD, V_resolution, deformMode, er):

        if flavor != 'TFT' and flavor != 'EDLT' and flavor !='OECT':
            print('Not a valid flavor of transistor (options: TFT, EDLT, or OECT).')
        self.flavor = flavor

        if Ttype != 'n' and Ttype != 'p':
            print('Not a valid type of transistor (options: n or p).')
        self.Ttype = Ttype

        self.W = W
        self.L = L
        self.d = d
        self.C = C
        self.mu = mu
        self.V_T = V_T

        self.V_DD = V_DD
        self.V_resolution = V_resolution

        self.V_range = np.linspace(0,V_DD,V_resolution)

        if self.Ttype == 'n':
            self.V_G = self.V_range
            self.V_SD = self.V_range
            self.I_SD_maxidx = len(self.V_G)-1
        elif self.Ttype == 'p':
            self.V_G = self.V_range-V_DD
            self.V_SD = self.V_range-V_DD
            self.I_SD_maxidx = 0

        self.V_SD_satidx = np.abs(np.abs(self.V_SD)-np.abs(self.V_DD)).argmin()

        if deformMode != 'uniaxial-L' and deformMode != 'uniaxial-W' and deformMode != 'biaxial-WL':
            print('Not a valid deformation mode (options: uniaxial-L, uniaxial-W, biaxial-WL).')
        self.deformMode = deformMode

        if 1 not in er:
            print('The range of extension ratios to be modeled must include 1 (the undeformed state).')
        self.er = np.asarray(er)

        self.er_1_idx = np.where(self.er==1)

        # Define the extension ratios in all three dimensions based on deformation mode.
        if self.deformMode == 'uniaxial-L':

            self.erL = self.er
            self.erW = 1/(self.erL**(1/2))
            self.ert = 1/(self.erL**(1/2))

        elif self.deformMode == 'uniaxial-W':

            self.erW = self.er
            self.erL = 1/(self.erW**(1/2))
            self.ert = 1/(self.erW**(1/2))

        elif self.deformMode == 'biaxial-WL':

            self.erL = self.er
            self.erW = self.er
            self.ert = np.zeros((len(self.erW), len(self.erL)))
            for i in range(len(self.erW)):
                for j in range(len(self.erL)):
                    self.ert[i,j] = 1/(self.erL[j]*self.erW[i])

    def calculateStrainDependence(self):
        """Define the constant beta, the strain-dependent V_T, and the
        strain-dependent C based on transistor type.
        """

        if self.flavor == 'TFT':
            self.beta = (self.W/self.L)*self.mu*self.C
            self.V_T_er = self.V_T*self.ert
            self.C_er = self.C*(self.ert**(-1))
            if self.deformMode == 'uniaxial-L' or self.deformMode == 'uniaxial-W':
                self.geo_er = (self.W*self.erW)/(self.L*self.erL)
            elif self.deformMode == 'biaxial-WL':
                self.geo_er = np.zeros((len(self.erW), len(self.erL)))
                for i in range(len(self.erW)):
                    for j in range(len(self.erL)):
                        self.geo_er[i,j] = (self.W*self.erW[i])/(self.L*self.erL[j])

        elif self.flavor == 'EDLT':
            self.beta = (self.W/self.L)*self.mu*self.C
            self.V_T_er = self.V_T*np.ones(np.shape(self.ert))
            self.C_er = self.C*np.ones(np.shape(self.ert))
            if self.deformMode == 'uniaxial-L' or self.deformMode == 'uniaxial-W':
                self.geo_er = (self.W*self.erW)/(self.L*self.erL)
            elif self.deformMode == 'biaxial-WL':
                self.geo_er = np.zeros((len(self.erW), len(self.erL)))
                for i in range(len(self.erW)):
                    for j in range(len(self.erL)):
                        self.geo_er[i,j] = (self.W*self.erW[i])/(self.L*self.erL[j])

        elif self.flavor == 'OECT':
            self.beta = (self.W/self.L)*self.d*self.mu*self.C
            self.V_T_er = self.V_T*np.ones(np.shape(self.ert))
            self.C_er = self.C*np.ones(np.shape(self.ert))
            if self.deformMode == 'uniaxial-L' or self.deformMode == 'uniaxial-W':
                self.geo_er = ((self.W*self.erW)/(self.L*self.erL))*(self.d*self.ert)
            elif self.deformMode == 'biaxial-WL':
                self.geo_er = np.zeros((len(self.erW), len(self.erL)))
                for i in range(len(self.erW)):
                    for j in range(len(self.erL)):
                        self.geo_er[i,j] = ((self.W*self.erW[i])/(self.L*self.erL[j]))*(self.d*self.ert[i,j])

    def calculateI_SD(self):
        """Calculate the strain-dependent source-drain current I_SD."""

        if self.deformMode == 'uniaxial-L' or self.deformMode ==  'uniaxial-W':

            self.I_SD = np.zeros((np.size(self.V_SD), np.size(self.V_G), np.size(self.erL)))

            for i in range(np.size(self.erL)):
                for j in range(np.size(self.V_G)):
                    for k in range(np.size(self.V_SD)):
                        if abs(self.V_G[j])<abs(self.V_T_er[i]):
                            self.I_SD[k,j,i]=0
                        elif abs(self.V_SD[k])<abs(self.V_G[j]-self.V_T_er[i]): # linear regime
                            self.I_SD[k,j,i]=self.geo_er[i]*self.C_er[i]*self.mu*(((self.V_G[j]-self.V_T_er[i])*self.V_SD[k])-((self.V_SD[k]**2)/2))
                        elif abs(self.V_SD[k])>=abs(self.V_G[j]-self.V_T_er[i]): # saturation regime
                            self.I_SD[k,j,i]=self.geo_er[i]*self.C_er[i]*self.mu*(((self.V_G[j]-self.V_T_er[i])**2)/2)

        elif self.deformMode == 'biaxial-WL':

            self.I_SD = np.zeros((np.size(self.V_SD), np.size(self.V_G), np.size(self.erW), np.size(self.erL)))

            for h in range(np.size(self.erL)):
                for i in range(np.size(self.erW)):
                    for j in range(np.size(self.V_G)):
                        for k in range(np.size(self.V_SD)):
                            if abs(self.V_G[j])<abs(self.V_T_er[i,h]):
                                self.I_SD[k,j,i,h]=0
                            elif abs(self.V_SD[k])<abs(self.V_G[j]-self.V_T_er[i,h]): # linear regime
                                self.I_SD[k,j,i,h]=self.geo_er[i,h]*self.C_er[i,h]*self.mu*(((self.V_G[j]-self.V_T_er[i,h])*self.V_SD[k])-((self.V_SD[k]**2)/2))
                            elif abs(self.V_SD[k])>=abs(self.V_G[j]-self.V_T_er[i,h]): # saturation regime
                                self.I_SD[k,j,i,h]=self.geo_er[i,h]*self.C_er[i,h]*self.mu*(((self.V_G[j]-self.V_T_er[i,h])**2)/2)

    def plotIVvsDeformation(self, er_plot):
        """Plot a series of I-V for a single transistor vs. deformation.

        Inputs:
            er_plot (List[float]): list of the extension ratios to plot.
            Note: These values should be contained in your extension ratio sweep.
        """

        if self.deformMode == 'uniaxial-L' or self.deformMode ==  'uniaxial-W':

            idx_er = np.zeros((np.size(er_plot)))

            for i in range((np.size(er_plot))):
                idx_er[i] = (np.abs(self.er - er_plot[i])).argmin()

            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
            ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
            ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
            ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')

            ax.set_xlabel('$V_\mathrm{SD}$ (V)', labelpad=10)
            ax.set_ylabel('$I_\mathrm{SD}$ (A)', labelpad=10)

            cm = mpl.cm.get_cmap('viridis')

            for i in range((np.size(er_plot,0))):
                ax.plot(self.V_SD, self.I_SD[:,self.I_SD_maxidx,int(idx_er[i])],
                        linewidth=2,
                        color=cm(1.*i/np.size(er_plot)),
                        label=('$\lambda$ = '+str(er_plot[i])))

            plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')

            plt.show()

            #plt.savefig('Final_Plot.png', dpi=300, transparent=False, bbox_inches='tight')

        else:

            print('Currently unsupported.')

    def calculateRelativeI_SD(self):
        """Calculate the relative source-drain current (I_SD/I_SD(er=1)) in the saturation regime.
        The value of V_G that gives the highest magnitude I_SD is used.
        """

        if self.deformMode == 'uniaxial-L' or self.deformMode ==  'uniaxial-W':

            self.I_SDrel = np.zeros(np.size(self.erL))
            self.I_SD_undeformed = self.I_SD[self.V_SD_satidx, self.I_SD_maxidx, self.er_1_idx]

            for i in range(np.size(self.erL)):
                self.I_SDrel[i] = self.I_SD[self.V_SD_satidx, self.I_SD_maxidx,i]/self.I_SD_undeformed

        elif self.deformMode == 'biaxial-WL':

            self.I_SDrel = np.zeros((np.size(self.erW), np.size(self.erL)))
            self.I_SD_undeformed = self.I_SD[self.V_SD_satidx, self.I_SD_maxidx, self.er_1_idx, self.er_1_idx]

            for i in range(np.size(self.erL)):
                for j in range(np.size(self.erW)):
                    self.I_SDrel[j,i] = self.I_SD[self.V_SD_satidx, self.I_SD_maxidx, j, i]/self.I_SD_undeformed

    def plotRelativeI_SD(self):

        """Plot relative source-drain current in the saturation regime vs. extension ratio."""

        if self.deformMode == 'uniaxial-L' or self.deformMode ==  'uniaxial-W':

            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
            ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
            ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
            ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')

            ax.set_xlabel('Extension Ratio, $\lambda$', labelpad=10)
            ax.set_ylabel('Relative Current, $I_\mathrm{SD}/I_\mathrm{SD}^\mathrm{initial}$', labelpad=10)

            ax.plot(self.er, self.I_SDrel, linewidth=2)

            plt.show()

        elif self.deformMode == 'biaxial-WL':

            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_axes([0, 0, 1, 1])

            ax.set_xlabel('$\lambda_L$', labelpad=10)
            ax.set_ylabel('$\lambda_W$', labelpad=10)

            im = ax.imshow(self.I_SDrel, interpolation='none', cmap=mpl.cm.get_cmap('viridis'),
                   origin='lower', extent=[np.min(self.er), np.max(self.er), np.min(self.er), np.max(self.er)])

            fig.colorbar(im, ax=ax)

            plt.show()

class inverter():

    """
    Define a stretchable inverter object.

    Instance Variables:
        ntype (transistor): The n-type transistor in the inverter
        ptype (transistor): The p-type transistor in the inverter

    Functions:
        buildVTC(self)
        plotLoadCurves(self, V_in_LCplot, er_LCplot)
        plotLoadCurves_alternative(self, V_in_LCplot, er_LCplot)
        plotVTC(self, er_plot)
        plotVTCeye(self, er_plot)

    """

    def __init__(self, ntype, ptype):

        self.ntype = ntype
        self.ptype = ptype

        if np.array_equal(ntype.er, ptype.er) == False:
            print('The n-type and p-type transistors must have identical arrays of extension ratios (er).')

        if ntype.V_DD != ptype.V_DD:
            print('The n-type and p-type transistors must have identical supply voltages (V_DD).')

    def buildVTC(self):

        """
        Build the voltage transfer curve (VTC) by finding where the n- and p-type load curves cross.
        The accuracy of this calculation depends on your voltage scan resolution, V_resolution.
        """

        if (self.ntype.deformMode == 'uniaxial-L' or self.ntype.deformMode ==  'uniaxial-W') and (self.ptype.deformMode == 'uniaxial-L' or self.ptype.deformMode ==  'uniaxial-W'):
            self.V_out_cross = np.zeros((np.size(self.ntype.V_G,0),np.size(self.ntype.er,0)))
            for i in range(np.size(self.ntype.er)):
                for j in range (np.size(self.ntype.V_G)):
                    for k in range (np.size(self.ntype.V_SD)):
                        if self.ntype.I_SD[k,j,i]>=self.ptype.I_SD[k,j,i]:
                            self.V_out_cross[j,i] = self.ntype.V_SD[k]
                            break

        elif (self.ntype.deformMode == 'biaxial-WL') and (self.ptype.deformMode == 'biaxial-WL'):
            self.V_out_cross = np.zeros((np.size(self.ntype.V_G),np.size(self.ntype.er),np.size(self.ntype.er)))
            for h in range(np.size(self.ntype.er)):
                for i in range(np.size(self.ntype.er)):
                    for j in range (np.size(self.ntype.V_G)):
                        for k in range (np.size(self.ntype.V_SD,0)):
                            if self.ntype.I_SD[k,j,i,h]>=self.ptype.I_SD[k,j,i,h]:
                                self.V_out_cross[j,i,h] = self.ntype.V_SD[k]
                                break

        else:
            print('This deformation scenario is unsupported.')

    def plotLoadCurves(self, V_in_LCplot, er_LCplot):
        """Plot a series of load curves.

        Inputs:
            V_in_LCplot (List[float]): list of the input voltages to plot
                Note: These values should be contained in your voltage sweep.
        """

        if (self.ntype.deformMode == 'uniaxial-L' or self.ntype.deformMode ==  'uniaxial-W') and (self.ptype.deformMode == 'uniaxial-L' or self.ptype.deformMode ==  'uniaxial-W'):

            idx_er = (np.abs(self.ntype.er - er_LCplot)).argmin()
            idx_n = np.zeros((np.size(V_in_LCplot)))
            idx_p = np.zeros((np.size(V_in_LCplot)))

            for i in range((np.size(V_in_LCplot))):
                idx_n[i] = (np.abs(self.ntype.V_G - V_in_LCplot[i])).argmin()

            for i in range((np.size(V_in_LCplot))):
                idx_p[i] = (np.abs(self.ptype.V_G - (V_in_LCplot[i]-self.ptype.V_DD))).argmin()

            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
            ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
            ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
            ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')

            ax.set_xlim(0, self.ntype.V_DD)
            ax.set_ylim(0, 1.05*np.max(self.ntype.I_SD[:,:,idx_er]))

            ax.set_xlabel('$V_\mathrm{SD}$ (V)', labelpad=10)
            ax.set_ylabel('$I_\mathrm{SD}$ (A)', labelpad=10)

            cm = mpl.cm.get_cmap('viridis')

            for i in range((np.size(V_in_LCplot))):
                ax.plot(self.ntype.V_SD, self.ntype.I_SD[:,int(idx_n[i]),idx_er],
                        linewidth=2, color=cm(1.*i/np.size(V_in_LCplot)),
                        label=('$V_\mathrm{in}$ = '+str(V_in_LCplot[i])+' V'))
                ax.plot(self.ntype.V_SD, self.ptype.I_SD[:,int(idx_p[i]),idx_er],
                        linewidth=2, color=cm(1.*i/np.size(V_in_LCplot)), linestyle='--')

            plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')

            plt.show()

            #plt.savefig('Final_Plot.png', dpi=300, transparent=False, bbox_inches='tight')

        else:

            print('Currently unsupported.')

    def plotLoadCurves_alternative(self, V_in_LCplot, er_LCplot):
        """Plot a series of load curves, alternative coordinate system.

        Inputs:
            V_in_LCplot (List[float]): list of the input voltages to plot
                Note: These values should be contained in your voltage sweep.
            er_LCplot (float): extension ratio to plot
                Note: This value should be contained in your extension ratio sweep.
        """

        if (self.ntype.deformMode == 'uniaxial-L' or self.ntype.deformMode ==  'uniaxial-W') and (self.ptype.deformMode == 'uniaxial-L' or self.ptype.deformMode ==  'uniaxial-W'):

            idx_er = (np.abs(self.ntype.er - er_LCplot)).argmin()
            idx_n = np.zeros((np.size(V_in_LCplot)))
            idx_p = np.zeros((np.size(V_in_LCplot)))

            for i in range((np.size(V_in_LCplot))):
                idx_n[i] = (np.abs(self.ntype.V_G - V_in_LCplot[i])).argmin()

            for i in range((np.size(V_in_LCplot))):
                idx_p[i] = (np.abs(self.ptype.V_G - (V_in_LCplot[i]-self.ptype.V_DD))).argmin()

            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
            ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
            ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
            ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')

            ax.set_xlabel('$V_\mathrm{SD}$ (V)', labelpad=10)
            ax.set_ylabel('$I_\mathrm{SD}$ (A)', labelpad=10)

            cm = mpl.cm.get_cmap('viridis')

            for i in range((np.size(V_in_LCplot))):
                ax.plot(self.ntype.V_SD, self.ntype.I_SD[:,int(idx_n[i]),idx_er],
                        linewidth=2, color=cm(1.*i/np.size(V_in_LCplot)),
                        label=('$V_\mathrm{in}$ = '+str(V_in_LCplot[i])+' V'))
                ax.plot(self.ptype.V_SD, self.ptype.I_SD[:,int(idx_p[i]),idx_er],
                        linewidth=2, color=cm(1.*i/np.size(V_in_LCplot)), linestyle='--')

            plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')

            plt.show()

        else:

            print('Currently unsupported.')

    def plotVTC(self, er_plot):
        """Plot the voltage transfer curve of the inverter vs. deformation.

        Inputs:
            er_plot (List[float]): list of the extension ratios to plot
                Note: These values should be contained in your extension ratio sweep.
        """

        if (self.ntype.deformMode == 'uniaxial-L' or self.ntype.deformMode ==  'uniaxial-W') and (self.ptype.deformMode == 'uniaxial-L' or self.ptype.deformMode ==  'uniaxial-W'):

            idx_er = np.zeros((np.size(er_plot)))

            for i in range((np.size(er_plot))):
                idx_er[i] = (np.abs(self.ntype.er - er_plot[i])).argmin()

            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
            ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
            ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
            ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')

            ax.set_xlim(np.min(self.ntype.V_G), np.max(self.ntype.V_G))
            ax.set_ylim((np.min(self.V_out_cross)-0.05*np.max(self.V_out_cross)), 1.05*np.max(self.V_out_cross))

            ax.set_xlabel('$V_\mathrm{in}$ (V)', labelpad=10)
            ax.set_ylabel('$V_\mathrm{out}$ (V)', labelpad=10)

            cm = mpl.cm.get_cmap('viridis')

            for i in range((np.size(er_plot,0))):
                ax.plot(self.ntype.V_G, self.V_out_cross[:,int(idx_er[i])],
                        linewidth=2, color=cm(1.*i/np.size(er_plot)),
                        label=('$\lambda$ = '+str(er_plot[i])))

            plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')

            plt.show()

        else:

            print('Currently unsupported.')

    def plotVTCeye(self, er_plot):
        """Plot the voltage transfer curve of the inverter vs. deformation as an eye diagram.
        The VTC is overlaid with itself, flipped and rotated.

        Inputs:
            er_plot (List[float]): list of the extension ratios to plot
                Note: These values should be contained in your extension ratio sweep.
        """

        if (self.ntype.deformMode == 'uniaxial-L' or self.ntype.deformMode ==  'uniaxial-W') and (self.ptype.deformMode == 'uniaxial-L' or self.ptype.deformMode ==  'uniaxial-W'):

            eye_x = np.zeros((np.size(self.V_out_cross,0),np.size(self.ntype.er,0)))
            eye_y = np.zeros((np.size(self.V_out_cross,0),np.size(self.ntype.er,0)))

            for i in range(np.size(self.ntype.er,0)):
                eye_x[:,i] = -self.V_out_cross[:,i]+self.ntype.V_DD
                eye_y[:,i] = np.flip(self.ntype.V_G)

            idx_er = np.zeros((np.size(er_plot)))

            for i in range((np.size(er_plot))):
                idx_er[i] = (np.abs(self.ntype.er - er_plot[i])).argmin()

            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
            ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
            ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
            ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')

            ax.set_xlim((np.min(self.ntype.V_G)-0.05*np.max(self.ntype.V_G)), 1.05*np.max(self.ntype.V_G))
            ax.set_ylim((np.min(self.V_out_cross)-0.05*np.max(self.V_out_cross)), 1.05*np.max(self.V_out_cross))

            ax.set_xlabel('$V_\mathrm{in}$ (V)', labelpad=10)
            ax.set_ylabel('$V_\mathrm{out}$ (V)', labelpad=10)

            cm = mpl.cm.get_cmap('viridis')

            for i in range((np.size(er_plot,0))):
                ax.plot(self.ntype.V_G, self.V_out_cross[:,int(idx_er[i])],
                        linewidth=2, color=cm(1.*i/np.size(er_plot)),
                        label=('$\lambda$ = '+str(er_plot[i])))
                ax.plot(eye_x[:,int(idx_er[i])], eye_y[:,int(idx_er[i])],
                        linewidth=2, color=cm(1.*i/np.size(er_plot)),linestyle='--')

            plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')

            plt.show()

        else:

            print('Currently unsupported.')
