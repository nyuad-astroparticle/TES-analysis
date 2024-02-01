# import libraries
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors
import pyperclip
import datetime
import cupy as cp
import seaborn as sns

from matplotlib.patches import ConnectionPatch
from matplotlib.colors import ListedColormap
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
from scipy import signal
from tqdm.notebook import tqdm
from sklearn.cluster import KMeans
from sklearn import manifold
from matplotlib.dates import DateFormatter


from .data_processing import Data_Processing

class PCA(Data_Processing):

    def __init__(self, traces_path, channel = 'C1'):
        super().__init__(traces_path, channel)
    
        self.X_compressed = None

        self.pyper = []
        self.reduced = []

        self.reduced_the_data = []
        self.X_compressed_red = None


    def plot_signal(self, voltage,fig=None,ax=None,order=2,mysize=1/50,**kwargs):
        '''
        Returns a Pyplot figure of the smoothed voltage signal as a function of
        event number

        Args:
        voltage (arr): voltage of a signal from the data

        Returns:
        fig,ax

        '''

        sos=signal.butter(order,mysize, output='sos')
        butterybiscuits=signal.sosfilt(sos,voltage)
        
        # When K-means plot is first made the subplot for the signal is initially undefined since no signal is selected to be plotted
        if fig is None: fig = plt.figure()
        if ax is None: ax  = fig.add_subplot(111)
        
        ax.plot(butterybiscuits,**kwargs)
        
        return fig,ax
    
    def values_of_interest(self,time,data,order,mysize):
        """
        These are functions used as dimentions in the PCA
        
        Args:
        voltage (arr): voltage of a signal from the data
        time (arr): time of a signal from the data
        Returns:
        Arbitraty combination of the functions defined below

        """
        sos=signal.butter(order,mysize, output='sos')
        butterybiscuits=signal.sosfilt(sos,data)
        
        # Finds maximum voltage amplitude value after applied smoothing
        peakheight      = np.min(butterybiscuits)
        # Finds index of the peakheight point
        peak            = np.argmin(butterybiscuits)
        # Calculates total area of the smoothed pulse 
        area            = np.sum(butterybiscuits[:-1]*(time[1:]-time[:-1]))
        # Finds time points where the signal amplitude is smaller than peakheight
        halfheights     = np.where(butterybiscuits-peakheight/2<0)[0]
        # Calculates full width at a half maxima (time)
        fwhm            = 0 if len(halfheights)<1 else time[halfheights[-1]]-time[halfheights[0]]
        # Noise signals usually have a lot of positive voltage values across the signal, this function checks how many points are above arbitrarily small positivce value (important that this function uses raw signal rather than smoothed to preserve fluctuations)
        noize           = len([x for x in data if x>0.00005])
        # Calculates median voltage of the signal after applied smoothing
        medheight       = np.median(butterybiscuits)
        # Calculates mean voltage of the signal after applied smoothing
        meanheight      = np.mean(butterybiscuits)
        # Calculates mean time of the signal after applied smoothing
        meantime        = np.mean(np.abs(butterybiscuits)*(time))/np.sum(np.abs(butterybiscuits[:-1])*(time[1:]-time[:-1]))
        # Calculates standard deviation of the signal after applied smoothing
        std             = np.std(butterybiscuits)
        # Calculates a gradient (derivative) of the signal after applied smoothing
        grad            = (butterybiscuits[1:]-butterybiscuits[:-1])/(time[1:]-time[:-1])
        # Finds mean value of the negative gradient
        grad_min        = np.mean(grad[np.where(grad<0)])
        # Finds mean value of the positive gradient
        grad_max        = np.mean(grad[np.where(grad>0)])
        # Correlates the signal with the scaled photon-like reference signal
        corr            = signal.correlate(butterybiscuits/np.min(butterybiscuits)*np.min(self.REFERENCE_VOLTAGE),self.REFERENCE_VOLTAGE)
        # Calculates cumulative area up to every point of the smoothed signal (arr)
        area_cumulative = np.cumsum(butterybiscuits[:-1]*(time[1:]-time[:-1]))
        # Calculates cumulative area as a percentage of total area
        cumt_percentage = area_cumulative/area
        # Define the suitable percentage of the area that can separate the signals the most
        percentage      = 0.97
        # Finds the time stamp of the point where the cumulative area reaches defined percentage of total area
        time_at_perc = time[np.argmax(cumt_percentage >= percentage)]
        # Measures how many points are exceeding the trigger value
        time_under_trig = len([x for x in butterybiscuits if x >= self.trigger])
        # Returns arbitrary composite functions to further improve separation in feature reduction
        return (time_under_trig, noize, area,area/std, peakheight, time_at_perc, meantime, np.sqrt(np.mean(grad**2)), grad_min, grad_max, np.max(corr))

    def feature_reduction(self, values_of_interest, a_data, order=2, mysize=1/30):
        """
        This function performs Principal component analysis on the selected functions together with feature reduction and signle value decomposition.

        Args:
        None: works with the data

        Returns:
        None: saves self.X_compressed for future uses

        """
        
        # Get the data for each of the pulses
        values      = cp.array([values_of_interest(*data[:2],order=order,mysize=mysize) for data in tqdm(a_data)])

        # Center data
        mu          = cp.tile(np.mean(values,axis=0),len(values)).reshape(values.shape)
        X_centered  = (values-mu)
        X_centered  = X_centered / cp.tile(np.max(X_centered,axis=0)-cp.min(X_centered,axis=0),len(X_centered)).reshape(X_centered.shape)

        # Single Value Decomposition
        u,s,vT      = cp.linalg.svd(X_centered)

        # plot the singlar values for the  D  matrix.
        # 1. Calculate the D matrix using s: D is s*s
        D       = cp.asnumpy(s*s)
        labels  = range(len(X_centered[0]))

        # 2. Set the fig size to (15,5)
        fig     = plt.figure(figsize=(10,5))
        ax      = fig.add_subplot(111)

        # 3. Add the line chart using plt.plot( ?? ,'bo-')
        ax.plot(labels,D,'bo-')

        # 3. Add proper tital, ticks, axis labels
        ax.set_title('Singular Values of Datapoints')
        ax.set_xlabel('Label for Linear Combination of (PCs)')
        ax.set_ylabel('Singular Value')
        ax.set_xticks(labels)
        # ax.set_yscale('log')

        # Obtaining our compressed data representation:
        # 1. Determine at least k singular values that are needed to represent the data set from the fig above
        k = np.array(list(labels))

        # 2. Obtain the first k of v^T and store it
        v = vT[k].T

        # 3. Calculate the compressed data using np.matmul(), X and stored first k of v^T
        X_compressed = cp.matmul(X_centered,v).get()

        # 4. Print the compressed value of X
        #print(v)

        plt.show()

        return X_compressed

    ################################### KMeans ###################################
        
    # Initialization Function
    def initialize_kmeans(self, n_clusters, DATA):
        """
        ----------------------------------------------------------------------------
        
        """
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(DATA)
        return kmeans

    # Plotting Function
    def plot_clusters(self, kmeans, DATA):
        """
        ----------------------------------------------------------------------------
        
        """
        sns.set_palette("colorblind")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # Create a colormap
        colorblind_palette = sns.color_palette("colorblind")
        cmap = ListedColormap(colorblind_palette)
        colors = np.array([kmeans.predict([x[:2]])[0] for x in DATA])
        norm = matplotlib.colors.Normalize(vmin=colors.min(), vmax=colors.max())

        points = ax1.scatter(*DATA.T, s=10, alpha=0.5, c=cmap(norm(colors)))
        centers = ax1.scatter(*kmeans.cluster_centers_.T, color='k', marker='x', s=50)
        ax1.set_title('Clusters')
        ax2.set_xlabel('Time [ms]')
        ax2.set_ylabel('Voltage [V]')

        
        return fig, ax1, ax2, points, centers, colors

    # Callback Functions
    def callback(self, sel, fig, ax2, points, centers, colors, cursor_p, cursor_c, N_CLUSTERS, a_data):
        """
        ----------------------------------------------------------------------------
        
        """
        I = sel.index
        cursor_p.visible = True
        cursor_c.visible = False
        sel.annotation.set_text(sel.index)
        fig.suptitle(f'Idx {I}')
        ax2.clear()
        self.pyper.append(I)

        self.plot_signal(a_data[I][1],fig=fig,ax=ax2,order=0,mysize=1/50,color='k',alpha=0.2,lw=0.5)
        self.plot_signal(a_data[I][1],fig=fig,ax=ax2,order=2,mysize=1/50,color='k',lw=1)
        ax2.plot(self.REFERENCE_VOLTAGE*self.the_data[I][1].min()/self.REFERENCE_VOLTAGE.min(), color='deepskyblue',lw=1,ls=(0,(5,10)))
        sel.annotation.get_bbox_patch().set(fc="white",lw=0,alpha=0)
        sel.annotation.arrow_patch.set(arrowstyle="]-", fc="k")
        CC = ['k']*N_CLUSTERS
        CC[colors[I]] = 'deeppink'
        centers.set_color(CC)

    def callback_center(self, sel, fig, ax2, points, centers, colors, cursor_p, cursor_c, N_CLUSTERS, a_data):
        """
        ----------------------------------------------------------------------------
        
        """
        I = sel.index
        idx_c = np.where(colors==I)[0]
        sel.annotation.set_text(len(idx_c))
        sel.annotation.get_bbox_patch().set(fc="white",lw=0,alpha=0)
        sel.annotation.arrow_patch.set(arrowstyle="wedge", fc="k")
        fig.suptitle(f'Center {I}')
        ax2.clear()
        cursor_p.visible = False
        cursor_c.visible = True
        #pyperclip.copy(", ".join([str(i) for i in idx_c]))
        #PYPER = eval("[" + ", ".join([str(i) for i in idx_c]) + "]")
        self.pyper.append(idx_c)
        print(len([str(i) for i in idx_c]))
        print((", ".join([str(i) for i in idx_c])))
        CC = ['k']*N_CLUSTERS
        CC[I] = 'cornflowerblue'
        centers.set_color(CC)

        ax2.set_title(f'Number of Points {len(idx_c)}')
        for i in idx_c:
            pass
            #plot_signal(the_data[I][1],fig=fig,ax=ax2,order=0,mysize=1/50,color='k',alpha=0.2,lw=0.5)
            self.plot_signal(a_data[i][1],fig=fig,ax=ax2,order=2,mysize=1/50,lw=1,label=f'{i}')

    # Main Function
    def KMeans_clustering(self, N_CLUSTERS, red = False, first_dim = 0, second_dim = 1):
        """
        ----------------------------------------------------------------------------
        
        """

        self.reset_pyper()
        
        if not red:
            self.reduced_the_data = []
            if self.X_compressed is None : self.X_compressed = self.feature_reduction(self.values_of_interest, self.the_data)
            DATA = self.X_compressed[:, [first_dim, second_dim]]
            a_data = self.the_data

        else:
            if self.X_compressed_red is None : self.secondary_feature_reduction()
            DATA = self.X_compressed_red[:, [first_dim, second_dim]]
            a_data = self.reduced_the_data
        # print(DATA)

        kmeans = self.initialize_kmeans(N_CLUSTERS, DATA)
        fig, ax1, ax2, points, centers, colors = self.plot_clusters(kmeans, DATA)
        
        cursor_p = mplcursors.cursor(points)
        cursor_c = mplcursors.cursor(centers)

        cursor_p.connect("add", lambda sel: self.callback(sel, fig, ax2, points, centers, colors, cursor_p, cursor_c, N_CLUSTERS, a_data))
        cursor_c.connect('add', lambda sel: self.callback_center(sel, fig, ax2, points, centers, colors, cursor_p, cursor_c,N_CLUSTERS, a_data))
    
    def reset_pyper(self):
        """
        This function should be used when you misclick on a cluster center and want to save different ones.

        Args: 
        None

        Returns:
        None

        """
        self.pyper = []


    ############################## Secondary KMeans ###################################
        
    def reduce_pyper(self, to_print = False):
        """
        This function selects points from kmeans plot clicking, on which we want to run secondary pca. 
        
        """

        if len(self.pyper) == 0: return
        
        self.reduced = np.array([])

        for i, x in enumerate(self.pyper):
            
            # If x is a list, add it to reduced and continue to the next iteration.
            if isinstance(x, np.ndarray):
                self.reduced = np.hstack((self.reduced, np.array(x)))
                if to_print: print('added a list')
                continue

            # For a point, we check if it's the last element
            if i == len(self.pyper) - 1:
                self.reduced = np.append(self.reduced, x)
                if to_print: print('added the last point')
                continue

            
            if isinstance(self.pyper[i + 1], np.ndarray):
                if to_print: print('did not add a number')
                continue

            # If not the last element and the next element is not a list, append x to reduced.
            self.reduced = np.append(self.reduced, x)
            if to_print: print('added a point')


        # self.reduced.append(self.pyper[1])
        # self.reduced = np.concatenate(self.reduced)

    def reduce_data(self):
        """
        ----------------------------------------------------------------------------
        
        """
        self.reduce_pyper()
        # print(len(self.reduced_the_data))
        for x in self.reduced:
            self.reduced_the_data.append(self.the_data[int(x)])
        # print(len(self.reduced_the_data))
        


    
    def values_of_interest_red(self, time,data,order=2,mysize=1/50):
        """
        ----------------------------------------------------------------------------
        
        """
        sos=signal.butter(order,mysize, output='sos')
        butterybiscuits=signal.sosfilt(sos,data)
        
        peak            = np.argmin(butterybiscuits)
        area            = np.sum(butterybiscuits[:-1]*(time[1:]-time[:-1]))
        peakheight      = np.min(butterybiscuits)
        halfheights     = np.where(butterybiscuits-peakheight/2<0)[0]
        fwhm            = 0 if len(halfheights)<1 else time[halfheights[-1]]-time[halfheights[0]]
        noize           = len([x for x in data if x>0.0055])
        medheight       = np.median(butterybiscuits)
        meanheight      = np.mean(butterybiscuits)
        meantime        = np.mean(np.abs(butterybiscuits)*(time))/np.sum(np.abs(butterybiscuits[:-1])*(time[1:]-time[:-1]))
        std             = np.std(butterybiscuits)
        grad            = (butterybiscuits[1:]-butterybiscuits[:-1])/(time[1:]-time[:-1])
        grad_min        = np.mean(grad[np.where(grad<0)])
        grad_max        = np.mean(grad[np.where(grad>0)])
        corr            = signal.correlate(butterybiscuits/np.min(butterybiscuits)*np.min(self.REFERENCE_VOLTAGE),self.REFERENCE_VOLTAGE)
        area_cumulative = np.cumsum(butterybiscuits[:-1]*(time[1:]-time[:-1]))
        cumt_percentage = area_cumulative/area
        percentage      = 0.96
        time_at_94_perc = time[np.argmax(cumt_percentage >= percentage)]
        time_under_trig = len([x for x in butterybiscuits if x >= -0.032])
        
        return (peakheight, time_under_trig, time[peak]/meantime ,noize, area/std, time_at_94_perc, np.max(corr))
    
    
    def secondary_feature_reduction(self, order, mysize):
        """
        Does what feature_reduction does but with the reduced data instead

        Args:
        None

        Return:
        None

        """
        # check if data has been reduced. if not does reduction.
        if len(self.reduced_the_data) == 0:
            self.reduce_data()

        
        self.X_compressed_red = self.feature_reduction(self.values_of_interest_red, self.reduced_the_data, order=order, mysize = mysize)
        # return self.X_compressed_red

    
    def secondary_KMeans_clustering(self, N_CLUSTERS, first_dim, second_dim):
        """
        This function does Kmean on a smaller subset of selected clusters 

        """
        self.KMeans_clustering(N_CLUSTERS, red=True, first_dim = first_dim, second_dim = second_dim)

    