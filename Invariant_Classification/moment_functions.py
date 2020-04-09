# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:51:36 2020

@author: markb
"""
import numpy as np
from scipy.special import binom

#Function accepts a rectangular nparray - the image stored in X
#(p,q) indicates which moment to compute x^p.y^q
def Moments(X,p,q):
    n=X.shape[0]
    m=X.shape[1]
    A=np.zeros([n,m])
    
    for i in range(n):
        for j in range(m):
            x=(i+1/2)**(p+1)-(i-1/2)**(p+1)
            y=(j+1/2)**(q+1)-(j-1/2)**(q+1)
            A[i,j]=x*y
            
    c=(p+1)*(q+1)
    A=(1/c)*A

    M=np.multiply(A,X)
    return M.sum()



def Centroid(X):
    a=Moments(X,0,0)
    b=Moments(X,1,0)
    c=Moments(X,0,1)
    
    return np.array([b/a,c/a])

def Central_Moment(X,p,q):
    """
    Calculate the (p,q)th moment of array X, centralized by the centroid of X
    Central Moments are invariant to translations of the domain of X
    
    Parameters
    ----------
    X : ndarray
        Rectangular array storing image
    p : int
        moment in x direction
    q : int
        moment in y direction

    Returns
    -------
    M.sum() : float
        (p,q)th centralized moment of X.

    Examples
    --------

    """
    n=X.shape
    c=Centroid(X)
    A=np.zeros(n)
    
    for i in range(n[0]):
        for j in range(n[1]):
            a=i-c[0] #translate the domain by centroid
            b=j-c[1]
            x=(a+1/2)**(p+1)-(a-1/2)**(p+1) #integrate pqth moment
            y=(b+1/2)**(q+1)-(b-1/2)**(q+1)
            A[i,j]=x*y
            
    
    c=(p+1)*(q+1)
    A=(1/c)*A
    M=np.multiply(A,X)
    
    return M.sum()


def Standard_Moment(X,p,q,log_arg=False):
    """
    Calculate the (p,q)th standard moment of array X
        Centralized by the centroid of X
        Normalized by taking quotient of (pq)th central moment
         by (0,0)th moment raised to the power of (p+q+2)/2
    Standard Moments are invariant to translations and isotropic
     scaling of the domain 
    
    Parameters
    ----------
    X : ndarray
        Rectangular array storing image
    p : int
        moment in x direction
    q : int
        moment in y direction
    log_arg : bool
        input True to compute the natural log of the moments

    Returns
    -------
     : float
        (p,q)th standardized moment of X.

    Examples
    --------

    """
    n=X.shape
    tot_power=Central_Moment(X,0,0)
    cent_moment=Central_Moment(X,p,q)
    tolerance=1e-10
    
    a=(p+q+2)/2
    
    if (log_arg==True):
        if (np.abs(cent_moment)>tolerance):        
            s=np.sign(cent_moment)
            r=np.abs(cent_moment/tot_power**a)
            return s*np.log(r)
        else:
            return np.floor(np.log(tolerance))
            
    
    return cent_moment/tot_power**a




#Input:rectangular array of "geometric" moments 
#Output:rectangular array of complex moments
def Complexify_Moments(M,dMax):
    C=np.zeros(M.shape,dtype=np.complex)
    C[0,0]=M[0,0]
    for d in range(1,dMax+1):
        for p in range(d+1):
            q=d-p
            z=0
            for a in range(p+1):
                for b in range(q+1):
                    s=binom(p,a)
                    t=binom(q,b)
                    r=1j**(p+3*q-a-3*b)
                    z+=s*t*r*M[a+b,p+q-a-b]
            C[p,q]=z
    
    return C

#for idx, x in np.ndenumerate(M):
#p=idx[0]   q=idx[1]
    
#Normalizing gives invariance to isotropic scaling of the domain
def Normalize_Moments(M):
    m=M[0,0]
    N=np.zeros(M.shape)
    for idx, x in np.ndenumerate(M):
        p=idx[0]
        q=idx[1]
        k=(p+q+2)/2
        N[idx]=M[idx]/m**k    
    return N
#def szof(dMax):
#    s=1
#    for d in range(1,dMax+1):
#        s+=np.floor(d/2)+1
#    
#    return int(s)-1
    
def Flus_Basis(C,dMax):
    m=C[2,1] #keypoint moment
    s=1
    for d in range(1,dMax+1):
        s+=np.floor(d/2)+1
        
    f=int(s)-1 #This is the size of the output- no. of basis elts
    
    F=np.zeros(f,dtype=np.complex)

    c=0
    
    for d in range(1,dMax+1):
        for q in range(d):
            p=d-q
            F[c]=C[p,q]*m**(p-q)
            c+=1
            if c>(dMax+1):
                break
            
    return F

#def Flus_Moment(C):
#    m=C[2,1]
#    F=np.zeros(C.shape,dtype=np.complex)
#    for idx, x in np.ndenumerate(C):
#        p=idx[0]
#        q=idx[1]
#        if p >= q:
#            F[idx]=C[p,q]*m**(p-q)
#    
#    return F
    
def Flus_Moment(C,dMax):
    m=C[2,1]
    num_cmoms=1
    for d in range(1,dMax+1):
        for q in range(int(np.floor(d/2)+1)):
            num_cmoms+=1
            
    F=np.zeros(num_cmoms,dtype=np.complex)
    fidx=np.zeros(num_cmoms,dtype=tuple)
    c=0
    for d in range(dMax+1):
        for q in range(int(np.floor(d/2)+1)):
            p=d-q
            fidx[c]=(p,q)
            F[c]=C[p,q]*m**(p-q)
    return F

#class moment_class(object):
#    def __init__(self,X=None,d=5):
#        self.data=X
#        self.dMax=d
#        
#    
#    def compute_all(self):
#        chr_num=self.data.shape[0]
#        img_num=self.data.shape[1]
#        dMax=self.dMax
#        
#        self.M=np.zeros([chr_num,img_num,dMax+1,dMax+1])
#        
#        print("----------- Computing Standardized Moments -----------")
#        for n in range(chr_num):
#            for k in range(img_num):
#                for d in range(dMax+1):
#                    for p in range(d+1):
#                        q=d-p
#                        self.M[n,k,p,q]=Standard_Moment(self.data[n,k,:].reshape(28,28),p,q)
#            print("All of character ",n," moments computed")
#        print("----------- Finished Standardized Moments -----------")
#        
#        #Vectorize the moments
#        num_moms=int((dMax+1)*(dMax+2)/2)
#        
#        self.Mvec=np.zeros([chr_num,img_num,num_moms])
#        self.vec_idx=np.empty(num_moms,dtype=tuple)
#        c=0
#        for d in range(dMax+1):
#            for q in range(d+1):
#                p=d-q
#                self.vec_idx[c]=(p,q)
#                c+=1
#                
#        c=0
#        for n in range(chr_num):
#            for k in range(img_num):
#                c=0
#                for d in range(dMax+1):
#                    for q in range(d+1):
#                        p=d-q
#                        self.Mvec[n,k,c]=self.M[n,k,p,q]
#                        c+=1
#        
#        #Compute the complex moments
#        self.C=np.zeros(self.M.shape,dtype=np.complex)
#        for n in range(chr_num):
#            for k in range(img_num):
#                self.C[n,k,:,:]=Complexify_Moments(self.M[n,k,:,:],dMax)
#        
#        print("----------- Finished Complex Moments -----------")
#        
#        #Vectorized complex moments (take only p>=q)
#        num_cmoms=1
#        for d in range(1,dMax+1):
#            for q in range(int(np.floor(d/2)+1)):
#                num_cmoms+=1
#        
#        self.cvec_idx=np.empty(num_cmoms,dtype=tuple)
#        self.Cvec=np.zeros([chr_num,img_num,num_cmoms],dtype=complex)
#        
#        for n in range(chr_num):
#            for k in range(img_num):
#                c=0
#                for d in range(dMax+1):
#                    for q in range(int(np.floor(d/2)+1)):
#                        p=d-q
#                        self.Cvec[n,k,c]=self.C[n,k,p,q]
#                        self.cvec_idx[c]=(p,q)
#                        c+=1
#                        
#        
#        
#        
#        #M=np.zeros([dMax+1,dMax+1])
#        #
#        #
#        #for d in range(dMax+1):
#        #    for p in range(d+1):
#        #        q=d-p
#        #        M[p,q]=Standard_Moment(Y,p,q)
#        #
#        #C=Complexify_Moments(M,dMax)
#        
#        #Compute the Flusser Invariant Basis
#        self.B=np.zeros([chr_num,img_num,num_cmoms],dtype=np.complex)
#        for n in range(chr_num):
#            for k in range(img_num):
#                self.B[n,k,:]=Flus_Moment(self.C[n,k,:,:],dMax)
#        
#        self.num_flusmoments=sum(self.B[0,0,:].shape)
#        print("----------- Finished Flusser Basis -----------")
#        
#    def StandardizedMoments(self,flag=False):
#        if flag==True:
#            return np.log(self.Mvec)
#        else:
#            return self.Mvec[3:]
#    
#    def ComplexMoments(self,flag=False):
#        v=self.Cvec[:,:,2:] #image has been normalized so 00 and 10 cplx mmt are removed
#        if flag==True:
#            return np.log(v)
#        else:
#            return v
#        
#    def ImagePower(self):
#        v=np.zeros([chr_num,img_num])
#        for n in range(chr_num):
#            for k in range(img_num):
#                v[n,k]=Moments(self.data,0,0)
#        return v
#    
#    def FlusserMoments(self,flag=False):
#        v=self.B[:,:,2:]
#        if flag==True:
#            return np.log(v)
#        else:
#            return v
#








class momentClass(object):
    def __init__(self,X=None,d=5):
        self.data=X
        self.dMax=d
        
    
    def compute_all(self):
        chr_num=self.data.shape[0]
        img_num=self.data.shape[1]
        dMax=self.dMax
        #####################################################################
        #Compute ordinary Cartesian moments
        self.Moments=np.zeros([chr_num,img_num,dMax+1,dMax+1])
        print("----------- Computing Geometric Moments -----------")
        for n in range(chr_num):
            for k in range(img_num):
                for d in range(dMax+1):
                    for p in range(d+1):
                        q=d-p
                        self.Moments[n,k,p,q]=Moments(self.data[n,k,:].reshape(28,28),p,q)
            print("All of character ",n," moments computed")
        print("----------- Finished Geometric Moments -----------\n")
        #Vectorize
        num_moms=int((dMax+1)*(dMax+2)/2)
        
        self.MomentsVec=np.zeros([chr_num,img_num,num_moms])
                
        c=0
        for n in range(chr_num):
            for k in range(img_num):
                c=0
                for d in range(dMax+1):
                    for q in range(d+1):
                        p=d-q
                        self.MomentsVec[n,k,c]=self.Moments[n,k,p,q]
                        c+=1
        
        ####################################################################
        
        
        
        
        ####################################################################
        #Compute Standardized Moments
        self.StandardizedMoments=np.zeros([chr_num,img_num,dMax+1,dMax+1])
        print("----------- Computing Standardized Moments -----------")
        for n in range(chr_num):
            for k in range(img_num):
                for d in range(dMax+1):
                    for p in range(d+1):
                        q=d-p
                        self.StandardizedMoments[n,k,p,q]=Standard_Moment(self.data[n,k,:].reshape(28,28),p,q)
            print("All of character ",n," moments computed")
        print("----------- Finished Standardized Moments -----------\n")
        #Vectorize the moments
        num_moms=int((dMax+1)*(dMax+2)/2)
        
        self.StandardizedMomentsVec=np.zeros([chr_num,img_num,num_moms])
        self.vec_idx=np.empty(num_moms,dtype=tuple)
        c=0
        for d in range(dMax+1):
            for q in range(d+1):
                p=d-q
                self.vec_idx[c]=(p,q)
                c+=1
                
        c=0
        for n in range(chr_num):
            for k in range(img_num):
                c=0
                for d in range(dMax+1):
                    for q in range(d+1):
                        p=d-q
                        self.StandardizedMomentsVec[n,k,c]=self.StandardizedMoments[n,k,p,q]
                        c+=1
        ####################################################################
        
        
        
        ####################################################################
        #Compute the complex moments
        self.ComplexMoments=np.zeros(self.StandardizedMoments.shape,dtype=np.complex)
        for n in range(chr_num):
            for k in range(img_num):
                self.ComplexMoments[n,k,:,:]=Complexify_Moments(self.StandardizedMoments[n,k,:,:],dMax)
        
        print("----------- Finished Complex Moments -----------\n")
        
        #Vectorized complex moments (take only p>=q)
        num_cmoms=1
        for d in range(1,dMax+1):
            for q in range(int(np.floor(d/2)+1)):
                num_cmoms+=1
        
        self.cvec_idx=np.empty(num_cmoms,dtype=tuple)
        self.ComplexMomentsVec=np.zeros([chr_num,img_num,num_cmoms],dtype=complex)
        
        for n in range(chr_num):
            for k in range(img_num):
                c=0
                for d in range(dMax+1):
                    for q in range(int(np.floor(d/2)+1)):
                        p=d-q
                        self.ComplexMomentsVec[n,k,c]=self.ComplexMoments[n,k,p,q]
                        self.cvec_idx[c]=(p,q)
                        c+=1
                        
        ####################################################################
        
        
  
        ####################################################################
        #Compute the Flusser Invariant Basis
        self.FlusserMoments=np.zeros([chr_num,img_num,num_cmoms],dtype=np.complex)
        for n in range(chr_num):
            for k in range(img_num):
                self.FlusserMoments[n,k,:]=Flus_Moment(self.ComplexMoments[n,k,:,:],dMax)
        
        self.num_flusmoments=sum(self.FlusserMoments[0,0,:].shape)
        print("----------- Finished Flusser Basis -----------\n")
        ####################################################################
        
        
        self.stdMoments=self.StandardizedMomentsVec[:,:,3:]
        
        self.cmpMoments=self.ComplexMomentsVec[:,:,2:]
        
        self.imgPower=self.MomentsVec[:,:,0]
        
        self.flusMoments=self.FlusserMoments[:,:,2:]
    
  