# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:33:50 2020

@author: markb
"""

import numpy as np
from scipy.ndimage.measurements import center_of_mass
from scipy.special import binom

#one class for geo_all_exact and geo_all
#will have a get function to index by p,q
#will take in a range of 
class geo(object):
    pass


def geo_all_exact(img,d_max):
    cent=np.array(center_of_mass(img))
    sz=img.shape[0]
    num_mts=int((d_max+1)*(d_max+2)/2)
    
    #Image not scaled to unit box
    #simply used integer pixel coords
    #now centralize:
    xcoords=np.arange(sz)-cent[0]*np.ones(sz)
    ycoords=np.arange(sz)-cent[1]*np.ones(sz)
        
    #multiply by x to advance p by 1
    #multiply by y to advance q by 1
    #use x and y to make mesh of the integral values
    x=np.array([xcoords-1/2*np.ones(sz),xcoords+1/2*np.ones(sz)])
    y=np.array([ycoords-1/2*np.ones(sz),ycoords+1/2*np.ones(sz)])
    
    #Initialize array of moments and current x and y
    geo=np.zeros(num_mts)
    ctr=0  
    xcur=np.copy(x)
    ycur=np.copy(y)
    ynxt=ycur*y
    
    for q in range(d_max+1):
        for p in range(d_max+1-q):
            xdlt=(xcur[1]-xcur[0]).reshape(sz,1)
            ydlt=(ycur[1]-ycur[0]).reshape(1,sz)
            mesh=img*(xdlt*ydlt) #pointwise mult of img with integral mesh
            geo[ctr]=1/((p+1)*(q+1))*mesh.sum()
            ctr+=1
            #update current x coords to x**(p+1)
            xcur*=x
        #reset vars to next
        xcur[:]=x #moves p back to 0
        ycur[:]=ynxt
        ynxt*=y #moves q up 1 for next iteration
        
    return geo
        


def compute_geo(img,p,q):
    cent=np.array(center_of_mass(img))
    sz=img.shape[0]
    
    
    #Image not scaled to unit box
    #simply used integer pixel coords
    #now centralize:
    xcoords=np.arange(sz)-cent[0]*np.ones(sz)
    ycoords=np.arange(sz)-cent[1]*np.ones(sz)
        
    #multiply by x to advance p by 1
    #multiply by y to advance q by 1
    #use x and y to make mesh of the integral values
    x=np.array([xcoords-1/2*np.ones(sz),xcoords+1/2*np.ones(sz)])**(p+1)
    y=np.array([ycoords-1/2*np.ones(sz),ycoords+1/2*np.ones(sz)])**(q+1)
    
    xdlt=(x[1]-x[0]).reshape(sz,1)
    ydlt=(y[1]-y[0]).reshape(1,sz)
    mesh=img*(xdlt*ydlt) #pointwise mult of img with integral mesh
    
    return 1/((p+1)*(q+1))*mesh.sum()
    
def compute_cmp(img,p,q):
    deg=p+q
    tot=0
    for a in range(p+1):
        for b in range(q+1):
            n=a+b
            mlt=(1j)**(deg-n)*((-1)**(q-b))*binom(p,a)*binom(q,b)
            geo_mt=compute_geo(img,n,deg-n)
            tot+=mlt*geo_mt
    
    return tot
    
    
def binom_naive(geo,idx,p,q):
    deg=p+q
    tot=0
    for a in range(p+1):
        for b in range(q+1):
            n=a+b
            #crd=idx[n,deg-n]
            crd=idx[deg-n,n]
            mlt=(1j)**(a+3*b)*binom(p,a)*binom(q,b)
            #mlt=(1j)**(deg-n)*((-1)**(q-b))*binom(p,a)*binom(q,b)
            tot+=mlt*geo[crd]
    return tot
        

def compute_all(img,d_max,idx):
    geo=geo_all_exact(img,d_max)
    num_geo=int((d_max+1)*(d_max+2)/2)
    cmp=np.zeros(num_geo,dtype=np.complex)
    
    ctr=0
    for q in range(d_max+1):
        for p in range(d_max+1-q):
            cmp[ctr]=binom_naive(geo,idx,p,q)
            ctr+=1
    
    return geo,cmp


def flus(cmp,idx,crd=(2,1)):
    key=cmp[idx[crd]]
    flus_arr=np.zeros(cmp.shape[0],dtype=np.complex)
    for ctr in range(cmp.shape[0]):
        p,q=idx[ctr]
        flus_arr[ctr]=key**(q-p)*cmp[ctr]
    return flus_arr      
        
class Moments(object):
    def __init__(self,X):
        self.data=X
        self.img_sz=28
        self.num_imgs=X.shape[0]
        
        
    def compute(self,d_max):
        num_imgs=self.num_imgs
        idx={}
        ctr=0
        for q in range(d_max+1):
            for p in range(d_max+1-q):
                idx[(p,q)]=ctr
                idx[ctr]=(p,q)
                ctr+=1
        self.idx=idx
        
        num_mts=int((d_max+1)*(d_max+2)/2)
        print("-"*20,"Computing Moments","-"*20)
        geo_all=np.zeros([num_imgs,num_mts])
        cmp_all=np.zeros([num_imgs,num_mts],dtype=np.complex)
        flus_all=np.zeros([num_imgs,num_mts],dtype=np.complex)
        ctr=0
        for n in range(num_imgs):
            cur_img=self.data[n].reshape(self.img_sz,self.img_sz)
            geo_all[ctr],cmp_all[ctr]=compute_all(cur_img,d_max,idx)
            flus_all[ctr]=flus(cmp_all[ctr],self.idx)
            ctr+=1
            #print("Finished Moment ",n)
            
        self.geo=geo_all
        self.cmp=cmp_all
        self.flus=flus_all
        return
        


    
    

#Make a class to handle triangular (as opposed to rectangular) arrays
class Tri_Array(object):
    def __init__(self,arr,idx):
        self.arr=arr
        self.idx=idx
        
    def get(self,p,q):
        return self.arr[self.idx[p,q]]
    
    def gen(deg):
        return gen_diag(deg)
    
    def gen_all(d_max):
        return gen_diag_all(d_max)
    
####### 
#Generators to iterate over a diagonal in moments plane (p,q) p+q=d
def gen_diag(deg):
    for q in range(deg+1):
        p=deg-q
        yield (p,q)

def gen_diag_all(d_max):
    for d in range(d_max+1):
        for q in range(d+1):
            p=d-q
            yield (p,q)
            





           
##########################    
    
def geo_all(X,dMax,numPts=100):
    #X is image in square grid of size numPts by numPts
    Box=np.ones((numPts,numPts)) #compute this outside the function
    ImgMesh=np.kron(X,Box) #recheck this to make sure kron dimensions broadcast proper
    sz=ImgMesh.shape[0]
    cent=(1/sz)*np.array(center_of_mass(ImgMesh)) 
    
    #0th moment
    dA=1/((sz-1)**2)
    imPower=dA*ImgMesh.sum()
    
    #create coordinate array scaled to be in unit square and centralized
    #add extra axis-to be used for broadcasting when computing moments
    coordX=np.linspace(-1*cent[0],-1*cent[0]+1,sz).reshape(sz,1)
    coordY=np.linspace(-1*cent[1],-1*cent[1]+1,sz).reshape(1,sz)
    
    
    
    #Iterate over (p,q) plane to compute (p,q)th moment
        
    #Initialize array to store complex moments
    numGeo,numCmp=numMoms(dMax)
    geoMoments=np.zeros(numGeo)
    
    #Build Up and Right Array
    z=np.zeros(sz)
    UpArr=coordY+z.reshape(sz,1)
    RtArr=coordX+z.reshape(1,sz)
    #do first column minus origin (q>0) outside of main loop
    CurrArr=np.copy(coordY)
    ctr=0
    for q in range(1,dMax+1):
        geoMoments[ctr]=(CurrArr*ImgMesh).sum()
        ctr+=1
        CurrArr=CurrArr*coordY
        
    CurrArr=coordX+(1j)*coordY
    NextArr=UpArr*CurrArr #Move up i.e. q++1
    ctr=0    
    for p in range(1,dMax+1):
        cmpMoments[ctr]=(CurrArr*ImgMesh).sum()
        CurrArr=CurrArr*RtArr #move right p++1
        ctr+=1
        
    #Main loop
    for q in range(1,qMax+1):
        CurrArr=np.copy(NextArr)
        NextArr=CurrArr*UpArr
        for p in range(q,dMax-q+1):
            cmpMoments[ctr]=(CurrArr*ImgMesh).sum()
            CurrArr=CurrArr*RtArr
            ctr+=1
    
    #scale by area form for Riemann sum
    
    geoMmts=dA*geoMoments
    return geoMoments


def tensor_view(geo,d_max):
    idx={}
    ctr=0
    for q in range(d_max+1):
        for p in range(d_max+1-q):
            idx[(p,q)]=ctr
            #idx[ctr]=(p,q)
            ctr+=1
    
    view=[]
    for d in range(d_max+1):
        crds=np.zeros(d+1,dtype=int)
        for q in range(d+1):
            p=d-q
            crds[q]=idx[p,q]
        view.append(geo[crds])
        #both evaluate to false, so not actually a view...fix!
        #print(view[q].base is geo)
        #print(geo[crds].base is geo)
    
    return view


##########################################################


def numMoms(dMax):
    numGeo=int((dMax+1)*(dMax+2)/2)
    numCmp=0
    for d in range(dMax+1):
       for q in range(int(np.floor(d/2)+1)):
           numCmp+=1
    return numGeo, numCmp


def computeAll(Data,dMax,numPts=100):
    print("*"*15,"Computing Moments","*"*15)
    numImgs=Data.shape[0]
    Box=np.ones((numPts,numPts))
    numGeo,numCmp=numMoms(dMax)
    cmpMoms=np.zeros((numImgs,numCmp),dtype=np.complex)
    for n in range(numImgs):
        X=Data[n].reshape(28,28)
        cmpMoms[n]=cmpMoments(X,dMax,Box,numPts)
        print("\t","Finished Complex Moments of image number ",n)
        
    
    return cmpMoms

def cmpMoments(X,dMax,Box,numPts=100):
    #X is image in square grid of size numPts by numPts
    #Box=np.ones((numPts,numPts)) #compute this outside the function
    ImgMesh=np.kron(X,Box)
    sz=ImgMesh.shape[0]
    cent=(1/sz)*np.array(center_of_mass(ImgMesh)) 
    
    #create coordinate array scaled to be in unit square and centralized
    #add extra axis-to be used for broadcasting when computing moments
    coordX=np.linspace(-1*cent[0],-1*cent[0]+1,sz).reshape(sz,1)
    coordY=np.linspace(-1*cent[1],-1*cent[1]+1,sz).reshape(1,sz)
    
    #Iterate over (p,q) plane to compute (p,q)th moment

    qMax=int(np.floor(dMax/2))
    RtArr=coordX+(1j)*coordY #multiply by this array to move right i.e. p+1
    UpArr=np.conj(RtArr) #multiply by this array to move up  i.e. q++1
    
    #initialize array to store complex moments
    numGeo,numCmp=numMoms(dMax)
    cmpMoments=np.zeros(numCmp,dtype=np.complex)
    
    #Do first row q=0, p=1..dMax outside of main loop
    #avoids checking for p=q=0 case
    CurrArr=np.copy(RtArr)
    NextArr=UpArr*CurrArr #Move up i.e. q++1
    ctr=0    
    for p in range(1,dMax+1):
        cmpMoments[ctr]=(CurrArr*ImgMesh).sum()
        CurrArr*=RtArr #move right p++1
        ctr+=1
        
    #Main loop
    for q in range(1,qMax+1):
        CurrArr[:]=NextArr
        NextArr=CurrArr*UpArr
        for p in range(q,dMax-q+1):
            cmpMoments[ctr]=(CurrArr*ImgMesh).sum()
            CurrArr=CurrArr*RtArr
            ctr+=1
    
    #scale by area form for Riemann sum
    dA=1/((sz-1)**2)
    cmpMoments*=dA
    return cmpMoments

def cmp_moment(X,p,q,num_pts=100):
    box=np.ones((num_pts,num_pts))
    img_mesh=np.kron(X,box)
    sz=img_mesh.shape[0]
    cent=(1/sz)*np.array(center_of_mass(img_mesh))  
    
    x_coords=np.linspace(-1*cent[0],-1*cent[0]+1,sz).reshape(sz,1)
    y_coords=np.linspace(-1*cent[1],-1*cent[1]+1,sz).reshape(1,sz)
    
    new_img=(x_coords+1j*y_coords)**p*(x_coords-1j*y_coords)**q
    new_img=new_img*img_mesh
    dA=1/((sz-1)**2)
    mmt=dA*(new_img.sum())
    return mmt, new_img
    
    
    
    
def idxCmp3(dMax):
    ctr=0
    idx={}
    qMax=int(np.floor(dMax/2))
    for p in range(1,dMax+1):
        idx[ctr]=(p,0)
        idx[(p,0)]=ctr
        ctr+=1

    #Main loop
    for q in range(1,qMax+1):
        for p in range(q,dMax-q+1):
            idx[ctr]=(p,q)
            idx[(p,q)]=ctr
            ctr+=1
    return idx

def view_moment(X,p,q,Box,numPts=100):
    ImgMesh=np.kron(X,Box)
    sz=ImgMesh.shape[0]
    cent=(1/sz)*np.array(center_of_mass(ImgMesh))
    coordX=np.linspace(-1*cent[0],-1*cent[0]+1,sz).reshape(sz,1)
    coordY=np.linspace(-1*cent[1],-1*cent[1]+1,sz).reshape(1,sz)    
    
    RtArr=coordX+(1j)*coordY
    UpArr=np.conj(RtArr)
    
    Arr=((RtArr)**p)*((UpArr)**q)
    plt.imshow(Arr.real)
    plt.figure()
    plt.imshow(Arr.imag)
        
def rotate(X,theta):
    cent=np.array(center_of_mass(X))
    Y=np.zeros(X.shape)
    R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    
    for idx, pxl in np.ndenumerate(X):
        idx=np.array(idx)
        new_idx=np.matmul(R,idx-cent)+cent

        rnd_idx=np.array(np.round(new_idx),dtype=int)
        if (rnd_idx[0] >=28 or rnd_idx[1]>= 28):
            pass
        else:
            Y[tuple(rnd_idx)]=pxl  
            
    return Y


    

    