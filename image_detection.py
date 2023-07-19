#Pentru modificarea contrastului, au fost analizate 5 functii dintre care au fost alese cele
#care au dat rezultatele cele mai bune

#Pentru postprocesare, au fost analizate 4 functii si s-a ales una-eroziunea

import numpy as np 
import matplotlib.pyplot as plt
import os 
import scipy.ndimage as sc

cale = r'D:\~AN4\IM\Imagini_set'
files = os.listdir( cale )

#lista imagini selectate
list_img=['mdb001.pgm', 'mdb015.pgm',  'mdb033.pgm', 'mdb035.pgm', 'mdb049.pgm', 'mdb154.pgm', 'mdb184.pgm', 'mdb212.pgm', 'mdb222.pgm', 'mdb315.pgm']

#vizualizarea imaginilor din baza de date
for i in files:
    if i in list_img:
        cale_img = os.path.join(cale, i)
        img_plt = plt.imread(cale_img)
        plt.figure()
        plt.imshow(img_plt)
        plt.show()

      
#%%
#PAS 1 - verificare alb-negru/color
def rgb_or_gri(img_in):
    img_in=img_in.astype('float')
    s=img_in.shape
    if len(s)==3 and s[2]==3:
        print("Imaginea este color")
    else:
        print('Imaginea este cu niveluri de gri')

for i in files:
    if i in list_img:
        cale_img = os.path.join(cale, i)
        img_plt = plt.imread(cale_img)
        rgb_or_gri(img_plt)

#%%    
# PAS 4 - Modificarea contrastului
#op 1 - negativare
def negativare(img_in,L):
    s=img_in.shape
    img_out=np.empty_like(img_in)
    img_in=img_in.astype(float)
    for i in range(0,s[0]):
        for j in range(0,s[1]):
            img_out[i,j]=L-1-img_in[i,j]
    img_out=np.clip(img_out,0,255)
    img_out=img_out.astype('uint8')
    return img_out


#op 2 - clipping
def clipping(img_in,L,a,b,Ta,Tb):
    s=img_in.shape
    img_out=np.empty_like(img_in)
    img_in=img_in.astype(float)
    for i in range(0,s[0]):
        for j in range(0,s[1]):
            if (img_in[i,j]<a):
                img_out[i,j]=0
            if(img_in[i,j]>=a and img_in[i,j]<=b):
                img_out[i,j]=Ta+((Tb-Ta)/(b-a))*(img_in[i,j]-a)
            if(img_in[i,j]>b):
                img_out[i,j]=0
                
    img_out=np.clip(img_out,0,255)
    img_out=img_out.astype('uint8')
    return img_out

#op 3 - putere
def putere(img_in,L,r):
    s=img_in.shape
    img_out=np.empty_like(img_in)
    img_in=img_in.astype(float)
    for i in range(0,s[0]):
        for j in range(0,s[1]):
            img_out[i,j]=(L-1)*(img_in[i,j]/(L-1))**r
                
    img_out=np.clip(img_out,0,255)
    img_out=img_out.astype('uint8')
    return img_out

#op 4 - logaritm
def logaritm(img_in,L):
    s=img_in.shape
    img_out=np.empty_like(img_in)
    img_in=img_in.astype(float)
    for i in range(0,s[0]):
        for j in range(0,s[1]):
            img_out[i,j]=(L-1)/np.log(L)*np.log(img_in[i,j]+1)
                
    img_out=np.clip(img_out,0,255)
    img_out=img_out.astype('uint8')
    return img_out

#op 5 - exponential
def exponential(img_in,L):
    s=img_in.shape
    img_out=np.empty_like(img_in)
    img_in=img_in.astype(float)
    for i in range(0,s[0]):
        for j in range(0,s[1]):
            img_out[i,j]=L**(img_in[i,j]/(L-1))-1       
                
    img_out=np.clip(img_out,0,255)
    img_out=img_out.astype('uint8')
    return img_out


for i in files:
    if i in list_img:
        cale_img = os.path.join(cale, i)
        img_plt = plt.imread(cale_img)
        plt.figure()
        #L este numarul de niveluri de gri 256
        img_neg=negativare(img_plt, 256)
        img_clip=clipping(img_plt,256,100,160,80,180)
        img_putere=putere(img_plt,256,8)
        img_logaritm=logaritm(img_plt,256)
        img_exponential=exponential(img_plt,256)
        plt.subplot(2,3,1), plt.imshow(img_plt, cmap='gray'), plt.title('Imaginea initiala')
        plt.subplot(2,3,2), plt.imshow(img_neg, cmap='gray'), plt.title('Imaginea negativata')
        plt.subplot(2,3,3), plt.imshow(img_clip, cmap='gray'), plt.title('Imaginea clipping')
        plt.subplot(2,3,4), plt.imshow(img_putere, cmap='gray'), plt.title('Imaginea putere')
        plt.subplot(2,3,5), plt.imshow(img_logaritm, cmap='gray'), plt.title('Imaginea logaritmica')
        plt.subplot(2,3,6), plt.imshow(img_exponential, cmap='gray'), plt.title('Imaginea exponentiala')
        plt.show()
        
#%%
#PAS 5 - Segmentare
#binarizare
def binarizare(img_in,L,a):
    s=img_in.shape
    img_out=np.empty_like(img_in)
    img_in=img_in.astype(float)
    for i in range(0,s[0]):
        for j in range(0,s[1]):
            if (img_in[i,j]<a):
                img_out[i,j]=0
            if(img_in[i,j]>=a):
                img_out[i,j]=255
        
                
    img_out=np.clip(img_out,0,255)
    img_out=img_out.astype('uint8')
    return img_out

#slicing
def slicing(img_in,L,a,b,Ta):
    s=img_in.shape
    img_out=np.empty_like(img_in)
    img_in=img_in.astype(float)
    for i in range(0,s[0]):
        for j in range(0,s[1]):
            if (img_in[i,j]<a):
                img_out[i,j]=0
            if(img_in[i,j]>=a and img_in[i,j]<=b):
                img_out[i,j]=Ta
            if(img_in[i,j]>b):
                img_out[i,j]=0
                
    img_out=np.clip(img_out,0,255)
    img_out=img_out.astype('uint8')
    return img_out        

for i in files:
    if i in list_img:
        cale_img = os.path.join(cale, i)
        img_plt = plt.imread(cale_img)
        plt.figure()
        img_putere=putere(img_plt,256,8)
        img_binarizare=binarizare(img_putere,256,70)
        selectie_zona_interes=binarizare(img_putere,256,70)/255*img_putere
        img_slicing=slicing(img_putere,256,70,255,255)
        #imaginea 'mdb15' exemplu negativ al segmentarii, iar 'mdb184' exemplu pozitiv
        plt.subplot(2,2,1), plt.imshow(img_plt, cmap='gray'), plt.title('Imaginea initiala')
        plt.subplot(2,2,2), plt.imshow(img_putere, cmap='gray'), plt.title('Imaginea putere')
        plt.subplot(2,2,3), plt.imshow(img_binarizare, cmap='gray'), plt.title('Imaginea binarizare')
       #plt.subplot(2,2,3), plt.imshow(selectie_zona_interes, cmap='gray'), plt.title('Selectia zonei de interes')
        plt.subplot(2,2,4), plt.imshow(img_slicing, cmap='gray'), plt.title('Imaginea slicing')
        plt.show()
        
#%%
#PAS 6 - Postprocesarea

e1=np.array([[1,1,1],[1,1,1],[1,1,1]])

for i in files:
    if i in list_img:
        cale_img = os.path.join(cale, i)
        img_plt = plt.imread(cale_img)
        plt.figure()
        img_putere=putere(img_plt,256,8)
        img_slicing=slicing(img_putere,256,70,255,255)
        img_erodare1=sc.binary_erosion(img_slicing, structure=e1)
        img_dilatare1=sc.binary_dilation(img_slicing, structure=e1)
        img_deschidere_1=sc.binary_opening(img_slicing,structure=e1)
        img_inchidere_1=sc.binary_closing(img_slicing,structure=e1)

        plt.subplot(2,3,1), plt.imshow(img_slicing,cmap='gray'), plt.title('Imaginea dupa segmentare')
        plt.subplot(2,3,2), plt.imshow(img_erodare1, cmap='gray'), plt.title('Imaginea erosion')
        plt.subplot(2,3,3), plt.imshow(img_dilatare1,cmap='gray'), plt.title('Imaginea dilation')
        plt.subplot(2,3,4), plt.imshow(img_deschidere_1, cmap='gray'), plt.title('Imaginea opening')
        plt.subplot(2,3,5), plt.imshow(img_inchidere_1,cmap='gray'), plt.title('Imaginea closing')
        plt.subplot(2,3,6), plt.imshow(img_plt, cmap='gray'), plt.title('Imaginea initiala')
        plt.show()   