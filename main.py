
# 64bit version of the program is also available in GitHub:
# https://github.com/logicool-repo/shifted_image_subpixel_coregistration

import numpy as np
from scipy.fftpack import *
import matplotlib.pyplot as plt
import cv2
import argparse
import glob
import os
from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D
import time

def fft_coreg(master,slave,slave0):

    # fft2
    master_fd = fft2(master)
    slave_fd = fft2(slave)

    # normalization
    master_nfd = master_fd/np.abs(master_fd)
    slave_nfd = slave_fd/np.abs(slave_fd)

    usfac = 100
    output, Nc, Nr, peak_map = dftregistration(master_nfd,slave_nfd,usfac)

    nr, nc = slave.shape
    diffphase = output[1]
    row_shift = output[2]
    col_shift = output[3]

    # print(col_shift[0],row_shift[0])

    # shift by cmul and ifft
    # slave_fd_shift = slave_fd*np.exp(1j*2*np.pi*(-row_shift*Nr/nr-col_shift*Nc/nc))*np.exp(1j*diffphase)
    # slave_shift = ifft2(slave_fd_shift)
    # slave_shift = np.abs(slave_shift)

    # shift by affine transform
    rows,cols = slave0.shape
    M_coreg = np.float32([[1,0,col_shift],[0,1,row_shift]])
    slave_shift = cv2.warpAffine(slave0,M_coreg,(cols,rows))

    return slave_shift, peak_map, col_shift[0], row_shift[0]

def dftregistration(buf1ft,buf2ft,usfac):

    nr,nc = buf2ft.shape
    Nr = ifftshift(np.arange(-np.fix(nr/2),np.ceil(nr/2)))
    Nc = ifftshift(np.arange(-np.fix(nc/2),np.ceil(nc/2)))

    if usfac == 0:
        # Simple computation of error and phase difference without registration
        CCmax = np.sum(buf1ft*np.conjugate(buf2ft))
        row_shift = 0
        col_shift = 0
    elif usfac == 1:
        # Single pixel registration
        CC = ifft2(buf1ft*np.conjugate(buf2ft))
        CCabs = np.abs(CC)
        row_shift, col_shift = np.where(CCabs == np.max(CCabs))
        CCmax = CC[row_shift,col_shift]*nr*nc
        # Now change shifts so that they represent relative shifts and not indices
        row_shift = Nr[row_shift]
        col_shift = Nc[col_shift]
    elif usfac > 1:
        # Start with usfac == 2
        CC = ifft2(FTpad(buf1ft*np.conjugate(buf2ft),(2*nr,2*nc)))
        CCabs = np.abs(CC)
        
        ##
        row_shift, col_shift = np.where(CCabs == np.max(CCabs))
        peak_map = ifftshift(CCabs)
        peak_map = np.roll(peak_map,-row_shift,axis=0)
        peak_map = np.roll(peak_map,-col_shift,axis=1)
        ##
        
        # row_shift, col_shift = row_shift[0], col_shift[0]
        CCmax = CC[row_shift,col_shift]*nr*nc
        # Now change shifts so that they represent relative shifts and not indices
        Nr2 = ifftshift(np.arange(-np.fix(nr),np.ceil(nr)))
        Nc2 = ifftshift(np.arange(-np.fix(nc),np.ceil(nc)))
        row_shift = Nr2[row_shift]/2
        col_shift = Nc2[col_shift]/2

        # If upsampling > 2, then refine estimate with matrix multiply DFT
        if usfac > 2:
            ### DFT computation ###
            # Initial shift estimate in upsampled grid
            row_shift = np.round(row_shift*usfac)/usfac
            col_shift = np.round(col_shift*usfac)/usfac
            dftshift = np.fix(np.ceil(usfac*1.5)/2) ## Center of output array at dftshift+1
            # Matrix multiply DFT around the current shift estimate
            CC = np.conjugate(dftups(buf2ft*np.conjugate(buf1ft),np.ceil(usfac*1.5),np.ceil(usfac*1.5),usfac,dftshift-row_shift*usfac,dftshift-col_shift*usfac))
            # Locate maximum and map back to original pixel grid 
            CCabs = np.abs(CC)
            rloc, cloc = np.where(CCabs == np.max(CCabs))
            # rloc, cloc = rloc[0], cloc[0]
            CCmax = CC[rloc,cloc]
            rloc = rloc - dftshift
            cloc = cloc - dftshift
            row_shift = row_shift + rloc/usfac
            col_shift = col_shift + cloc/usfac

        # If its only one row or column the mportift along that dimension has no
        # effect. Set to zero.
        if nr == 1:
            row_shift = 0

        if nc == 1:
            col_shift = 0

    rg00 = np.sum(np.abs(buf1ft)**2)
    rf00 = np.sum(np.abs(buf2ft)**2)
    error = 1.0 - np.abs(CCmax)**2/(rg00*rf00)
    error = np.sqrt(np.abs(error))
    diffphase = np.angle(CCmax)

    output=[error,diffphase,row_shift,col_shift]

    Nc,Nr = np.meshgrid(Nc,Nr)

    return output, Nc, Nr, peak_map

def dftups(in_arr,nor,noc,usfac,roff,coff):

    nr,nc=in_arr.shape

    # Compute kernels and obtain DFT by matrix products
    kernc=np.exp((-1j*2*np.pi/(nc*usfac))*( ifftshift(np.arange(0,nc)[:, np.newaxis]) - np.floor(nc/2) )*( np.arange(0,noc) - coff ))
    kernr=np.exp((-1j*2*np.pi/(nr*usfac))*( np.arange(0,nor)[:, np.newaxis] - roff )*( ifftshift(np.arange(0,nr)) - np.floor(nr/2)  ))

    out=np.dot(np.dot(kernr,in_arr),kernc)
    return out

def FTpad(imFT,outsize):

    Nin = np.array(imFT.shape)
    Nout = np.asarray(outsize)
    imFT = fftshift(imFT)
    center = np.floor(Nin/2)

    imFTout = np.zeros(outsize).astype('complex64')
    centerout = np.floor(Nout/2)

    cenout_cen = (centerout - center).astype(int)

    imFTout[slice(cenout_cen[0],cenout_cen[0]+Nin[0]), slice(cenout_cen[1],cenout_cen[1]+Nin[1])] = imFT
    imFTout = ifftshift(imFTout)*Nout[0]*Nout[1]/(Nin[0]*Nin[1])

    return imFTout

def main():
    oParser = argparse.ArgumentParser()
    oParser.add_argument('--threshold', '-th', default=5,help='')
    oArgs = oParser.parse_args()
    threshold = np.float32(oArgs.threshold)	# threshold

    default_shift = np.array([[45.3,28.2],[-28.9,-32.6],[-46.1,-28.5]])
    sArrFilePath = glob.glob('lena\\' + '/*lena*.jpg')
    
    # load master image
    sFileName = os.path.basename(sArrFilePath[0])
    sTmpFileName_master, sTmpExt = os.path.splitext(sFileName)
    master = cv2.imread('lena\\' + sTmpFileName_master + '.jpg',0)
    master = np.float32(master)/255*(2**16-1)

    # # adjust background level of each channel
    # rows,cols = master.shape
    # blv_ch0 = np.median(master[slice(int(rows/2)),slice(int(cols/2))])
    # blv_ch1 = np.median(master[slice(int(rows/2),rows),slice(int(cols/2))])
    # blv_ch2 = np.median(master[slice(int(rows/2)),slice(int(cols/2),cols)])
    # blv_ch3 = np.median(master[slice(int(rows/2),rows),slice(int(cols/2),cols)])

    # adj_arr = np.zeros_like(master).astype(np.float32)
    # adj_arr[slice(int(rows/2)),slice(int(cols/2))] = blv_ch0 - blv_ch0
    # adj_arr[slice(int(rows/2),rows),slice(int(cols/2))] = blv_ch1 - blv_ch0
    # adj_arr[slice(int(rows/2)),slice(int(cols/2),cols)] = blv_ch2 - blv_ch0
    # adj_arr[slice(int(rows/2),rows),slice(int(cols/2),cols)] = blv_ch3 - blv_ch0

    # master = master - adj_arr
    # master = master - np.min(master)

    master_crop = master[slice(800,1200),slice(800,1200)]
    
    i = 0

    for sFilePath in sArrFilePath[1:len(sArrFilePath)]:

        # load slave image
        sFileName = os.path.basename(sFilePath)
        sTmpFileName_slave, sTmpExt = os.path.splitext(sFileName)
        slave = cv2.imread('lena\\' + sTmpFileName_master + '.jpg',0)
        slave = np.float32(slave)/255*(2**16-1)

        rows,cols = slave.shape

        # # adjust background level of each channel
        # blv_ch0 = np.median(slave[slice(int(rows/2)),slice(int(cols/2))])
        # blv_ch1 = np.median(slave[slice(int(rows/2),rows),slice(int(cols/2))])
        # blv_ch2 = np.median(slave[slice(int(rows/2)),slice(int(cols/2),cols)])
        # blv_ch3 = np.median(slave[slice(int(rows/2),rows),slice(int(cols/2),cols)])

        # adj_arr = np.zeros_like(slave).astype(np.float32)
        # adj_arr[slice(int(rows/2)),slice(int(cols/2))] = blv_ch0 - blv_ch0
        # adj_arr[slice(int(rows/2),rows),slice(int(cols/2))] = blv_ch1 - blv_ch0
        # adj_arr[slice(int(rows/2)),slice(int(cols/2),cols)] = blv_ch2 - blv_ch0
        # adj_arr[slice(int(rows/2),rows),slice(int(cols/2),cols)] = blv_ch3 - blv_ch0

        # slave = slave - adj_arr
        # slave = slave - np.min(slave)
            
        # randomly shift slave image
        # M = np.float32([[1,0,10*np.random.rand(1)-5],[0,1,10*np.random.rand(1)-5]])
        M = np.float32([[1,0,default_shift[i,0]],[0,1,default_shift[i,1]]])
        slave = cv2.warpAffine(slave,M,(cols,rows))
        
        # print(M[0,2],M[1,2])
        i = i + 1
        
        slave_crop = slave[slice(800,1200),slice(800,1200)]
        
        # plot blurred images
        img = np.uint8((master_crop+slave_crop)/np.max((master_crop+slave_crop))*255)
        plt.figure(figsize=(12, 6))
        plt.subplot(1,2,1)
        
        label_text = 'row_shift=' + "{0:.2f}".format(M[0,2])
        bottomLeftCornerOfText = (10,30)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 0.8
        fontColor              = (255,255,255)
        thickness              = 1
        lineType               = cv2.LINE_AA

        cv2.putText(img, label_text, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

        label_text = 'col_shift=' + "{0:.2f}".format(M[1,2])
        bottomLeftCornerOfText = (10,60)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 0.8
        fontColor              = (255,255,255)
        thickness              = 1
        lineType               = cv2.LINE_AA

        cv2.putText(img, label_text, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

        plt.imshow(img,cmap='gray')
        plt.title(sTmpFileName_master + ' + ' + sTmpFileName_slave + ' (BLURRED)')
        plt.xlabel('x')
        plt.ylabel('y')
        
        start = time.time()

        # master&slave coregistration
        slave_shift, peak_map, col_shift, row_shift = fft_coreg(master_crop,slave_crop,slave)
        
        elapsed_time = time.time() - start
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

        # crop slave_shift
        slave_shift_crop = slave_shift[slice(800,1200),slice(800,1200)]

        # check image stacked
        img = np.uint8((master_crop+slave_shift_crop)/np.max((master_crop+slave_shift_crop))*255)
        plt.subplot(1,2,2)

        label_text = 'row_shift=' + "{0:.2f}".format(col_shift) + ' (err=' + "{0:.2f}".format(col_shift+M[0,2]) + ')'
        bottomLeftCornerOfText = (10,30)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 0.8
        fontColor              = (255,255,255)
        thickness              = 1
        lineType               = cv2.LINE_AA

        cv2.putText(img, label_text, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

        label_text = 'col_shift=' + "{0:.2f}".format(row_shift) + ' (err=' + "{0:.2f}".format(row_shift+M[1,2]) + ')'
        bottomLeftCornerOfText = (10,60)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 0.8
        fontColor              = (255,255,255)
        thickness              = 1
        lineType               = cv2.LINE_AA

        cv2.putText(img, label_text, 
            bottomLeftCornerOfText, 
            font,
            fontScale,
            fontColor,
            thickness,
            lineType)

        plt.imshow(img,cmap='gray')
        plt.title(sTmpFileName_master + ' + ' + sTmpFileName_slave + ' (COREGISTERED)')
        plt.xlabel('x')
        plt.ylabel('y')

    # plot peak row-wise/col-wise
    peak_map = np.roll(peak_map,-int(row_shift),axis=0)
    peak_map = np.roll(peak_map,-int(col_shift),axis=1)
    m,n = peak_map.shape
    peak_map = peak_map[slice(int(m/2-100),int(m/2+100)),slice(int(n/2-100),int(n/2+100))]
    plt.figure(figsize=(6, 6))
    img = np.uint8(peak_map/np.max(peak_map)*255)

    label_text = 'row_shift=' + "{0:.2f}".format(-col_shift)
    bottomLeftCornerOfText = (5,170)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.6
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = cv2.LINE_AA

    cv2.putText(img, label_text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)

    label_text = 'col_shift=' + "{0:.2f}".format(-row_shift)
    bottomLeftCornerOfText = (5,190)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.6
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = cv2.LINE_AA

    cv2.putText(img, label_text, 
        bottomLeftCornerOfText, 
        font,
        fontScale,
        fontColor,
        thickness,
        lineType)

    plt.imshow(img,cmap='gray',extent=[-100,99,99,-100])
    plt.title('estimated shift')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()

if __name__ == '__main__':

    main()