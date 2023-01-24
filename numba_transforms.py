import numpy as np
from numba import jit

@jit
def nn_scale_image(im, d,aspec=None):
    '''Uses a nearest neighbour method to scale an image by a ratio of d. If a different aspect ratio is desired, it can be specified
    im = the original image (3d numpy array)
    d = the ratio by which the width will be scaled
    aspec = the desired height-to-width ratio of the image's pixels. If none is specified, the original aspect ratio will be kept.
    '''
    (h,w,b)=np.shape(im)
    if aspec == None:
        m = int(np.floor(d*h))
        n = int(np.floor(d*w))
        dh = d
    else:
        m = int(np.floor(d*h*aspec))
        n = int(np.floor(d*w))
        dh = d*aspec
    scaled_im = np.zeros((m,n,b)).astype(np.uint8)
    for i in range(m):
        for j in range(n):
            k = int(np.round(i/dh))
            l = int(np.round(j/d))
            scaled_im[i,j,:]=im[k,l,:]
    return scaled_im

@jit
def bl_scale_image(im, d, aspec=None):
    '''Uses a bilinear method to scale an image by a ratio of d. If a different aspect ratio is desired, it can be specified
    im = the original image (3d numpy array)
    d = the ratio by which the width will be scaled (float)
    aspec = the desired height to width ratio of the images pixels. (float) If none is specified, the original aspect ratio will be kept
    '''
    (h,w,b)=np.shape(im)
    if aspec == None:
        m = int(np.floor(d*h))
        n = int(np.floor(d*w))
        dh = d
    else:
        m = int(np.floor(d*h*aspec))
        n = int(np.floor(d*w))
        dh = d*aspec
    padded_im = np.zeros((h+1,w+1,b))
    padded_im[:h,:w,:]=im
    scaled_im = np.zeros((m,n,b))
    for i in range(m):
        for j in range(n):
            k1 = int(np.floor(i/dh))
            l1 = int(np.floor(j/d))
            k2 = int(np.floor((i+1)/dh))
            l2 = int(np.floor((j+1)/d))
            if k1<k2:
                di = i-(dh*k1)
            else:
                di = 0
            if l1<l2:
                dj = j-(d*l1)
            else:
                dj = 0
            A = padded_im[k1:k1+2,l1:l1+2,:]
            scaled_im[i,j,:] = ((1-dj)*(((1-di)*A[0,0,:])+(di*A[0,1,:])))+(dj*(((1-di)*A[1,0,:])+(di*A[1,1,:])))
    return scaled_im.astype(np.uint8)

@jit
def nn_rotate_image(im, r, rad = True):
    '''Uses a nearest neighbour method to rotate an image by a r radians. If degrees are used instead, it can be specified
    im = the original image (3d numpy array)
    r = the rotation angle, counter clockwise, around the centre of the image (float)
    rad = True if r is in radians, False if it is degrees 
    '''
    if not(rad):
        r = np.radians(r)
    (h,w,b)=np.shape(im)
    m = int(np.floor(np.abs(w*np.sin(r)) + np.abs(h*np.cos(r))))
    n = int(np.floor(np.abs(w*np.cos(r)) + np.abs(h*np.sin(r))))
    rotated_im = np.zeros((m,n,b))
    rm = np.array([[np.cos(r), -np.sin(r)],[np.sin(r), np.cos(r)]])
    centre = np.array([[w/2],[h/2]])
    rotated_centre = np.array([[n/2],[m/2]])
    for i in range(m):
        for j in range(n):
            new_pos = np.dot(rm,np.array([[j],[i]])-rotated_centre)+centre
            if -0.5<new_pos[1,0]<h-1 and -0.5<new_pos[0,0]<w-1:
                k = int(np.round(new_pos[1,0]))
                l = int(np.round(new_pos[0,0]))
                rotated_im[i,j,:]=im[k,l,:]
    return rotated_im.astype(np.uint8)