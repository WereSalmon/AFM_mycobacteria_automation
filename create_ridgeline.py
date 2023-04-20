import numpy as np
import os, re, cv2, sknw
from networkx import shortest_path
from networkx import number_connected_components
from scipy.signal import argrelmax
from scipy.interpolate import splprep
from scipy.interpolate import splev
from cellpose import utils
from skimage.morphology import skeletonize
import warnings
import copy

def create_ridgeline(m,outline,center):
    '''
    Finds the ridgeline of a cell
    
    Parameters
    ---------
    m = the mask from cellpose (should have only one nonzero value)
    outline = outline of the mask (list of 2d points) from cellpose
    center = center of the mask
    
    Returns
    -------
    ridgeline = a 2d boolean array (True where the centerline is, False everywhere else)
    left_pole = the first endpoint of the cell, returned by endpoints
    right_pole = the second endpoint of the cell, returned by endpoints
    max_k = the maximum curvature of the spline
    '''
    if np.shape(outline)[1] != 2:
        raise ValueError('Outline must be output from cellpose')
        
    # copy the mask
    mask = copy.deepcopy(m)
    if len(np.shape(mask))>2.5:
        mask = mask[:,:,0]
    
    # find the topological skeleton
    skel = padskel(mask)
    
    #find the outline of the cell
    left_pole,right_pole = endpoints(outline,center)
    
    ridge,max_k = ridgeline (skel, outline, mask, center, left_pole, right_pole,k_thresh=0.2)
    return ridge, left_pole, right_pole, max_k

### Necessary helper functions. If interested only in called functions see the bottom of the page

def order_by_NN(points,source=None,destination=None,thresh=5):
    '''
    Orders a list of points by their nearest neighbours
    
    Parameters
    ---------
    Points = a list of points (numpy arrays)
    source (optional) = the point to start sorting with (default to the first poin tin the list)
    destination (optional) = the desired final point in the sorted list (default to None, will not check which point is last in the sorted list) 
    thresh (optional) = the desired maximum distance two points can be apart (default is 5)
    
    Returns:
    -------
    ordered = a sublist of points, ordered such that each point is adjacent to it's two nearest neighbours
    '''
    
    ordered = []
    if source is None:
        source = points[0]
    ordered.append(source)
    pos = source

    while points != []:
        next_pos,r = find_NN(pos,points)
        if r>thresh:
            warnings.warn('Sorting terminated early when points exceeded allowable distance')
            break
        ind =np.where(np.array([np.all(p == next_pos) for p in points]))
        points = list(np.delete(np.array(points),ind,axis=0))
        ordered.append(next_pos)
        pos = next_pos
    if np.linalg.norm(ordered[-1]-destination)>2 and not destination is None:
        warnings.warn('path did not reach destination')
    return ordered

def find_NN(point,posl):
    ''' Finds the closest point to a given point in a list
    point = reference point (numpy array)
    posl = a list of points (numpy arrays)
    
    Returns
    NN = the nearest neighbour to the reference point in posl
    rmin = the distance between NN and the reference point
    '''
    NN = posl[0]
    rmin = np.linalg.norm(posl[0]-point)
    for pos in posl:
        r = np.linalg.norm(pos-point)
        if r<rmin:
            NN = pos
            rmin = r
    return NN,rmin

def dist_function(x,y,c):
    '''
    Find the distance function of a 2d curve for a point
    
    Parameters
    ---------
    x,y = coordinates of the curve
    c = 2d point
    
    Returns
    -------
    r = a 1D array containing the distance from to each point on the curve to c
    '''
    dists = np.transpose(np.array([x,y]))-c
    return np.array([np.linalg.norm(v) for v in dists])

def longest_path(graph,start,end):
    '''Find the path in the skeleton with the most pixels between a given disjoint start and end points
    
    Parameters:
    graph = sknw graph, built from the desired skeleton
    start = list of integers (start nodes)
    end = list of integers (end nodes)
    
    Returns:
    longest_path = list of np.arrays of integers, representing the longest path
    
    '''
    
    # initialize longest path
    longest_path = []
    start_node = -1
    end_node = -1
    
    for i in start:
        for j in end:
            path = shortest_path(graph,i,j)
            edges = [(path[i],path[i+1]) for i in range (0,len(path)-1)]
            points = []
            for (b,e) in edges:
                points = points + list(graph[b][e]['pts'])
            if len(longest_path)<len(points):
                longest_path = points
                start_node = i
                end_node = j
    return longest_path, start_node,end_node

def find_spline(points):
    '''
    Helper function. Return spline of a centerline based on a set of ordered points.
    Parameters
    --------
    points = list of points (numpy arrays) on a line in order of the arclength parametrization, initiated at the start node
    '''
    s = np.linspace(0,1,1000)
    
    # remove repeated points
    diff = np.diff(points,axis=0)==0
    repeated = np.where(np.array([pt[0] and pt[1] for pt in diff]))
    points = list(np.delete(points,repeated,axis=0))
    tck,U=splprep(np.transpose(points))
    s = np.linspace(0,1,1000)
    [ys,xs]= splev(s,tck)

    #calculate the curvature of the spline
    v = np.transpose(splev(s,tck,der=1))
    a = np.transpose(splev(s,tck,der=2))
    k=abs(np.cross(v,a)/np.array([np.linalg.norm(V)**3 for V in v]))
    
    # find the maximum curvature of the spline
    max_k = np.max(k)
    return [xs,ys],s, max_k

def spline_to_img(pts,mask):
    '''Convert spline to image, ignoring all points outside a given region
    pts = list of 2-arrays
    mask = 2d binary image, representing the region where the spline can exist
    
    Returns:
    img = an image of a single, connected, spline
    '''
    (m,n) = np.shape(mask)
    img=np.zeros((m,n))
    for pt in pts:
        pt = np.floor(pt).astype(int)
        if (pt[0] in range(n)) and (pt[1] in range(m)) and mask[pt[1],pt[0]]:
            img[pt[1],pt[0]]=1
    
    graph = sknw.build_sknw(img)
    return img 

### Functions called directly in create_ridgeline.

def padskel(mask):
    '''
    Returns the skeleton using the Zhang-Suen Algorithm on a padded version of the mask, to avoid errors.
    
    Parameters
    ----------
    mask = the mask returned by cellpose
    
    Returns
    -------
    skel = a skeleton (boolean array)
    '''
    mask = cv2.copyMakeBorder(mask,20,20,20,20,cv2.BORDER_CONSTANT,None,value=0)>0 #add a border to the mask
    skel = skeletonize(mask)
    skel = skel[20:np.shape(skel)[0]-20,20:np.shape(skel)[1]-20]
    return skel

def endpoints(outline,center):
    '''
    Finds the expected endpoints of the ridgeline of the given outline and center
    Parameters
    ----------
    outline = outline of the cell from cellpose (list of points)
    center = center of the cell
    
    Returns
    --------
    pole1, pole2 = expected endpoints of the ridgeline 
    '''
    
    # create new variable out, which is the the x and y coordinates of the outline as separate 1d arrays
    out = np.transpose(outline)
    
    #the outline must be horizontal for this, so if it is vertical we transpose x and y
    [x,y] = out
    vert = np.max(y)-np.min(y)>np.max(x)-np.min(x) 
    if vert:
        [y,x] = out
        center = np.flip(center)
    #find the distance to the center, and the local maxima of this distance function
    rad = dist_function(x,y,center)
    maxr = argrelmax(rad,order = 10,mode='wrap')[0]
    
    #divide the cell vertically at the center
    d = np.floor(center[0]).astype(int)
    
    #set the radius of any point not in the right half to 0
    right=(x>d).astype(np.uint8)*rad
    
    # set the radius of any point not in the left half to 0
    left=(x<=d).astype(np.uint8)*rad
    
    # find the center of mass of each local maxima, with the mass of each point being it's distance to the center
    left_pole = np.array([np.sum(out[0][maxr]*left[maxr]),
                          np.sum(out[1][maxr]*left[maxr])])/np.sum(left[maxr])
    right_pole = np.array([np.sum(out[0][maxr]*right[maxr]),
                           np.sum(out[1][maxr]*right[maxr])])/np.sum(right[maxr])
    
    # check that poles are non-zero, numerical values
    if (np.linalg.norm(right_pole)==np.nan 
        or np.linalg.norm(left_pole)==np.nan):
        raise ValueError('One or more poles are not numerical values')
    if (np.linalg.norm(right_pole)==0 
        or np.linalg.norm(left_pole)==0):
        raise ValueError('One or more poles are zero. '
                         + 'This indicates an error in calculations')
    return left_pole, right_pole

def ridgeline(skel, outline, mask, center, lpole, rpole,k_thresh=0.2):
    '''Creates a ridgeline of the cell extending the central section of the topological skeleton to the given endpoints
    
    Parameters
    skel = topological skeleton of the cell (boolean array)
    outline = outline of the cell from cellpose (list of points)
    mask = mask of the cell from cellpose (booelan array)
    center = center of the cell
    lpole = the first endpoint of the cell, returned by find_endpoints
    rpole = the second endpoint of the cell, returned by find_endpoints
    k_thresh = threshold for curvature in the final ridgeline
    
    Returns:
    ridge = ridge of the cell (boolean array)
    max_k = the maximum curvature of the spline
    '''

    [x,y] = np.transpose(outline)
    # if the cell is vertical, flip it to be horizontal
    vert = np.max(y)-np.min(y)>np.max(x)-np.min(x) 
    if vert:
        skel = np.transpose(skel)
        mask = np.transpose(mask)
        [y,x] = np.transpose(outline)
        center = np.flip(center)
        rpole = np.flip(rpole)
        lpole = np.flip(lpole)
    
    # determine where we will divide the cell in two
    d = np.floor(center[0]).astype(np.uint8)
    
    # create graph from skeleton
    graph = sknw.build_sknw(skel)
    nodes = graph.nodes()
    
    #determine which half each node is in
    endleft = [i for i in nodes if nodes[i]['o'][1]<=d]
    endright = [i for i in nodes if nodes[i]['o'][1]>d]
    if endleft == [] or endright == []:
        return skel, np.nan
        raise ValueError('skeleton is on only one side of the cell')
    
    # find the longest path (in the image) between the nodes on the left and nodes on the right
    path,start_node,end_node = longest_path(graph,endleft,endright)
    
    # ensure path is properly ordered
    path = order_by_NN(path,nodes[start_node]['o'],nodes[end_node]['o'])
    if len(path)==0:
        raise ValueError('Skeleton has been erased')

    # crop 10% off the start and end of the path
    crop_10 = np.floor(len(path)*0.1).astype(int)
    path_cropped = copy.deepcopy(path[crop_10:-crop_10])
    if len(path_cropped)<=5:
        raise ValueError('Skeleton has been erased')
    
    # fit a spline to our path, including the start and end points
    ridge_path = [np.flip(lpole)]+path_cropped+[np.flip(rpole)]
    [xs,ys],s,max_k = find_spline(ridge_path)
    
    # if spline has unusually high curvature crop 10% off of the start and end of our path again
    if max_k>k_thresh:
        crop_19 = np.floor(len(path_cropped)*0.1).astype(int)
        path_cropped_cropped = copy.deepcopy(path_cropped[crop_19:-crop_19])
        ridge_path_cropped = [np.flip(lpole)]+path_cropped_cropped+[np.flip(rpole)]
        [xs_cropped,ys_cropped], s, max_k_cropped = find_spline(ridge_path_cropped)
        if max_k_cropped<max_k:
            [xs,ys,s,max_k] = [xs_cropped,ys_cropped,s,max_k_cropped]
    
    # if spline still has high curvature, warn user
    if max_k>k_thresh:
        warnings.warn('Curvature exceeds threshold')
    
    # convert spline into image
    pts = np.transpose([xs,ys])
    ridge = spline_to_img(pts,mask)
    
    # check if ridgeline is connected
    ridge_graph = sknw.build_sknw(ridge)
    if number_connected_components(ridge_graph)>1.5:
        raise ValueError('Ridgeline is not connected')
        
    # check if ridgeline has been erased
    if np.count_nonzero(ridge)< (np.max(y)-np.min(y))/2:
        raise ValueError('Ridgeline is too short (' 
                         + str(np.count_nonzero(ridge)) + ' pixels long)')
    
    # if ridgeline is verticle, flip the image so that output has correct orientation
    if vert:
        ridge = np.transpose(ridge)
    return ridge.astype(bool), max_k