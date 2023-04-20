import numpy as np
import os, re, cv2, sknw
from networkx import shortest_path
from scipy.signal import argrelmax
from scipy.interpolate import splprep
from scipy.interpolate import splev
from cellpose import utils
from skimage.morphology import skeletonize
import warnings
import copy

def create_centerline(m,outline):
    '''
    Finds the centerline of a mask
    
    Parameters
    ---------
    m = an opencv image (should have only one nonzero value)
    outline = outline of the mask (list of 2d points)
    
    Returns
    -------
    centerline = a 2d boolean array (True where the centerline is, False everywhere else)
    '''
    if np.shape(outline)[0] != 2:
        raise ValueError('Outline must be output from cellpose')
    # find the center of the mask
    center = np.array([np.sum(outline[0]),np.sum(outline[1])])/len(outline[0])
    
    # copy the mask, and crop to appropriate size
    large_mask = copy.deepcopy(m)
    mask = only_masks(large_mask)
    
    # find the topological skeleton
    unpruned_skel = padskel(mask)
    
    #find the outline of the cell
    poles = find_endpoints(np.transpose(outline),center)
    
    #centerline,length,pts,s = prune2(unpruned_skel,outline,mask,poles,sensitivity=5)
    #large_centerline = rev_embed_only_mask(large_mask,centerline)
    #return large_centerline

def only_masks(mask):
    '''Finds a bounding box of a mask and returns and crops that image to within that bounding box
    mask = mask in an image (2d numpy array)
    
    Returns:
    
    small_mask = a cropped image of the mask
    '''
    if len(np.shape(mask))>2:
        embed_mask=copy.deepcopy(mask[:,:,0])
    else:
        embed_mask=copy.deepcopy(mask)
    (hor,ver) = np.nonzero(embed_mask)
    l1 = min(hor)
    l2 = max(hor)+1
    k1 = min(ver)
    k2 = max(ver)+1
    if len(np.shape(mask))>2:
        small_mask = mask[l1:l2,k1:k2,:]
    else:
        small_mask = mask[l1:l2,k1:k2]
    return small_mask

def rev_embed_only_mask(mask,image):
    '''embeds a boolean array with the same dimensions as the output of only_mask into the original mask
    mask = the original mask(boolean array)
    image = an image with the same dimensions as the ouput of only_mask (boolean array)
    '''
    (hor,ver) = np.nonzero(mask)
    c = np.array([min(hor),min(ver)])
    (hcoord,vcoord) = np.nonzero(image)
    embed_im = np.zeros(np.shape(mask))
    for coord in zip(hcoord,vcoord):
        new_coord = np.array(coord)+c
        embed_im[new_coord[0],new_coord[1]] = 1
    return embed_im

def padskel(mask):
    '''
    Runs skimage.morphology.skeletonize on a padded version of the mask, to avoid errors.
    
    Parameters
    ----------
    mask = an opencv image
    
    Returns
    -------
    skel = a skeleton (boolean array)
    '''
    mask = cv2.copyMakeBorder(mask,20,20,20,20,cv2.BORDER_CONSTANT,None,value=0)>0 #add a border to the skel
    skel = skeletonize(mask)
    skel = skel[20:np.shape(skel)[0]-20,20:np.shape(skel)[1]-20]
    return skel

def intersection(line1,line2,width1=3,width2=3):
    '''
    Find if two lines intersect, or nearly intersect
    
    Parameteres
    -----------
    line1,line2=boolean arrays with 'True' values where the line exists
    width1,width2=amount (in pixels) the line should be dilated to see if a near intersection occurs
    '''
    check = np.sum(line1+line2==2)>0
    if check:
        return True
    else:
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(width1,width1))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(width2,width2))
        dilated1 = cv2.dilate(line1.astype(np.uint8),kernel1,iterations=1)
        dilated2 = cv2.dilate(line2.astype(np.uint8),kernel2,iterations=1)
        check = np.sum(dilated1+dilated2==2)>0
        return check

def bool_sort(array,axis=True):
    '''
    Sort the "True" values in a boolean array, starting from left to right, then top to bottom. If top is true, then start from top to bottom, then
    then left to right.
    Parameters
    ---------
    array = a boolean array to be sorted
    axis = which axis is whose lowest values are put first, default True. If True put the 0 (top-bottom) axis first, if False put the 1 (left-right) axis first.
    '''
    if axis:
        output = np.transpose(np.where(array))
        output = output[output[:,1].argsort()]
        output = output[output[:,0].argsort(kind='mergesort')]
    else:
        output = np.transpose(np.where(array))
        output = output[output[:,0].argsort()]
        output = output[output[:,1].argsort(kind='mergesort')]
    return output

def pts_to_img(pts,base):
    '''converts a list of points to a binary image, with the dimensions of the skeleton'''
    path_in_img=np.zeros(np.shape(base)).astype(bool)
    for pt in pts:
        pt = pt.astype(np.uint16)
        path_in_img[pt[0],pt[1]]=True
    return path_in_img 

def order_by_NN(points,source=None,destination=None,thresh=5):
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
    NN = posl[0]
    rmax = np.linalg.norm(posl[0]-point)
    for pos in posl:
        r = np.linalg.norm(pos-point)
        if r<rmax:
            NN = pos
            rmax = r
    return NN,rmax

def in_mask (path,mask):
    '''
    Returns values of a path which are in a mask
    Parameters:
    ---------
    path = numpy array consisting of coordinates in the mask
    mask = numpy boolean array, showing the mask
    '''
    #first remove all values that are not in the picture
    not_in_img =[not(pt[0]<np.shape(mask)[0]-1 and pt[0]>=0 and pt[1]<np.shape(mask)[1]-1 and pt[1]>=0) for pt in path]
    if False in not_in_img:
        not_in_img = np.where(not_in_img)
        path = np.delete(path,not_in_img,0)
    #next remove all values which are not in the mask
    not_in_mask = []
    for i in range (0,len(path)):
        pt = path[i]
        if mask[pt[0],pt[1]]:
            continue
        else:
            not_in_mask.append(i)
    return np.delete(path,not_in_mask,0)

def crop_centerline(centerline_path,skel,outline,true_starts,true_ends,crop,axis):
    '''Helper function. Crops a proportion of the points (determined by the crop parameter) from the ends of a centerline with a false pole
    Paramters:
    --------
    centerline_path = list of points (numpy arrays) on a line in order of the arclength parametrization, initiated at the start node
    '''
    crop_length=round(len(centerline_path)*crop)
    image = pts_to_img(centerline_path,skel)
    (m,n)=np.shape(image)
    if true_starts == []:
        if axis:
            start_centerline = image[:,0:n//2]
            start_outline = outline[:,0:n//2]
        else:
            start_centerline=image[0:m//2,:]
            start_outline=outline[0:m//2,:]
        if intersection(start_centerline,start_outline):
            centerline_path= centerline_path[crop_length:]

    if true_ends == []:
        if axis:
            end_centerline = image[:,n//2:n]
            end_outline = outline[:,n//2:n]
        else:
            end_centerline = image[m//2:m,:]
            end_outline = outline[m//2:m,:]
        if intersection(end_centerline,end_outline): 
            centerline_path = centerline_path[:len(centerline_path)-crop_length]
    return centerline_path

def find_splines(points,true_starts,true_ends,start_pole,end_pole,k_thresh,crop):
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
    if true_starts ==[]:
        points = [np.array([start_pole[1],start_pole[0]])] + points
    if true_ends ==[]:
        points = points + [np.array([end_pole[1],end_pole[0]])]
    tck,U=splprep(np.transpose(points))
    s = np.linspace(0,1,1000)
    [ys,xs]= splev(s,tck)
    #calculate the curvature of the spline
    v = np.transpose(splev(s,tck,der=1))
    a = np.transpose(splev(s,tck,der=2))
    k=abs(np.cross(v,a)/np.array([np.linalg.norm(V)**3 for V in v]))
    # check if the curvature exceeds a threshold
    if np.any(k>k_thresh):
        warnings.warn('curvature threshold exceeded. additional pruning executed in response')
        crop_length=round(len(points)*crop)
        if true_starts == []:
            points= points[crop_length:]
        if true_ends == []:
            points = points[:len(points)-crop_length]
        #create a new spline
        if true_starts ==[]:
            points = [np.array([start_pole[1],start_pole[0]])] + points
        if true_ends ==[]:
            points = points + [np.array([end_pole[1],end_pole[0]])]
        #remove repeated points
        diff = np.diff(points,axis=0)==0
        repeated = np.where(np.array([pt[0] and pt[1] for pt in diff]))
        points = list(np.delete(points,repeated,axis=0))
        #create a new spline
        tck,U=splprep(np.transpose(points))
        s = np.linspace(0,1,1000)
        [ys,xs]= splev(s,tck)
        v = np.transpose(splev(s,tck,der=1))
        a = np.transpose(splev(s,tck,der=2))
        k=abs(np.cross(v,a)/np.array([np.linalg.norm(V)**3 for V in v]))
        if np.any(k>k_thresh):
            warnings.warn('Curvature exceeds threshold')
    return [xs,ys],s

def prune2(skel,outline,mask,poles,sensitivity=5,crop=0.1,k_thresh=0.2):
    '''Creates a centerline of the cell from the skeleton, removing branches and extending the centerline to the poles
    skel = topological skeleton of the cell (boolean array)
    outline = outline of the cell (boolean array)
    mask = mask of the cell (booelan array)
    poles = two poles of the cell. Ordered as [desired start pole of the centerline, desired end pole of the centerline]
    sensitivity = distance to a pole (in pixels) which a branch of the skeleton must be to be considered to "reach" that pole. Default is 5 pixels
    crop = proportion of the centerline to crop at a false pole (an end of a branch which intersects the outline not at a pole). Default is 0.1
    k_thresh = curvature (radius of curvature^{-1}) threshold for the final spline. above this the centerline will be cropped for a second time to ensure
    the ends of the centerline don't have abnormally high curvature. Default = 0.2 (corresponding to a radius of curvature of 5 pixels)
    '''
        
    if np.any(np.isnan(np.array(poles))):
        raise ValueError('Poles must be numerical values, not np.NaN')
    #initializing parameters
    m = np.shape(skel)[0]
    n = np.shape(skel)[1]
    axis = m<n
    poles = list(poles)
    graph = sknw.build_sknw(skel) #creating the graph from the skeleton
    all_paths=shortest_path(graph) #shortest possible paths between the nodes in the skeleton
    nodes = graph.nodes()
    [start_pole,end_pole] = poles
    paths = []
    centerlines=[]
    lengths=[]
    
    #initializing suitable starting and ending positions for the skeleton
    if axis:
        start_nodes = list([i for i in nodes if nodes[i]['o'][1]<n//2])
        end_nodes= list([i for i in nodes if nodes[i]['o'][1]>n//2])
    else:
        start_nodes = list([i for i in nodes if nodes[i]['o'][0]<m//2])
        end_nodes= list([i for i in nodes if nodes[i]['o'][0]>m//2])
    
    # if there are some nodes close to the poles, check only those nodes
    true_starts = [i for i in start_nodes if np.linalg.norm(nodes[i]['o']-np.array([start_pole[1],start_pole[0]]))<sensitivity]
    true_ends = [i for i in end_nodes if np.linalg.norm(nodes[i]['o']-np.array([end_pole[1],end_pole[0]]))<sensitivity]
    if true_starts != []:
        start_nodes = true_starts
    if true_ends !=[]:
        end_nodes = true_ends
    if start_nodes == [] or end_nodes == []:
        raise ValueError('skeleton is on only one side of the cell')
    # take all paths between starting and ending nodes
    for b in  start_nodes:
        for e in end_nodes:
            path = all_paths[b][e]
            paths.append(path)
    if len(paths) == 1:
        path = paths[0]
        #initializing centerline
        centerline_path = []
        #calling points from the graph
        edges = [(path[i],path[i+1]) for i in range (0,len(path)-1)]
        for (b,e) in edges:
            edge = graph[b][e]['pts']
            centerline_path = centerline_path + list(edge)
        centerline_path=order_by_NN(centerline_path,nodes[path[0]]['o'],nodes[path[-1]]['o'],max(m,n)/2)
        if len(centerline_path)==0:
            raise ValueError('Skeleton has been erased')
        if np.linalg.norm(np.array([centerline_path[0][1],centerline_path[0][0]])-end_pole) < np.linalg.norm(np.array([centerline_path[0][1],centerline_path[0][0]])-start_pole):
            centerline_path.reverse()
        centerline_path = crop_centerline(centerline_path,skel,outline,true_starts,true_ends,crop,axis)
        if len(centerline_path)<=5:
            raise ValueError('Skeleton has been erased')
        [xs,ys],u = find_splines(centerline_path,true_starts,true_ends,start_pole,end_pole,k_thresh,crop)
        path = np.round(np.transpose(np.array([ys,xs]))).astype(np.uint32)
        path = in_mask(path,mask)
        length = arc_length([xs,ys])
        pruned_skel = pts_to_img(path,skel)
        return pruned_skel, length, [xs,ys],np.linspace(0,1,100)
    #convert paths (lists of nodes) to centerlines (lists of points)
    for path in paths:
        #initializing centerline
        centerline_path = []
        #calling points from the graph
        edges = [(path[i],path[i+1]) for i in range (0,len(path)-1)]
        for (b,e) in edges:
            edge = graph[b][e]['pts']
            centerline_path = centerline_path + list(edge)
        #convert path to binary image
        if len(centerline_path)==0:
            raise ValueError('Skeleton has been erased')
        #crop the centerline, if it has a false pole
        centerline_path=order_by_NN(centerline_path,nodes[path[0]]['o'],nodes[path[-1]]['o'],max(m,n)/2)
        if np.linalg.norm(np.array([centerline_path[0][1],centerline_path[0][0]])-end_pole) < np.linalg.norm(np.array([centerline_path[0][1],centerline_path[0][0]])-start_pole):
            centerline_path.reverse()
        centerline_path = crop_centerline(centerline_path,skel,outline,true_starts,true_ends,crop,axis)
        #find the length of each centerline
        length = len(centerline_path)
        # add to the list of centerlines and lengths
        centerlines.append(centerline_path)
        lengths.append(length)
    #choose the longest centerline
    max_index=lengths.index(max(lengths))
    centerline_path=centerlines[max_index]
    if len(centerline_path)<=5:
        raise ValueError('Skeleton has been erased')
    [xs,ys],u = find_splines(centerline_path,true_starts,true_ends,start_pole,end_pole,k_thresh,crop)
    path = np.round(np.transpose(np.array([ys,xs]))).astype(np.uint32)
    path = in_mask(path,mask)
    pruned_skel = pts_to_img(path,skel)
    length = arc_length([xs,ys])
    return pruned_skel, length, [xs,ys], u

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

def find_endpoints(outline,center):
    '''
    Finds the expected endpoints of the ridgeline of the given outline and center
    Parameters
    ----------
    [X,Y] = list of x and y coordinates of the outline of the cell
    center = center of the cell
    
    Returns
    --------
    pole1, pole2 = expected endpoints of the ridgeline 
    '''
    
    #the outline must be horizontal for this, so if it is vertical we transpose x and y
    [x,y] = outline
    vert = np.max(y)-np.min(y)>np.max(x)-np.min(x) 
    if vert:
        [y,x] = outline
    #find the distance to the center, and the local maxima of this distance function
    rad = dist_function(x,y,center)
    maxr = argrelmax(rad,order = 10,mode='wrap')[0]
    
    #divide the cell vertically at the center
    d = np.floor(center[0]).astype(np.uint8)
    
    #set the radius of any point not in the right half to 0
    right=(x>d).astype(np.uint8)*rad
    
    # set the radius of any point not in the left half to 0
    left=(x<=d).astype(np.uint8)*rad
    
    # find the center of mass of each local maxima, with the mass of each point being it's distance to the center
    left_pole = np.array([np.sum(outline[0][maxr]*left[maxr]),np.sum(outline[1][maxr]*left[maxr])])/np.sum(left[maxr])
    right_pole = np.array([np.sum(outline[0][maxr]*right[maxr]),np.sum(outline[1][maxr]*right[maxr])])/np.sum(right[maxr])
    
    #if the outlne was originally vertical we transpose the poles back to their correct order
    return left_pole, right_pole

def arc_length(pts):
    '''
    Find the arclength of a curve given by a set of points
    Paramters
    --------
    pts = array-like coordinates [x1,x2,....]
    '''
    pts = np.array(np.transpose(pts))
    lengths = []
    for i in range (1,len(pts)):
        length = np.linalg.norm(pts[i] - pts[i-1])
        lengths.append(length)
    return np.sum(lengths)