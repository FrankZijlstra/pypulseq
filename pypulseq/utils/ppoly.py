import numpy as np

from math import comb, inf
from scipy.interpolate import PPoly

def ppoly_copy(pp, shallow_c=False, shallow_x=False):
    c = pp.c
    x = pp.x
    
    if not shallow_c:
        c = c.copy()
    if not shallow_x:
        x = x.copy()
        
    return PPoly(c, x, extrapolate=pp.extrapolate, axis=pp.axis)

def ppoly_interval(pp, interval=[-inf, inf], shallow=True):
    i1 = np.searchsorted(pp.x, interval[0])
    i2 = np.searchsorted(pp.x, interval[0], side='right')
    
    c = pp.c[i1:i2-1]
    x = pp.x[i1:i2]
    
    if not shallow:
        c = c.copy()
        x = x.copy()
    
    return PPoly(c, x, extrapolate=pp.extrapolate, axis=pp.axis)
    

def ppoly_nth_moment(pp, n=0, reference_point=0):
    if pp == None:
        return None
    
    if n > 0:
        pp = ppoly_copy(pp, shallow_c=True)
    
    for i in range(n):
        # Calculate coefficients for pp*t
        order = pp.c.shape[0]+1
        c_new = np.zeros((order, pp.c.shape[1]), dtype=pp.c.dtype)
        x = pp.x[:-1] - reference_point

        for i in range(order):
            if i < order - 1:
                c_new[i] += pp.c[i]
            if i > 0:
                c_new[i] += pp.c[i-1]*x
        
        pp.c = c_new
        pp.c[-1] -= pp(np.clip(reference_point, pp.x[0], pp.x[-1])) 
    
    return pp.antiderivative()


def ppoly_split(pp, t_split, mode='', extrapolate=True):
    if pp == None:
        return None
    
    if len(t_split) > 0:
        pp = ppoly_copy(pp)
    
    for t in t_split:
        i = np.searchsorted(pp.x, t)
        v = pp(np.clip(t, pp.x[0], pp.x[-1]))

        # Add new break(s) if split timepoint is after the last break
        if i >= pp.x.shape[0]:
            if not extrapolate:
                return pp
            
            c_insert = np.zeros((pp.c.shape[0],2), dtype=pp.c.dtype)
            c_insert[-1] = v
            pp.c = np.append(pp.c, c_insert, axis=1)
            pp.x = np.append(pp.x, [t,t])
                
        # Split the polynomial segment at the split timepoint if necessary
        elif i > 0 and pp.x[i] != t:
            shift = t - pp.x[i-1]
            c_insert = np.zeros(pp.c.shape[0], dtype=pp.c.dtype)
    
            order = pp.c.shape[0]
            for j in range(order):
                for k in range(j+1):
                    c_insert[j] += comb(order-k-1,j-k) * shift**(j-k) * pp.c[k,i-1]
    
            pp.c = np.insert(pp.c, i, c_insert, axis=1)
            pp.x = np.insert(pp.x, i, t)
        
        if mode == 'refocus':
            pp.c[-1,i:] -= 2*v
        elif mode == 'reset':
            pp.c[-1,i:] -= v
            
    return pp