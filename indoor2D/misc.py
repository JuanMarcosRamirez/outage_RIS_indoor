#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:08:26 2023

@author: juan
"""

import numpy as np

def single_reflection_coordinate(a, b, x1, y1, x2, y2, wall_side):
    if ((wall_side == 'left')|(wall_side == 'right')):
        if (wall_side == 'left'):
            xo = 0
        if (wall_side == 'right'):
            xo = a
        x = np.array([x1,x2])
        y = np.array([y1,y2])
        ymin = y[np.argmin(y)]
        ydif = np.absolute(y1 - y2)
        ym   = ydif * np.absolute(xo - x[np.argmin(y)]) / (np.absolute(xo - x1) + np.absolute(xo - x2))
        yo   = ymin + ym
    if ((wall_side == 'upper')|(wall_side == 'bottom')):
        if (wall_side == 'upper'):
            yo = b
        if (wall_side == 'bottom'):
            yo = 0
        x = np.array([x1,x2])
        y = np.array([y1,y2])
        xmin = x[np.argmin(x)]
        xdif = np.absolute(x1 - x2)
        xm   = xdif * np.absolute(yo - y[np.argmin(x)]) / (np.absolute(yo - y1) + np.absolute(yo - y2))
        xo   = xmin + xm
    return xo, yo

def reflection_coordinates(Lx, Ly, x_Tx, y_Tx, x_Rx, y_Rx):
    xr = np.zeros(4)
    yr = np.zeros(4)
    xr[0], yr[0] = single_reflection_coordinate(Lx, Ly, x_Tx, y_Tx, x_Rx, y_Rx,'left')
    xr[1], yr[1] = single_reflection_coordinate(Lx, Ly, x_Tx, y_Tx, x_Rx, y_Rx,'right')
    xr[2], yr[2] = single_reflection_coordinate(Lx, Ly, x_Tx, y_Tx, x_Rx, y_Rx,'upper')
    xr[3], yr[3] = single_reflection_coordinate(Lx, Ly, x_Tx, y_Tx, x_Rx, y_Rx,'bottom')
    return xr, yr

def reflection_angles(x_Tx, y_Tx, x_Rx, y_Rx, xr, yr):
    theta_i = np.zeros(4)
    theta_i[0] = np.arctan(np.absolute(yr[0] - y_Tx)/np.absolute(x_Tx - xr[0]))
    theta_i[1] = np.arctan(np.absolute(yr[1] - y_Tx)/np.absolute(xr[1] - x_Tx))
    theta_i[2] = np.arctan(np.absolute(xr[2] - x_Tx)/np.absolute(yr[2] - y_Tx))
    theta_i[3] = np.arctan(np.absolute(xr[3] - x_Tx)/np.absolute(y_Tx - yr[3]))
    return theta_i

class line_equation_coefficients:
    def __init__(self, x_start, x_end, y_start, y_end, type_d):
        if (type_d == 'single'):
            if ((x_end - x_start) != 0):
                self.m = np.divide(y_end - y_start, x_end - x_start)
            else:
                self.m = np.array(1e12)
        if (type_d == 'multiple'):
            self.m = np.divide(y_end - y_start, x_end - x_start)
            self.m[np.isinf(self.m)] = 1e12       
        self.c = y_end - self.m * x_end
        self.B = np.ones(self.m.shape)
        self.A = np.array(- self.m)
        self.C = np.array(- self.c)
        self.d = np.sqrt(np.power(x_end - x_start, 2) + np.power(y_end - y_start, 2))
        
def dist_to_obstructions(x_1, y_1, A, B, C):
    num = np.absolute(A*x_1 + B*y_1 + C)
    den = np.sqrt(A**2 + B**2)
    distance = num / den
    return distance

def blocked_rays_main_paths(x, y, x1, x2, y1, y2, r, DS):
    minx = np.min([x1, x2])
    maxx = np.max([x1, x2])
    miny = np.min([y1, y2])
    maxy = np.max([y1, y2])
    D3 = (((x >= minx) & (x <= maxx)) & ((y >= miny) & (y <= maxy))) & (DS <= r)               
    return D3

def fcombine(wall, RIS_type, x_RIS, y_RIS, x_Tx, y_Tx, x_Rx, y_Rx, alphaRIS, alphaTx, alphaRx):
    if (wall == 'bottom'):
        x_center = np.absolute(x_RIS[-1] + x_RIS[0] + 2 * (x_RIS[1] - x_RIS[0]))/2
        y_center = 0
        # print(x_center, y_center)
        theta1 =  np.arctan(np.abs(x_Tx - x_center) /np.abs(y_Tx - y_center)) - np.arctan(np.abs(x_Tx - x_RIS) /np.abs(y_Tx - y_RIS))
        theta2 =  np.arctan(np.abs(x_Tx - x_RIS) /np.abs(y_Tx - y_RIS))
        theta3 =  np.arctan(np.abs(x_Rx - x_RIS) /np.abs(y_Rx - y_RIS))
        theta4 =  np.arctan(np.abs(x_Rx - x_center) /np.abs(y_Rx - y_center)) - np.arctan(np.abs(x_Rx - x_RIS) /np.abs(y_Rx - y_RIS))   

    if (wall == 'upper'):
        x_center = np.absolute(x_RIS[-1] + x_RIS[0] + 2 * (x_RIS[1] - x_RIS[0]))/2
        y_center = y_RIS[0]
        # print(x_center, y_center)
        theta1 =  np.arctan(np.abs(x_Tx - x_center) /np.abs(y_Tx - y_center)) - np.arctan(np.abs(x_Tx - x_RIS) /np.abs(y_Tx - y_RIS))
        theta2 =  np.arctan(np.abs(x_Tx - x_RIS) /np.abs(y_Tx - y_RIS))
        theta3 =  np.arctan(np.abs(x_Rx - x_RIS) /np.abs(y_Rx - y_RIS))
        theta4 =  np.arctan(np.abs(x_Rx - x_center) /np.abs(y_Rx - y_center)) - np.arctan(np.abs(x_Rx - x_RIS) /np.abs(y_Rx - y_RIS))

    if (wall == 'left'):
        x_center = 0
        y_center = np.absolute(y_RIS[-1] + y_RIS[0] + 2 * (y_RIS[1] - y_RIS[0]))/2
        # print(y_center)
        theta1 =  np.arctan(np.abs(y_Tx - y_center) /np.abs(x_Tx - x_center)) - np.arctan(np.abs(y_Tx - y_RIS) /np.abs(x_Tx - x_RIS))
        theta2 =  np.arctan(np.abs(y_Tx - y_RIS) /np.abs(x_Tx - x_RIS))
        theta3 =  np.arctan(np.abs(y_Rx - y_RIS) /np.abs(x_Rx - x_RIS))
        theta4 =  np.arctan(np.abs(y_Rx - y_center) /np.abs(x_Rx - x_center)) - np.arctan(np.abs(y_Rx - y_RIS) /np.abs(x_Rx - x_RIS))
    
    if (wall == 'right'):
        x_center = x_RIS[0]
        y_center = np.absolute(y_RIS[-1] + y_RIS[0] + 2 * (y_RIS[1] - y_RIS[0]))/2
        # print(x_center, y_center)
        theta1 =  np.arctan(np.abs(y_Tx - y_center) /np.abs(x_Tx - x_center)) - np.arctan(np.abs(y_Tx - y_RIS) /np.abs(x_Tx - x_RIS))
        theta2 =  np.arctan(np.abs(y_Tx - y_RIS) /np.abs(x_Tx - x_RIS))
        theta3 =  np.arctan(np.abs(y_Rx - y_RIS) /np.abs(x_Rx - x_RIS))
        theta4 =  np.arctan(np.abs(y_Rx - y_center) /np.abs(x_Rx - x_center)) - np.arctan(np.abs(y_Rx - y_RIS) /np.abs(x_Rx - x_RIS))

    F1 = np.power(np.cos(theta1), alphaTx)
    F2 = np.power(np.cos(theta2), alphaRIS)
    F3 = np.power(np.cos(theta3), alphaRIS)
    F4 = np.power(np.cos(theta4), alphaRx)
    
    if (RIS_type == 'Intelligent array'):
        Fcombine = F1 * F4
        return Fcombine
    if (RIS_type == 'Conventional array'):
        Fcombine = F1 * F2 * F3 * F4
        return Fcombine