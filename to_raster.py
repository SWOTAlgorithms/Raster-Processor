#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 15:19:44 2018
@author Shuai Zhang at UNC: 
utm == 0.4.2
netCDF4 == 1.2.4    
"""

from netCDF4 import Dataset
import utm
import numpy as np
from collections import Counter
import aggregate as ag

#######################################################################Input parameters#################################################################
res = 100 # Spatial resolution of raster product (100m or 250m for 'utm' and 1./360 for 'geo')
proj = 'utm' 
fin_folder = 'C:/Data/SWOT/UC/data/input/' #data directory
var_list_tbd = ['quality_flag','flag_surf_type','flag_rain','flag_frozen','flag_layover','sigma0_uncert',\
                'geoid_height','geoid_slope','earth_tide','pole_tide','iono_corr','dry_tropo_corr']
##########################################################################################################################################################

def endWith(s,*endstring):
        array = map(s.endswith,endstring)
        if True in array:
                return True
        else:
                return False
#Scan input files in a specific folder
def scan_input(f_folder):
    file_list = []
    if __name__ == '__main__':
        import os
        s = os.listdir(f_folder)
        for i in s:
            if endWith(i,'cloud.nc'):
                file_list.append(i)
    return file_list                

#Convert some of the variables which can not be simply averaged
def majority_vote(list_tmp):
    max_count = 0
    max_value = 0
    try:
        stat_array = Counter(list_tmp)
        for val, num in stat_array.iteritems():
            if num > max_count:
                max_count = num
                max_value = max_value
        return max_value
    except:
        print sum(list_tmp)


#Convert to utm projection
def convert_projection(good,klass,lats,lons):
    out_index = []#index of pixel cloud in each grid
    x_utm = []
    y_utm = []
    num_tmp = []
    zone_utm = []
    
    x_max = -1e9
    y_max = -1e9
    x_min = 1e9
    y_min = 1e9
    size_x = 0
    size_y = 0
        
    for x in range(0,len(lats)):                       
        if good[x] == False:                
            if lons[x] > 180:
                lons[x] = lons[x]-360   
            x_utm.append(utm.from_latlon(lats[x],lons[x])[0])
            y_utm.append(utm.from_latlon(lats[x],lons[x])[1])
            num_tmp.append(utm.from_latlon(lats[x],lons[x])[2])
            zone_utm.append(utm.from_latlon(lats[x],lons[x])[3])
            if klass[x] > 1 :                    
                if x_max < x_utm[x]:                        
                    x_max = x_utm[x]
                if y_max < y_utm[x]:
                    y_max = y_utm[x]
                if x_min > x_utm[x]:
                    x_min = x_utm[x]
                if y_min > y_utm[x]:
                    y_min = y_utm[x]
        else:
            x_utm.append(0)
            y_utm.append(0)
            num_tmp.append(0)
            zone_utm.append(0)
                 
    size_x = int((x_max-x_min)/res)+1
    size_y = int((y_max-y_min)/res)+1
    utm_num = num_tmp[int(len(lats)/2)]

    for i in range(0, size_y):
        out_index.append([])
        for j in range(0,size_x):
            out_index[i].append([])
            
    for x in range(0,len(lats)):
        i=int((y_utm[x]-y_min)/res)
        j=int((x_utm[x]-x_min)/res)
                
        if i >=0 and j >=0 and i< size_y and j< size_x :
            if klass[x] > 1:           
                out_index[i][j].append(x)

    return size_x,size_y,utm_num,x_min,y_min,out_index

def write_variable(dataset,var_name,var_array,unit='dimensionless'):
    locals()['var_'+s] = dataset.createVariable(var_name, np.float32,('ydim','xdim'),fill_value = -1)
    locals()['var_'+s][:] = var_array
    locals()['var_'+s].coordinates = "y x"
    locals()['var_'+s].grid_mapping = "UTM"
    locals()['var_'+s].units = unit
    
    
def write_raster_netcdf(fout,size_x,size_y,x_min,y_min,utm_num,h,h_uc,area,area_uc,cross_trk):
    dataset = Dataset(fout, 'w')    
    ydim = dataset.createDimension('ydim', size_y)
    xdim = dataset.createDimension('xdim', size_x)
           
    UTM = dataset.createVariable('UTM', np.uint8)
    UTM.grid_mapping_name = 'universal_transverse_mercator'
    UTM.utm_zone_number = utm_num; 
    UTM.semi_major_axis = 6378137;
    UTM.inverse_flattening = 298.257;
    UTM._CoordinateTransformType = "Projection";
    UTM_CoordinateAxisTypes = "GeoX GeoY";

    y = dataset.createVariable('y', np.float32,('ydim'))
    y.units = "m"
    y.long_name = "y coordinate of projection"
    y.standard_name = "projection_y_coordinate"
    y[:] = np.arange(y_min,y_min+size_y*res,res)

    x = dataset.createVariable('x', np.float32,('xdim'),fill_value = -1)
    x.units = "m"
    x.long_name = "x coordinate of projection"
    x.standard_name = "projection_y_coordinate"
    x[:] = np.arange(x_min,x_min+size_x*res,res)

    write_variable(dataset,'cross_trk',cross_trk,'m')
    write_variable(dataset,'water_frac',area)
    write_variable(dataset,'water_frac_uncert',area_uc)
    write_variable(dataset,'height_uncert',h_uc,'m')
    write_variable(dataset,'height',h,'m')
    
    for s in var_list_tbd:            
        print 'Writing TBD varibale ',s,'...'       
        out_var_tmp = []
        out_var = [[0 for i in range(size_x)] for i in range (size_y)]#total pixel number withn each grid 
        locals()['var_'+s] = dataset.createVariable(s, np.float32,('ydim','xdim'),fill_value = -1)
        locals()['var_'+s][:] = out_var
        locals()['var_'+s].coordinates = "y x"
        locals()['var_'+s].grid_mapping = "UTM"
     
    # Global Attributes
    dataset.description = 'Test version' 
    dataset.close()

        
#Check to see if L2_HR_PIXC granule contains water and convert the pixel clouds to raster file
def to_raster(f_file):
    fout_path = f_file[0:-3]+'_utm_raster.nc'# the path of raster file     
    fin = Dataset(f_file, mode='r')
    
    #Read the input pixel clouds
    lats = fin.groups['pixel_cloud'].variables['latitude'][:]
    heights = fin.groups['pixel_cloud'].variables['height'][:]
    lons = fin.groups['pixel_cloud'].variables['longitude'][:]
    klass = fin.groups['pixel_cloud'].variables['classification'][:]
    pixel_area = fin.groups['pixel_cloud'].variables['pixel_area'][:]
    num_rare_looks=fin.groups['pixel_cloud'].variables['num_rare_looks'][:]
    num_med_looks=fin.groups['pixel_cloud'].variables['num_med_looks'][:]
    ifgram=fin.groups['pixel_cloud'].variables['interferogram'][:]
    power1=fin.groups['pixel_cloud'].variables['power_minus_y'][:]
    power2=fin.groups['pixel_cloud'].variables['power_plus_y'][:]
    look_to_efflooks=fin.groups['pixel_cloud'].looks_to_efflooks
    dh_dphi=fin.groups['pixel_cloud'].variables['dheight_dphase'][:]
    ifgram = ifgram[:,0]+ifgram[:,0]*1j
    water_fraction = fin.groups['pixel_cloud'].variables['water_frac'][:]
    water_fraction_uncert = fin.groups['pixel_cloud'].variables['water_frac_uncert'][:] 
    darea_dheight = fin.groups['pixel_cloud'].variables['darea_dheight'][:]
    Pfd = fin.groups['pixel_cloud'].variables['false_detection_rate'][:]
    Pmd = fin.groups['pixel_cloud'].variables['missed_detection_rate'][:]
    cross_trk = fin.groups['pixel_cloud'].variables['cross_track'][:]
    
   
    b = np.isnan(lats)
    print 'Converting projection ...'
    # set the parameters for each projection and calculate the size of raster image

    utm_proj =convert_projection(b,klass,lats,lons)    
           
    size_y = utm_proj[1]
    size_x = utm_proj[0]
    utm_num = utm_proj[2]
    x_min = utm_proj[3]
    y_min = utm_proj[4]
    out_index = utm_proj[5]

# Aggregating to raster for each variable
    print 'Writing variables...'
 
    out_h = [[0 for i in range(size_x)] for i in range (size_y)]
    out_h_uc = [[0 for i in range(size_x)] for i in range (size_y)]
    out_area_frac = [[0 for i in range(size_x)] for i in range (size_y)]
    out_area_frac_uc = [[0 for i in range(size_x)] for i in range (size_y)]
    out_cross_trk = [[0 for i in range(size_x)] for i in range (size_y)]
             
 
    for i in range(0,size_y):
        for j in range(0,size_x):
            if len(out_index[i][j]) != 0:#set missing data as zero
                good = b[out_index[i][j]]                        
                grid_height = ag.height_with_uncerts(heights[out_index[i][j]], ~good,
                num_rare_looks[out_index[i][j]], num_med_looks[out_index[i][j]],
                ifgram[out_index[i][j]], power1[out_index[i][j]], power2[out_index[i][j]], 
                look_to_efflooks, dh_dphi[out_index[i][j]])
                out_h[i][j] =grid_height[0]
                out_h_uc[i][j] = grid_height[2]                        
                grid_area = ag.area_with_uncert(pixel_area[out_index[i][j]], water_fraction[out_index[i][j]], 
                                                water_fraction_uncert[out_index[i][j]],
                                                darea_dheight[out_index[i][j]], klass[out_index[i][j]],
                                                Pfd[out_index[i][j]], Pmd[out_index[i][j]], ~good)            
                out_area_frac[i][j] =grid_area[0]/(res*res)
                out_area_frac_uc[i][j] = grid_area[2]                
                out_cross_trk[i][j] = ag.simple(cross_trk[out_index[i][j]][~good])
         
    write_raster_netcdf(fout_path,size_x,size_y,x_min,y_min,utm_num,out_h,out_h_uc,out_area_frac,out_area_frac_uc,
                        out_cross_trk)
    
pc_file = scan_input(fin_folder)
#Rasterize the pixel clouds files and write these files to hard disk  
for s in pc_file:
    fin_path = fin_folder+s
    print fin_path
    to_raster(fin_path)
