import ellipsis as el
from glob import glob
from ENV_Vars import *

def upload_rasters(token=None, mapId=None, start=None, total=None):
    
    print("running upload function")
    
    rast_paths = glob("efs_cb_rgb/*.tif")
    startDate = "2019-01-01"
    endDate = "2020-12-31"
    new_timestamp = el.addTimestamp(mapId=mapId, startDate=startDate, endDate=endDate, token=token)
    print(new_timestamp)
    
    if start == None and total == None:
        start = 0 
        stop = len(rast_paths)
        
    if start != None and total == None:
        stop = len(rast_paths)
        
    if start == None and total != None:
        start = 0 
        stop = start+total
        
    print(start,stop)
    for i in range(start,stop):
        print(f'uploading raster {i+1} of {stop}', flush=True, end="\r")
        el.uploadRasterFile(mapId=mapId, timestampId=new_timestamp, file=rast_paths[i], token=token, fileFormat='tif')


if __name__ == "__main__":

    token = el.logIn(username, password)
    mapId = el.getMapId("Congo_Basin_PC_Full",token=token)

    upload_rasters(token=token, mapId=mapId)
