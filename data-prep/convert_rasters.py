import boto3
import rasterio as rio
import os
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


def convert_granules(granule_list, dest_dir, bands=[2, 3, 4], projection=4326)

    if type(projection) is not int:
        raise ValueError(f'Projection parameter {projection}; must be an integer')

    client = boto3.client('s3')
    dest_bucket = dest_dir.split('/')[0]
    num_granules = len(granule_list)
    
    for i, granule_path in enumerate(granule_list, start=1):
        print(f'Converting granule {i} of {num_granules}')

        granule_name = granule_path.split('/')[-1]

        with rio.open(granule_path) as src:

            if src.crs.to_epsg() != projection:
                with WarpedVRT(src, crs=f'EPSG:{projection}', resampling=Resampling.nearest) as vrt:
                    rast = vrt.read(bands)
            else:
                rast = src.read(bands)
                    
        dest_key = '/'.join(dest_dir.split('/')[1:])
        if dest_key[-1] != '/':
            dest_key += '/'
        dest_key += granule_name

        client.put_object(Body=rast, Bucket=dest_bucket, Key=dest_key)
