import boto3
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from glob import glob



def s3_dir_ls(granule_dir):

    objs = []
    bucket = granule_dir.split("/")[2]
    key = "/".join(granule_dir.split("/")[3:])

    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket(bucket)


    for obj in my_bucket.objects.filter(Prefix=key):
        objs.append("s3://" + bucket + "/" + obj.key)

    return objs[1:]


def convert_granules(granule_dir, dest_dir, bands=[13, 14, 15], projection=4326):

    if type(projection) is not int:
        raise ValueError(f'Projection parameter {projection}; must be an integer')

    client = boto3.client('s3')
    dest_bucket = dest_dir.split('/')[0]
    granule_list = s3_dir_ls(granule_dir)
    num_granules = len(granule_list)

    try:    
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

        return 'success'
    except Exception as e:
        return e


