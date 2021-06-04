import boto3
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.io import MemoryFile
#from glob import glob


# def s3_dir_ls(granule_dir):

#     objs = []
#     bucket = granule_dir.split("/")[2]
#     key = "/".join(granule_dir.split("/")[3:])

#     s3 = boto3.resource('s3')
#     my_bucket = s3.Bucket(bucket)


#     for obj in my_bucket.objects.filter(Prefix=key):
#         objs.append("s3://" + bucket + "/" + obj.key)

#     return objs[1:]


def convert_granule(granule_path, dest_dir, bands=[13, 14, 15], projection=4326):

    if type(projection) is not int:
        raise ValueError(f'Projection parameter {projection}; must be an integer')

    try:

        granule_name = granule_path.split('/')[-1]

        with rio.open(granule_path) as src:

            if src.crs.to_epsg() != projection:
                with WarpedVRT(src, crs=f'EPSG:{projection}', resampling=Resampling.nearest) as vrt:
                    rast = vrt.read(bands)
                    profile = vrt.profile
            else:
                rast = src.read(bands)
                profile = src.profile

        # https://github.com/mapbox/rasterio/issues/899
        # MemoryFile is like Python's BytesIO, but is backed by a file in
        # GDAL's in-memory filesysem ("/vsimem/").
        #
        # Here we create an empty, writeable in-memory file.
        memfile = MemoryFile()

        # MemoryFile.open() returns a Rasterio dataset object. If the 
        # in-memory file is initially empty, you can pass `driver`, `dtype`,
        # `count`, &c parameters to initialize the dataset.
        with memfile.open(**profile) as gtiff:
            gtiff.write(rast)

        # At this point, if you called `memfile.read()` you'd get all the
        # bytes of that GeoTIFF, as if you read them from disk. You can 
        # pass `memfile` to functions that expect a file-like object.
        # boto3's `put_object()` reads 8k bytes at a time from input files,
        # so you only have that amount of overhead in addition to the 
        # in-memory file's allocation.

        dest_bucket = dest_dir.split('/')[0]
        dest_key = '/'.join(dest_dir.split('/')[1:])
        if dest_key[-1] != '/':
            dest_key += '/'
        dest_key += granule_name

        client = boto3.client('s3')
        client.put_object(Body=memfile, Bucket=dest_bucket, Key=dest_key)

        return 'success'
    except Exception as e:
        return e


