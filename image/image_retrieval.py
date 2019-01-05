import time
import urllib.request

from motionless import VisibleMap

import config

c = config.get_config()
# put key in config
key = c['g_maps_key']
path_to_image_dir = c['path_to_image_dir']

import os

if not os.path.exists(path_to_image_dir):
    os.makedirs(path_to_image_dir)


def create_urls_no_marker(region_grid):
    urls = {}
    for id, r in region_grid.regions.items():
        url = create_url_for_region_no_marker(r)
        urls[id] = url
    return urls


def create_url_for_region_no_marker(r, x=640, y=640):
    vmap = VisibleMap(maptype='satellite', size_x=x, size_y=y, key=key)
    if r.nw:
        vmap.add_latlon(str(r.nw[0]), str(r.nw[1]))
    if r.se:
        vmap.add_latlon(str(r.se[0]), str(r.se[1]))
    if r.ne:
        vmap.add_latlon(str(r.ne[0]), str(r.ne[1]))
    if r.sw:
        vmap.add_latlon(str(r.sw[0]), str(r.sw[1]))
    return vmap.generate_url()


def get_images_for_all_no_marker(region_grid, file_prefix=""):
    urls = create_urls_no_marker(region_grid)

    for id, url in urls.items():
        fn = id.replace(',', '-')
        fn = f"{path_to_image_dir}/{file_prefix}{fn}.png"
        if os.path.isfile(fn):
            print(f"Skipping {fn}")
        else:
            print(f"Downloading {fn}")
            download_image(url, fn)


def download_image(url, fn):
    # exponential backoff
    current_delay = 0.1  # initial retry delay to 100ms.
    max_delay = 3600  # maximum retry 1 hour.
    while True:
        try:
            # download image
            urllib.request.urlretrieve(url, fn)
            print(f"Saved {fn}")
            return
        except IOError as e:
            print(e)
            pass  # Fall through to the retry loop.
        if current_delay > max_delay:
            raise Exception('Too many retry attempts.')
        print('Waiting', current_delay, 'seconds before retrying.')
        time.sleep(current_delay)
        current_delay *= 2
