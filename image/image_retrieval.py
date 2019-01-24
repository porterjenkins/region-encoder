import time
import urllib.request
from motionless import VisibleMap
from PIL import Image
import config
import os

c = config.get_config()
# put key in config
key = c['g_maps_key']
path_to_image_dir = c['path_to_image_dir']



if not os.path.exists(path_to_image_dir):
    os.makedirs(path_to_image_dir)


def create_urls_no_marker(region_grid):
    urls = {}
    for id, r in region_grid.regions.items():
        url = create_url_for_region_no_marker(r)
        urls[id] = url
    return urls


def create_url_for_region_no_marker(r, x=200, y=200):
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


def get_images_for_all_no_marker(region_grid, file_prefix="", clear_dir=False):
    urls = create_urls_no_marker(region_grid)

    if clear_dir:
        for file in os.listdir(path_to_image_dir):
            file_path = os.path.join(path_to_image_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    # elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

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


def compress_images(image_dir, resize=(50,50)):
    files = os.listdir(image_dir)

    for f in files:
        fname = image_dir + "/" + f
        img = Image.open(fname)
        img_small = img.resize(resize, Image.ANTIALIAS)
        img_small.save(fname, optimize=True, quality=95)



if __name__ == "__main__":
    compress_images(path_to_image_dir)