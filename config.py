import json
import os

def get_config():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    with open('{}/config.json'.format(dir_path), 'r') as f:
        config = json.load(f)

    for key, val in config.items():
        if 'file' in key:
            config[key] = config['data_dir_main'] + val

        if 'lat' in key or 'lon' in key:
            config[key] = float(val)

    return config


if __name__ == '__main__':

    c = get_config()
    print(c)