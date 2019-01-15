import json
import os

global BASELINES

BASELINES = ['deepwalk', 'nmf', 'embedding']



def get_config():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    embed_path = dir_path + '/embedding/'

    if not os.path.exists(embed_path):
        os.makedirs(embed_path)


    with open('{}/config.json'.format(dir_path), 'r') as f:
        config = json.load(f)

    for key, val in config.items():
        if 'file' in key:
            for model_name in BASELINES:
                if model_name in key:
                    config[key] = embed_path + val
                    break
                else:
                    config[key] = config['data_dir_main'] + val

        if 'lat' in key or 'lon' in key:
            config[key] = float(val)

        if key == 'grid_size':
            config[key] = int(val)

    return config


if __name__ == '__main__':

    c = get_config()
    print(c)