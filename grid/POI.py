from tract import Tract
from shapely.geometry import Point
import pickle
import numpy as np
from config import get_config
import os.path





def generatePOIfeature(config, gridLevel='ca'):
    """
    generate POI features and write out to a file

    regionLevel could be "ca" or "tract"

    ['Food', 'Residence', 'Travel', 'Arts & Entertainment',
    'Outdoors & Recreation', 'College & Education', 'Nightlife',
    'Professional', 'Shops', 'Event']
    """
    if gridLevel == 'ca':
        cas = Tract.createAllCAObjects()
    elif gridLevel == 'tract':
        cas = Tract.createAllTracts()

    ordKey = sorted(cas.keys())

    gcn = np.zeros((len(cas), 3))  # check-in count, user count, and POI count
    gcat = {}

    with open(config['poi_file'], 'rb') as fin:
        POIs = pickle.load(fin)

    #with open('category_hierarchy.pickle', 'r') as f2:
    #    poi_cat = pickle.load(f2)

    cnt = 0
    for poi in POIs.values():
        print("POI Progress: {:.4f}".format(cnt / len(POIs)), end='\r')
        loc = Point(poi.location.lon, poi.location.lat)
        #if poi.cat in poi_cat:
        #    cat = poi_cat[poi.cat]
        #else:
        #    continue
        cat = poi.cat

        for key, grid in cas.items():
            if grid.polygon.contains(loc):
                gcn[ordKey.index(key), 0] += poi.checkin_count
                gcn[ordKey.index(key), 1] += poi.user_count
                gcn[ordKey.index(key), 2] += 1
                """
                Build a two-level dictionary,
                first index by region id,
                then index by category id,
                finally, the value is number of POI under the category.
                """
                if key in gcat:
                    if cat in gcat[key]:
                        gcat[key][cat] += 1
                    else:
                        gcat[key][cat] = 1
                else:
                    gcat[key] = {}
                    gcat[key][cat] = 1

                # break the polygon loop
                cnt += 1
                break

    s = 0
    hi_catgy = []
    for catdict in gcat.values():
        hi_catgy += catdict.keys()
        for c in catdict.values():
            s += c

    hi_catgy = list(set(hi_catgy))
    print(hi_catgy)

    gdist = np.zeros((len(cas), len(hi_catgy)))
    for key, distDict in gcat.items():
        for idx, cate in enumerate(hi_catgy):
            if cate in distDict:
                gdist[ordKey.index(key), idx] = distDict[cate]
            else:
                gdist[ordKey.index(key), idx] = 0

    if gridLevel == 'ca':
        np.savetxt(config['data_dir_main'] + "/POI_dist.csv", gdist, delimiter=",")
        np.savetxt(config['data_dir_main'] + "/POI_cnt.csv", gcn, delimiter=",")
    elif gridLevel == 'tract':
        np.savetxt(config['data_dir_main'] + "/POI_dist_tract.csv", gdist, delimiter=",")
        np.savetxt(config['data_dir_main'] + "/POI_cnt_tract.csv", gcn, delimiter=",")
        with open(config['data_dir_main'] + "/POI_tract.pickle", 'wb') as fout:
            pickle.dump(ordKey, fout)
            pickle.dump(gcat, fout)


if __name__ == '__main__':
    c = get_config()
    generatePOIfeature(config=c, gridLevel='tract')