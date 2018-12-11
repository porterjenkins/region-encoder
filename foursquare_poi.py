# -*- coding: utf-8 -*-
"""
Generate the foursquare POI feature of Chicago.
Use the POI data at
    ../data/all_POIs_chicago
Created on Tue Jan 26 11:09:49 2016
@author: kok
"""

from tract import Tract
from shapely.geometry import Point
import pickle
import numpy as np

import os.path

here = os.path.dirname(os.path.abspath(__file__))


def getFourSquareCount(leaveOut=-1):
    """
    retrieve Foursquare count from local file
    """
    s = np.loadtxt("POI_cnt.csv", delimiter=",")

    if leaveOut > 0:
        s = np.delete(s, leaveOut - 1, 0)
    return s


def getFourSquarePOIDistributionHeader():
    header = ['Food', 'Residence', 'Travel', 'Arts & Entertainment',
              'Outdoors & Recreation', 'College & Education', 'Nightlife',
              'Professional', 'Shops', 'Event']
    return header


def getFourSquarePOIDistribution(leaveOut=-1, gridLevel='ca', useRatio=False):
    """
    retrieve Foursquare POI distribution from local file
    """
    if gridLevel == 'ca':
        d = np.loadtxt(here + "/POI_dist.csv", delimiter=",")
    elif gridLevel == 'tract':
        d = np.loadtxt(here + '/POI_dist_tract.csv', delimiter=",")

    if leaveOut > 0:
        d = np.delete(d, leaveOut - 1, 0)

    if useRatio:
        poi_sum = np.sum(d, axis=1)
        for i in range(len(poi_sum)):
            d[i] = d[i] / poi_sum[i]
        d = np.nan_to_num(d)

    return d


def generatePOIfeature(gridLevel='ca'):
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
        cas = Tract.createAllTractObjects()

    ordKey = sorted(cas.keys())

    gcn = np.zeros((len(cas), 3))  # check-in count, user count, and POI count
    gcat = {}

    with open('../data/all_POIs_chicago', 'r') as fin:
        POIs = pickle.load(fin)

    with open('category_hierarchy.pickle', 'r') as f2:
        poi_cat = pickle.load(f2)

    cnt = 0
    for poi in POIs.values():
        loc = Point(poi.location.lon, poi.location.lat)
        if poi.cat in poi_cat:
            cat = poi_cat[poi.cat]
        else:
            continue

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
    print
    hi_catgy

    gdist = np.zeros((len(cas), len(hi_catgy)))
    for key, distDict in gcat.items():
        for idx, cate in enumerate(hi_catgy):
            if cate in distDict:
                gdist[ordKey.index(key), idx] = distDict[cate]
            else:
                gdist[ordKey.index(key), idx] = 0

    if gridLevel == 'ca':
        np.savetxt(here + "/POI_dist.csv", gdist, delimiter=",")
        np.savetxt(here + "/POI_cnt.csv", gcn, delimiter=",")
    elif gridLevel == 'tract':
        np.savetxt(here + "/POI_dist_tract.csv", gdist, delimiter=",")
        np.savetxt(here + "/POI_cnt_tract.csv", gcn, delimiter=",")
        with open(here + "/POI_tract.pickle", 'w') as fout:
            pickle.dump(ordKey, fout)
            pickle.dump(gcat, fout)


def getPOIlabel():
    d = getFourSquarePOIDistribution(-1, "ca", False)
    header = getFourSquarePOIDistributionHeader()
    POIlabel = {}
    for i, h in enumerate(header):
        F = d[:, i]
        thrd = np.median(F)
        lbl = [1 if ele >= thrd else 0 for ele in F]
        POIlabel[h] = lbl
    pickle.dump(POIlabel, open("poi-label", "w"))
    return POIlabel, d


def tract_poi_profile():
    """
    Profile tract by their POIs

    For each POI category,
    firstly find the top 3 tract with the highest POI counts;
    next find the top 3 tract with the highest POI percentage
    with sufficent count supports (50).
    """
    with open(here + "/POI_tract.pickle") as fin:
        ordKey = pickle.load(fin)
        gcat = pickle.load(fin)

    POIcat = {}
    for tractk, val in gcat.items():
        totalPOI = float(sum(val.values()))
        for poik in val:
            if poik not in POIcat:
                POIcat[poik] = []
            else:
                POIcat[poik].append((tractk, val[poik], val[poik] / totalPOI))

    for poik in POIcat.keys():
        print
        poik
        POIcat[poik].sort(key=lambda x: -x[1])
        print
        POIcat[poik][:3]
        POIcat[poik] = POIcat[poik][:50]
        POIcat[poik].sort(key=lambda x: -x[2])
        print
        POIcat[poik][:3]
    return POIcat


if __name__ == '__main__':

    import sys

    #if len(sys.argv) > 1:
    #    if sys.argv[1] == "getPOI":
    #        d = getFourSquarePOIDistribution(useRatio=True)
    #    elif sys.argv[1] == "getPOIlabel":
    #        l, d = getPOIlabel()
    #    elif sys.argv[1] == "poiProfile":
    #        tp = tract_poi_profile()
    #else:
    #    generatePOIfeature(gridLevel='tract')
