
import matplotlib
matplotlib.rc('pdf', fonttype=42)
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
import shapefile
from config import get_config
#from feature_utils import *
import numpy as np


class Tract:
    """
    Define one tract.

    A tract is a census spatial unit with roughly 2000 populations. We use tract
    as the minimum unit, and later build Community Area (CA) on top of tract.
    For each tract we collect related urban features.

    Instance Attributes
    -------------------
    tid : int32
        The tract ID as a integer, e.g. 17031690900.
    polygon : shapely.geometry.Polygon
        The boundary coordinates of a tract.
    CA : int32
        The CA assignment of this tract.
    neighbors : list
        A list of tract instances that are adjacent.
    onEdge : boolean
        Whether current tract is on CA boundary.

    Class Attributes
    ----------------
    config: dict
        A dictionary housing project configuration parameters
    tracts : dict
        A dictionary of all tracts. Key is tract ID. Value is tract instance.
    tract_index : list
        All tract IDs in a list sorted in ascending order.
    features : pandas.DataFrame
        All tract features in a dataframe.
    featureNames: list
        The list of column names that will be used as predictor (X).
    boundarySet : set
        A set of tracts CA boundary given current partition
    """
    config = get_config()

    def __init__(self, tid, shp, rec=None):
        """Build one tract from the shapefile._Shape object"""
        self.id = tid
        self.bbox = box(*shp.bbox)
        self.polygon = Polygon(shp.points)
        self.centroid = (self.polygon.centroid.x, self.polygon.centroid.y)
        if rec != None:
            self.CA = int(rec[6])
        else:
            self.CA = None
        # for adjacency information
        self.neighbors = []
        self.onEdge = False

    @classmethod
    def get_tract_ca_dict(cls):
        tract_to_ca_map = dict()

        for t_id, tract in cls.tracts.items():
            tract_to_ca_map[t_id] = tract.CA

        return tract_to_ca_map

    @classmethod
    def createAllTracts(cls, calculateAdjacency=True):
        fname = cls.config['tract_shape_file']
        cls.sf = shapefile.Reader(fname)
        tracts = {}
        shps = cls.sf.shapes()
        for idx, shp in enumerate(shps):
            rec = cls.sf.record(idx)
            tid = int("".join([rec[0], rec[1], rec[2]]))
            trt = Tract(tid, shp, rec)
            tracts[tid] = trt
        cls.tracts = tracts
        # sorted index of all tract IDs
        cls.tract_index = sorted(cls.tracts.keys())
        # calculate spatial adjacency graph
        if calculateAdjacency:
            cls.spatialAdjacency()
        return tracts

    @classmethod
    def spatialAdjacency(cls):
        """
        Calculate the adjacent tracts.

        Notice that `shapely.touches` return True if there is one point touch.
        """
        for focalKey, focalTract in cls.tracts.items():
            for otherKey, otherTract in cls.tracts.items():
                if otherKey != focalKey and focalTract.polygon.touches(otherTract.polygon):
                    intersec = focalTract.polygon.intersection(otherTract.polygon)
                    if intersec.geom_type != 'Point':
                        focalTract.neighbors.append(otherTract)
        # calculate whether the tract is on CA boundary
        cls.initializeBoundarySet()

    @classmethod
    def visualizeTracts(cls, tractIDs=None, tractColors=None, fsize=(16,16), fname="tracts.png",labels=False):
        tracts = {}
        if tractIDs == None:
            tracts = cls.tracts
        else:
            for tid in tractIDs:
                tracts[tid] = cls.tracts[tid]
        if tractColors == None:
            tractColors = dict(zip(tracts.keys(), ['green']* len(tracts)))
        #print(tractColors)
        from descartes import PolygonPatch
        f = plt.figure(figsize=fsize)
        ax = f.gca()
        for k, t in tracts.items():
            ax.add_patch(PolygonPatch(t.polygon, alpha=0.5, fc=tractColors[k]))
            if labels:
                ax.text(t.polygon.centroid.x,
                        t.polygon.centroid.y,
                        int(t.id),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=8)
        ax.axis("scaled")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(fname)

    @classmethod
    def initializeBoundarySet(cls):
        """
        Initialize the boundary set on given partitions.
        """
        cls.boundarySet = set()
        for _, t in cls.tracts.items():
            for n in t.neighbors:
                if t.CA != n.CA:
                    t.onEdge = True
                    cls.boundarySet.add(t)
                    break

    @classmethod
    def updateBoundarySet(cls, tract):
        """
        Update bounary set for next round sampling
        """
        tracts_check = [tract] + tract.neighbors
        for t in tracts_check:
            onEdge = False
            for n in t.neighbors:
                if t.CA != n.CA:
                    onEdge = True
                    break
            if not onEdge:
                if t.onEdge:
                    t.onEdge = False
                    cls.boundarySet.remove(t)
            else:
                t.onEdge = True
                cls.boundarySet.add(t)

    @classmethod
    def visualizeTractsAdjacency(cls):
        """
        Plot tract adjacency graph. Each tract is ploted with its centroid.
        The adjacency
        """
        from matplotlib.lines import Line2D
        tracts = cls.tracts
        f = plt.figure(figsize=(16, 16))
        ax = f.gca()
        for _, t in tracts.items():
            for n in t.neighbors:
                ax.add_line(Line2D(*zip(t.centroid, n.centroid)))
        ax.axis('scaled')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig("adjacency.png")

    @classmethod
    def get_lat_lon_window(cls):
        min_lat = 9999999
        max_lat = -9999999
        min_lon = 9999999
        max_lon = -9999999

        for id, t in cls.tracts.items():
            lat, lon = t.centroid

            if lat > max_lat:
                max_lat = lat

            if lat < min_lat:
                min_lat = lat

            if lon > max_lon:
                max_lon = lon

            if lon < min_lon:
                min_lon = lon

        return (min_lon, max_lon), (min_lat, max_lat)






if __name__ == '__main__':
    tracts_all = Tract.createAllTracts()
    lon_range, lat_range = Tract.get_lat_lon_window()

    print(lon_range)
    print(lat_range)