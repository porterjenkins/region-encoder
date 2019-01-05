# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 14:44:49 2015

@author: feiwu
"""


class Point:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon
    def make_key(self):
        return '{},{}'.format(self.lat,self.lon)

class POI:
    def __init__(self,name,pid,lat,lon,cat,checkin_count,user_count):
        self.name = name
        self.pid  = pid
        self.location = Point(lat,lon)
        self.cat      = cat
        self.checkin_count = checkin_count
        self.user_count    = user_count
        self.extra_id      = '' # for debuging purposes
        self.popularity    = dict()
        
    def add_extra(self, extra_id):
        self.extra_id = extra_id
    def add_density(self, key,den):
        self.popularity[key] = den
  