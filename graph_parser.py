import numpy as np
from config import get_config
#from database_access import DatabaseAccess
# import ipdb


# in Jinan City, x should be around 117 and y should be around 36


class GraphParser(object):
    def __init__(self):
        self.node2id = {}
        self.id2node = {}
        self.id2coor = {}
        self.coor2id = {}

        self.node2edge = {}
        self.edges = []
        # self.init()

    def dist_gen(self, coor1, coor2):
        coor1 = np.array(coor1)
        coor2 = np.array(coor2)
        d = np.sqrt(np.sum(np.square(coor1 - coor2)))
        return d

    def init(self):
        self.initNode()
        self.initEdge()

    def graphParser(self, xml_file):
        import xml.etree.ElementTree as ET
        # focus on main roads
        # wayAttrSet = set(["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified"])
        wayAttrSet = set(["motorway", "trunk", "primary", "secondary", "tertiary"])

        tree = ET.parse(xml_file)
        root = tree.getroot()

        # first find all potential roads and nodes
        # nodes
        nodes = {}
        # roads
        roads = {}
        lanes = {}
        road_lane_cnt = 0
        for child in root:
            if child.tag == "node":
                nodes[child.attrib["id"]] = child
            if child.tag == "way":
                for subchild in child:
                    if subchild.tag == "tag":
                        if subchild.attrib["k"] == "highway":
                            if subchild.attrib["v"] in wayAttrSet:
                                # Lane count assumptions from: https://wiki.openstreetmap.org/wiki/Key:lanes
                                lane_cnt = 2
                                roads[child.attrib["id"]] = (child, subchild.attrib["v"],lane_cnt)
                                lanes[child.attrib["id"]] = lane_cnt
                        # update lane count if available
                        if subchild.attrib["k"] == 'lanes':
                            lane_cnt = int(subchild.attrib['v'])
                            lanes[child.attrib["id"]] = lane_cnt
                            break
        print(len(nodes), len(roads))

        # extract road segments and all relative nodes
        # nodes on road
        roadNode = {}
        realRoadNode = {}
        roadNodeLanes = {}
        for roadPair in roads.values():
            road = roadPair[0]
            if road.tag != "way":
                continue
            for i in range(len(road)):
                tmp = road[i]
                if road[i].tag == "nd":
                    # add node
                    nodeId = road[i].attrib["ref"]
                    if not nodeId in roadNode:
                        roadNode[nodeId] = 0
                    roadNode[nodeId] += 1
                    roadNodeLanes[nodeId] = lanes[road.attrib["id"]]
                    # # add segment
                    # if i + 1 < len(road):
                    #     if road[i+1].tag == "nd":
                    #         roadSeg[(road[i].attrib["ref"], road[i+1].attrib["ref"])] = road.attrib["id"]
        print(len(roadNode))

        # find node on at least two road
        for node in roadNode.items():
            if node[1] > 1:
                realRoadNode[node[0]] = (nodes[node[0]], node[1],roadNodeLanes.get(node[0],0))
        print("Num of road intersection node: {0}.".format(len(realRoadNode)))



        # clustering
        # id (number), list of node (read_id, (x, y)), first is average
        bigNode = {}
        bigNodeLanes = {}
        bigThre = self.dist_gen((-87.9536, 41.6499), (-87.5343, 42.0391)) * 1.5
        for node in realRoadNode.values():
            found = False
            # ipdb.set_trace()
            nodeObj = node[0]
            id = nodeObj.attrib["id"]
            coor = np.array((float(nodeObj.attrib["lon"]), float(nodeObj.attrib["lat"])))
            lane_cnt = node[2]
            for bignode in bigNode.keys():
                # ipdb.set_trace()
                if self.dist_gen(bigNode[bignode][0][1], coor) < bigThre:
                    # ipdb.set_trace()
                    bigNode[bignode][0][1] = bigNode[bignode][0][1] * len(bigNode[bignode]) + coor
                    bigNode[bignode].append((id, coor))
                    bigNode[bignode][0][1] = bigNode[bignode][0][1] / len(bigNode[bignode])
                    found = True
                    break
            if found is False:
                bigid = len(bigNode)
                bigNode[bigid] = []
                bigNode[bigid].append([bigid, coor])
                bigNode[bigid].append([id, coor])
                bigNodeLanes[bigid] = lane_cnt

        self.bigNode = bigNode
        bigNodeList = []
        for nn in self.bigNode.values():
            bigNodeList.append(list([nn[0][1][1], nn[0][1][0]]))
        with open("bignode_coors.txt", "w") as nf:
            nf.write(str(bigNodeList) + "\n")
        print("Num of big Node: {0}".format(len(bigNode)))

        # write number of lanes
        with open("bignode_lanes.txt", "w") as bnl:
            for node in bigNodeLanes.items():
                bnl.write(",".join((str(node[0]),str(node[1]))) + "\n")


        # seg on road
        roadSeg = {}
        for roadPair in roads.values():
            road = roadPair[0]
            if road.tag != "way":
                continue
            i = 0
            while i < len(road):
                if road[i].tag != "nd":
                    break
                # must start from an intersection
                if not road[i].attrib["ref"] in realRoadNode:
                    i += 1
                    continue
                # search for a segment
                j = i + 1
                while j < len(road):
                    if road[j].tag != "nd":
                        break
                    # find one
                    if road[j].tag == "nd" and road[j].attrib["ref"] in realRoadNode:
                        roadSeg[(int(road[i].attrib["ref"]), int(road[j].attrib["ref"]))] = int(road.attrib["id"])
                        i = j
                        break
                    else:
                        j += 1
                if road[j].tag != "nd":
                    break
                    # ipdb.set_trace()
        print("Num of real road segments: {0}.".format(len(roadSeg)))

        # big node graph
        bNEdge = np.zeros((len(bigNode), len(bigNode)))
        # mapping, int 2 int
        node2bignode = {}
        for bn in bigNode.values():
            for nn in bn[1:]:
                node2bignode[int(nn[0])] = bn[0][0]
        for seg in roadSeg:
            bNEdge[node2bignode[seg[0]], node2bignode[seg[1]]] = 1
        self.bigNodeGraphEdge = bNEdge

        # clean big node
        with open("bignodes.txt", "w") as bnf:
            for node in bigNode.values():
                bnf.write(",".join([str(node[0][0]), str(node[0][1][0]), str(node[0][1][1])]) + "\n")

        segThre = self.dist_gen((-87.9536, 41.6499), (-87.5343, 42.0391)) * 0.9
        ccnt = 0
        with open( "bigedges.txt","w") as bnef:
            for i in range(len(bigNode)):
                icoor = bigNode[i][0][1]
                # dont include self2self edge
                for j in range(len(bigNode)):
                    if i == j:
                        continue
                    if self.bigNodeGraphEdge[i, j] == 1:
                        jcoor = bigNode[j][0][1]
                        if self.dist_gen(icoor, jcoor) < segThre:
                            bnef.write(",".join([str(i), str(j)]) + "\n")
                            ccnt += 1

        print("Num of big node edge: {1}/{0}.".format(int(np.sum(self.bigNodeGraphEdge) / 2), ccnt))


if __name__ == '__main__':
    #region = [(-87.9536, -87.5343), (41.6499, 42.0391)]
    config = get_config()
    parser = GraphParser()
    parser.graphParser(xml_file=config['osm_file'])