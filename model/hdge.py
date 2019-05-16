import os
import sys

import numpy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config
from grid.create_grid import RegionGrid
import gensim
import logging

logging.basicConfig(filename='region.log', filemode='w', level=logging.INFO,
                    format='%(asctime)s %(message)s')


def random_walk_distance(m, high, n_walks=10000, walk_length=8):
    walks = []
    for walk_num in range(0, n_walks):
        # get starting vertex
        v = numpy.random.randint(low=0, high=high, size=1)[0]

        walk_sequence = []
        for i in range(0, walk_length):
            # start walk
            # sum of all distances
            distances = m[v]
            s = distances.sum(axis=0)
            # create probabilities
            p = [d / s for d in distances]
            # pick node to walk to
            walk = numpy.random.choice(list(range(0, len(distances))), size=1, p=p)[0]
            walk_sequence.append(walk)
            # set this vertex to walk from
            v = walk
        walks.append(walk_sequence)
    return walks


def random_walk_time(m, high, n_walks=10000, walk_length=8):
    walks = []
    for walk_num in range(0, n_walks):
        # get starting vertex
        v = numpy.random.randint(low=0, high=high, size=1)[0]

        walk_sequence = []
        for i in range(0, walk_length):
            # start walk
            # total counts for each time bucket
            sums = m[v].sum(axis=1)
            s = sums.sum()
            # create probabilities
            p = [x / s for x in sums]
            walk = numpy.random.choice(m[v].shape[0], size=1, p=p)[0]
            walk_sequence.append(walk)
            # set this vertex to walk from
            v = walk
        walks.append(walk_sequence)
    return walks


c = get_config()
# init region
region_grid = RegionGrid(config=c)


def e_distance(d):
    import math
    return math.exp(-d * 1.5)


distance_matrix = region_grid.get_distance_mtx(transform=e_distance)

W, t = region_grid.create_flow_matrix(c['raw_flow_file'], region_name=c['city_name'], time=8)

size = region_grid.grid_size * region_grid.grid_size

distance_sequences = random_walk_distance(distance_matrix, size)
time_sequences = random_walk_time(t, size)

sequences = []
for seq in distance_sequences + time_sequences:
    sequences.append(" ".join([str(id) for id in seq]))

print(sequences)

# each sequence is the walk from region idx to the other

model = gensim.models.Word2Vec(
    sequences,
    size=20,  # adjustable
    window=8,  # walk length
    min_count=2,
    negative=5,
    workers=4  # number of cores on machine
)

model.train(sequences, total_examples=len(sequences), epochs=10)

model.wv.save_word2vec_format("embedding.txt")
