import unittest

import numpy

from config import get_config
from grid.create_grid import RegionGrid

c = get_config()
TEST_REGION_GRID = RegionGrid(50, poi_file=open(c["poi_file"], 'rb'))


class CreateGridTest(unittest.TestCase):

    def test_basic_region(self):
        self.assertEqual(len(TEST_REGION_GRID.regions), 2500)

    def test_region_grid_has_proper_lat_long(self):
        region = TEST_REGION_GRID.regions['0,49']
        self.assertIsNotNone(region.points['nw'])
        self.assertIsNotNone(region.points['sw'])
        self.assertIsNotNone(region.points['ne'])
        self.assertIsNotNone(region.points['se'])

        last_y = TEST_REGION_GRID.y_space[-1]
        first_x = TEST_REGION_GRID.x_space[0]

        second_x = TEST_REGION_GRID.x_space[1]
        second_y = TEST_REGION_GRID.y_space[-2]

        self.assertEqual(TEST_REGION_GRID.lat_min, first_x)
        self.assertEqual(TEST_REGION_GRID.lon_max, last_y)

        # south corners have y that is the last slot in the space
        self.assertEqual(region.points['se'][1], last_y)
        self.assertEqual(region.points['sw'][1], last_y)

        self.assertEqual(region.points['sw'][0], first_x)
        self.assertEqual(region.points['se'][0], second_x)

        # nw corner have y that is the second to last slot in the space and first x
        self.assertEqual(region.points['nw'][0], first_x)
        self.assertEqual(region.points['nw'][1], second_y)

        # ne corner have y that is the second to last slot in the space and second x
        self.assertEqual(region.points['ne'][0], second_x)
        self.assertEqual(region.points['ne'][1], second_y)

    def test_categories(self):
        region = TEST_REGION_GRID.regions['25,25']
        # get feature matrix indices
        features = sorted(list(numpy.nonzero(TEST_REGION_GRID.feature_matrix[region.index])[0]))
        # get the categories for the region and get their ids
        cat_ids = sorted([TEST_REGION_GRID.categories[cat] for cat in region.categories])

        self.assertEqual(features, cat_ids)


if __name__ == '__main__':
    unittest.main()
