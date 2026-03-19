import unittest
import ase
from ..src.atoms_extractor import ExampleLayerExtractor

class TestStringMethods(unittest.TestCase):

    def test_layer_splitting(self):
        cube = ase.Atom

        ae = ExampleLayerExtractor()
        
        mask = ae.extract_interesting_regions(atoms)
        write("region_mask_already_grained.xyz", atoms[region_mask_already_grained])
        if len(regions_masks_to_grain) != 0:

        self.assertEqual('foo'.upper(), 'FOO')

if __name__ == '__main__':
    unittest.main()