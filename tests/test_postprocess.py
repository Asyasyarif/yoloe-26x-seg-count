import math
import unittest


class DetectionTests(unittest.TestCase):
    def setUp(self):
        from src.postprocess import normalize_detections
        self.normalize = normalize_detections

    def process(self, boxes, scores, indices, classes=None, names=None, confidence=.25):
        classes = classes or ['car', 'red car', 'helmet']
        names = names or dict(enumerate(classes))
        return self.normalize(boxes, scores, indices, names, classes, 100, 80, confidence)

    def test_clips_filters_sorts_and_assigns_ids(self):
        objects = self.process(
            [[60, 40, 120, 90], [-5, -4, 20.2, 20.9], [5, 5, 5, 6],
             [0, 0, math.nan, 30], [1, 1, 5, 5], [2, 2, 5, 5]],
            [.9, .8, .9, .9, .24999, float('nan')], [0, 2, 0, 0, 0, 0])
        self.assertEqual([o.id for o in objects], ['obj_001', 'obj_002'])
        self.assertEqual([o.label for o in objects], ['helmet', 'car'])
        self.assertEqual(objects[0].bbox, {'x1': 0, 'y1': 0, 'x2': 21, 'y2': 21})
        self.assertEqual(objects[1].bbox, {'x1': 60, 'y1': 40, 'x2': 100, 'y2': 80})

    def test_dynamic_mapping_never_uses_coco_ids(self):
        objects = self.process([[1, 1, 10, 10]], [.7], [0], classes=['forklift', 'person'])
        self.assertEqual((objects[0].label, objects[0].prompt_class_index), ('forklift', 0))
        objects = self.process([[1, 1, 10, 10]], [.7], [0], classes=['person', 'forklift'])
        self.assertEqual((objects[0].label, objects[0].prompt_class_index), ('person', 0))

    def test_invalid_model_class_mapping_fails_closed(self):
        for indices, names in [([8], {0: 'car'}), ([.5], {0: 'car'}),
                               ([0], {0: 'bus'}), ([0], {})]:
            with self.subTest(indices=indices, names=names):
                with self.assertRaises(ValueError):
                    self.normalize([[1, 1, 10, 10]], [.7], indices, names,
                                   ['car'], 100, 80, .25)

    def test_cross_class_duplicates_keep_highest_confidence(self):
        objects = self.process([[0, 0, 20, 20], [0, 0, 20, 20], [15, 0, 35, 20]],
                               [.6, .9, .8], [0, 1, 0])
        self.assertEqual([o.label for o in objects], ['red car', 'car'])
        self.assertEqual(len(objects), 2)

    def test_same_class_and_nested_overlaps_are_preserved(self):
        objects = self.process([[0, 0, 20, 20], [0, 0, 20, 20], [5, 5, 10, 10]],
                               [.6, .9, .8], [0, 0, 2])
        self.assertEqual(len(objects), 3)

    def test_dedup_uses_geometry_before_integer_rounding(self):
        objects = self.process([[0, 0, 1, 1], [.4, .4, 1, 1]], [.9, .8], [0, 1])
        self.assertEqual(len(objects), 2)

    def test_threshold_before_serialization_rounding_and_empty(self):
        self.assertEqual(self.process([], [], []), [])
        self.assertEqual(self.process([[0, 0, 5, 5]], [.24999999], [0]), [])
        self.assertEqual(len(self.process([[0, 0, 5, 5]], [.25], [0])), 1)

    def test_sequence_lengths_and_configuration_validated(self):
        with self.assertRaises(ValueError):
            self.process([[0, 0, 5, 5]], [], [0])
        for width, height, threshold in [(0, 80, .25), (100, -1, .25),
                                         (100, 80, float('nan')), (100, 80, 1.1)]:
            with self.subTest(width=width, height=height, threshold=threshold):
                with self.assertRaises(ValueError):
                    self.normalize([], [], [], {}, ['car'], width, height, threshold)

    def test_deterministic_order_with_shuffled_candidates(self):
        boxes = [[0, 0, 20, 20], [0, 0, 20, 20], [30, 10, 50, 40]]
        scores = [.8, .8, .9]
        indices = [0, 1, 2]
        expected = self.process(boxes, scores, indices)
        self.assertEqual(expected, self.process(boxes[::-1], scores[::-1], indices[::-1]))


if __name__ == '__main__':
    unittest.main()
