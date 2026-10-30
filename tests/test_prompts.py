import unittest


class PromptTests(unittest.TestCase):
    def setUp(self):
        from src.prompts import resolve_target_phrases
        self.resolve = resolve_target_phrases

    def test_examples(self):
        cases = {
            'person': ['person'],
            'car, motorcycle': ['car', 'motorcycle'],
            'count red helmets': ['red helmet'],
            'find forklift and pallet': ['forklift', 'pallet'],
            'hitung orang dan mobil': ['person', 'car'],
            'count red helmets and blue bottles': ['red helmet', 'blue bottle'],
            'hitung orang dan sepeda motor': ['person', 'motorcycle'],
            'cari semua helm merah, palet kayu': ['red helmet', 'wooden pallet'],
            'hitung orang yang ada': ['person'],
            'detect all the people and the cars': ['person', 'car'],
        }
        for prompt, expected in cases.items():
            with self.subTest(prompt=prompt):
                self.assertEqual(self.resolve(prompt), expected)

    def test_arbitrary_open_vocabulary_and_conservative_plural(self):
        self.assertEqual(self.resolve('glass, brass, citrus, cactus, industrial valve'),
                         ['glass', 'brass', 'citrus', 'cactus', 'industrial valve'])

    def test_deduplicates_after_translation_in_stable_order(self):
        self.assertEqual(self.resolve(' Cars, mobil, forklift, forklifts, person '),
                         ['car', 'forklift', 'person'])

    def test_rejects_unsupported_or_ambiguous_semantics(self):
        invalid = [
            '', '  ', 'count everything', 'objects', 'count all',
            'people holding two boxes', 'people standing next to cars',
            'all objects to the left of the forklift', 'count cars except blue ones',
            'people wearing red helmets', 'cars or motorcycles', 'cars without drivers',
            'how many cars are there?', 'count 2 cars', 'cars; motorbikes',
            'hitung orang di sebelah mobil', 'orang yang membawa kotak',
            'person,,car', ',car', 'person,', 'cars and', 'and cars',
            'cars and and motorcycles', 'person, and car', 'hitung semua objek yang ada',
            'orang yang merah', 'person\ncar', 'person\tcar',
            'person\x00', 'a' * 257,
            ','.join('category' + chr(97 + n) for n in range(9)),
        ]
        for prompt in invalid:
            with self.subTest(prompt=prompt):
                with self.assertRaises(ValueError):
                    self.resolve(prompt)

    def test_unknown_unicode_phrase_is_preserved(self):
        self.assertEqual(self.resolve('rambu lalu lintas, café sign'),
                         ['rambu lalu lintas', 'café sign'])

    def test_maximum_length_and_classes(self):
        self.assertEqual(self.resolve('a' * 256), ['a' * 256])
        self.assertEqual(len(self.resolve(','.join('thing' + chr(97 + n) for n in range(8)))), 8)

    def test_relation_verbs_rejected_but_noun_compounds_preserved(self):
        for prompt in ['people touching cars', 'people pushing carts', 'person pulling pallet', 'people carrying cases']:
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                self.resolve(prompt)
        self.assertEqual(self.resolve('standing desk, carrying case, walking stick'),
                         ['standing desk', 'carrying case', 'walking stick'])

    def test_punctuation_is_not_a_category(self):
        with self.assertRaises(ValueError):
            self.resolve('count ---')


if __name__ == '__main__':
    unittest.main()
