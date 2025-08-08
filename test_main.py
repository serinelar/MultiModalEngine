import unittest
from main import load_captions, CAPTIONS_FILE, search_captions

class TestCaptionLoader(unittest.TestCase):
    def test_load_captions_structure(self):
        captions = load_captions(CAPTIONS_FILE)
        self.assertIsInstance(captions, dict)
        self.assertGreater(len(captions), 0)

        for key, value in captions.items():
            self.assertTrue(key.endswith(".jpg"))
            self.assertIsInstance(value, list)
            self.assertIsInstance(value[0], str)
            break  # Just test the first one

    def test_search_captions(self):
        captions = {
            "img1.jpg": ["A cat on a tree", "The dog runs fast"],
            "img2.jpg": ["A man with a dog", "A peaceful lake"]
        }
        result = search_captions(captions, "dog")
        self.assertIn("img1.jpg", result)
        self.assertIn("img2.jpg", result)
        self.assertEqual(len(result["img1.jpg"]), 1)

if __name__ == "__main__":
    unittest.main()
