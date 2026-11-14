"""The architecture diagrams are generated from the code; this keeps the generator working."""

import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import make_architecture_diagrams as diagrams  # noqa: E402


class DiagramTest(unittest.TestCase):
    def test_all_four_svgs_are_written_and_well_formed(self):
        with tempfile.TemporaryDirectory() as out:
            self.assertEqual(diagrams.main(["--out", out]), 0)
            for name in ("overview", "stage1", "stage2", "stage3"):
                root = ET.parse(os.path.join(out, f"{name}.svg")).getroot()
                self.assertTrue(root.tag.endswith("svg"), name)
                self.assertEqual(root.get("width"), "1600")

    def test_trace_is_the_running_example_window(self):
        words, tags, _, records, gold = diagrams.running_example()
        self.assertEqual(" ".join(words), "box band nhi tha main bht upset hoon is bar bhi")
        self.assertEqual(len(tags), len(records), len(gold))


if __name__ == "__main__":
    unittest.main()
