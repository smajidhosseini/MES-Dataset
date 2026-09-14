import importlib.util
import unittest

import numpy as np
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "loao_pseudo_release_core.py"


def load_module():
    spec = importlib.util.spec_from_file_location("loao_pseudo_release_core", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest_rows():
    animal_videos = {
        "A001": ["V001", "V002", "V008", "V014"],
        "A002": ["V003", "V015"], "A003": ["V004"], "A004": ["V005"],
        "A005": ["V006"], "A006": ["V007"],
        "A007": ["V009", "V018", "V021"],
        "A008": ["V010", "V016", "V017", "V020"],
        "A009": ["V011"], "A010": ["V012"],
        "A011": ["V013", "V019", "V023"],
        "A012": ["V022"], "A013": ["V024"], "A014": ["V025"],
    }
    return [
        {"video_id": video, "anonymized_animal_id": animal}
        for animal, videos in animal_videos.items() for video in videos
    ]


class CoreTests(unittest.TestCase):
    def test_integrity_rejects_mes1_pseudopixels(self):
        core = load_module()
        row = {
            "image_sha256": "a", "release_image_path": "a.jpg",
            "release_mask_path": "a.png", "frac_1": 0.1,
            "held_out_animal": "A001", "anonymized_animal_id": "A001",
        }
        with self.assertRaises(AssertionError):
            core.assert_release_integrity([row], expected_count=1)

    def test_integrity_does_not_require_arbitrary_release_size(self):
        core = load_module()
        rows = [{
            "image_sha256": f"sha-{index}",
            "release_image_path": f"images/{index}.jpg",
            "release_mask_path": f"masks/{index}.png",
            "frac_1": 0,
            "held_out_animal": "A001",
            "anonymized_animal_id": "A001",
        } for index in range(3)]
        core.assert_release_integrity(rows)

    def test_video_animal_map_is_complete_and_unambiguous(self):
        core = load_module()
        mapping = core.build_video_animal_map(manifest_rows())
        self.assertEqual(len(mapping), 25)
        self.assertEqual(len(set(mapping.values())), 14)
        self.assertEqual(mapping["V001"], "A001")
        self.assertEqual(mapping["V025"], "A014")

    def test_teacher_assignment_excludes_candidate_animal(self):
        core = load_module()
        mapping = core.build_video_animal_map(manifest_rows())
        all_animals = set(mapping.values())
        assignment = core.teacher_assignment("V010", mapping, all_animals, seeds=(42, 1337, 2026))
        self.assertEqual(assignment["held_out_animal"], "A008")
        self.assertEqual(assignment["training_animals"], sorted(all_animals - {"A008"}))
        self.assertEqual(assignment["seeds"], [42, 1337, 2026])
        self.assertNotIn("A008", assignment["training_animals"])

    def test_global_sha_duplicates_are_rejected_even_across_videos(self):
        core = load_module()
        rows = [
            {"stem": "010_mp4-0001", "video_id": "V010", "image_sha256": "same"},
            {"stem": "019_mp4-0001", "video_id": "V019", "image_sha256": "same"},
        ]
        unique, rejected = core.unique_by_sha256(rows)
        self.assertEqual([row["stem"] for row in unique], ["010_mp4-0001"])
        self.assertEqual(rejected, [{"stem": "019_mp4-0001", "duplicate_of": "010_mp4-0001", "reason": "duplicate_sha256"}])

    def test_teacher_jobs_cover_three_seeds_for_every_animal(self):
        core = load_module()
        animals = [f"A{i:03d}" for i in range(1, 15)]
        jobs = core.teacher_jobs(animals, seeds=(42, 1337, 2026))
        self.assertEqual(len(jobs), 42)
        self.assertEqual({job["animal"] for job in jobs}, set(animals))
        for animal in animals:
            self.assertEqual(
                [job["seed"] for job in jobs if job["animal"] == animal],
                [42, 1337, 2026],
            )
        shards = [jobs[index::3] for index in range(3)]
        self.assertEqual([len(shard) for shard in shards], [14, 14, 14])

    def test_ensemble_statistics_are_normalized_and_zero_for_agreement(self):
        core = load_module()
        probabilities = np.asarray([
            [[[0.8]], [[0.1]], [[0.05]], [[0.05]]],
            [[[0.8]], [[0.1]], [[0.05]], [[0.05]]],
            [[[0.8]], [[0.1]], [[0.05]], [[0.05]]],
        ])
        result = core.ensemble_statistics(probabilities)
        self.assertEqual(result["prediction"].item(), 0)
        self.assertAlmostEqual(result["confidence"].item(), 0.8)
        self.assertAlmostEqual(result["disagreement"].item(), 0.0)
        self.assertGreater(result["normalized_entropy"].item(), 0.0)
        self.assertLess(result["normalized_entropy"].item(), 1.0)

    def test_calibrated_mask_excludes_mes1_and_uncertain_pixels(self):
        core = load_module()
        prediction = np.asarray([[0, 1, 2, 3]])
        confidence = np.asarray([[0.80, 1.00, 0.88, 0.90]])
        entropy = np.asarray([[0.10, 0.10, 0.10, 0.50]])
        disagreement = np.asarray([[0.01, 0.01, 0.01, 0.01]])
        mask = core.calibrated_publication_mask(
            prediction, confidence, entropy, disagreement,
            class_thresholds={0: 0.63, 2: 0.89, 3: 0.50},
            entropy_max=0.44, disagreement_max=0.04,
        )
        np.testing.assert_array_equal(mask, np.asarray([[1, 255, 255, 255]], dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
