from fractions import Fraction
import json
from pathlib import Path
import random
import tempfile
import unittest

from llm.maths import FAMILIES, make_problem, prepare_worked_maths
from llm.prepare import BALANCED_WEIGHTS
from llm.tokenizer import EOS_TOKEN


class MathsTests(unittest.TestCase):
    def test_answers_against_exact_arithmetic_and_equation_substitution(self):
        rng = random.Random(7)
        for family in FAMILIES:
            for _ in range(100):
                example = make_problem(rng, family)
                p, answer = example["parameters"], example["answer"]
                if family == "addition":
                    self.assertEqual(int(answer), p["a"] + p["b"])
                elif family == "subtraction":
                    self.assertEqual(int(answer) + p["b"], p["a"])
                elif family == "multiplication":
                    self.assertEqual(int(answer), p["a"] * p["b"])
                elif family == "division":
                    q, r = map(int, answer.split(" remainder "))
                    self.assertEqual(q * p["divisor"] + r, p["dividend"])
                    self.assertTrue(0 <= r < p["divisor"])
                elif family.startswith("fraction_"):
                    a, b = Fraction(p["an"], p["ad"]), Fraction(p["bn"], p["bd"])
                    expected = a + b if family == "fraction_addition" else a * b
                    self.assertEqual(Fraction(answer), expected)
                elif family == "percentage":
                    self.assertEqual(Fraction(answer) * 100, p["percent"] * p["amount"])
                elif family == "linear_equation":
                    self.assertEqual(p["a"] * int(answer) + p["b"], p["c"])
                elif family == "rectangle_area":
                    self.assertEqual(int(answer.removesuffix(" square cm")), p["a"] * p["b"])
                elif family == "triangle_area":
                    self.assertEqual(Fraction(answer.removesuffix(" square cm")) * 2, p["a"] * p["b"])
                elif family == "probability":
                    self.assertEqual(Fraction(answer) * p["total"], p["favourable"])
                    self.assertTrue(0 <= Fraction(answer) <= 1)
                else:
                    # Differentiate using polynomial coefficient indexing as an independent check.
                    coefficients = [0] * (p["n"] + 1)
                    coefficients[0], coefficients[1], coefficients[p["n"]] = p["c"], p["b"], p["a"]
                    derivative = [index * value for index, value in enumerate(coefficients)][1:]
                    leading, rest = answer.split("*x^")
                    power, constant = rest.split(" + ")
                    for x in (-2, 0, 3):
                        actual = int(leading) * x ** int(power) + int(constant.strip("()"))
                        expected = sum(value * x ** degree for degree, value in enumerate(derivative))
                        self.assertEqual(actual, expected)

    def test_generation_is_reproducible_and_holdout_has_no_shared_problems(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worked.txt"
            prepare_worked_maths(path, 1000, 100, 42)
            before = path.read_bytes()
            metadata = json.loads(path.with_suffix(".json").read_text())
            records = {split: [json.loads(line) for line in Path(source).read_text().splitlines()]
                       for split, source in metadata["records"].items()}
            train = {record["problem_id"] for record in records["train"]}
            valid = {record["problem_id"] for record in records["validation"]}
            self.assertEqual(len(train), 1000)
            self.assertEqual(len(valid), 100)
            self.assertFalse(train & valid)
            for split, items in records.items():
                for record in items:
                    held_out = int(record["problem_id"][:8], 16) % 10 == 0
                    self.assertEqual(held_out, split == "validation")
            self.assertEqual(set(metadata["family_counts"]["train"]), set(FAMILIES))
            self.assertEqual(path.read_text().count(EOS_TOKEN), 1000)
            prepare_worked_maths(path, 1000, 100, 42)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(ValueError):
                prepare_worked_maths(path, 0, 100)

    def test_maths_exposure_is_guaranteed_independent_of_book_sizes(self):
        self.assertAlmostEqual(sum(BALANCED_WEIGHTS.values()), 1)
        maths = BALANCED_WEIGHTS["maths_books"] + BALANCED_WEIGHTS["maths_worked"]
        self.assertAlmostEqual(maths, .14)
        self.assertAlmostEqual(maths + BALANCED_WEIGHTS["education"], .2)
        self.assertAlmostEqual(BALANCED_WEIGHTS["fiction"], .8)


if __name__ == "__main__":
    unittest.main()
