"""Deterministic worked examples with exact integer and rational arithmetic."""

from collections import Counter
from contextlib import ExitStack
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random

from .tokenizer import EOS_TOKEN

FAMILIES = ("addition", "subtraction", "multiplication", "division", "fraction_addition",
            "fraction_multiplication", "percentage", "linear_equation", "rectangle_area",
            "triangle_area", "probability", "polynomial_derivative")


def make_problem(rng, family=None):
    family = family or rng.choice(FAMILIES)
    if family not in FAMILIES:
        raise ValueError(f"Unknown maths family: {family}")
    if family in ("addition", "subtraction", "multiplication"):
        # Mix small and large operands rather than using only four-digit arithmetic.
        bound = rng.choice((20, 100, 1000, 10000))
        a, b = rng.randint(0, bound), rng.randint(0, bound)
        if family in ("addition", "multiplication"):
            a, b = sorted((a, b))
        else:
            a, b = max(a, b), min(a, b)
        p = {"a": a, "b": b}
        tens, units = b - b % 10, b % 10
        if family == "addition":
            question = f"What is {a} + {b}?"
            answer = str(a + b)
            explanation = f"Split {b} into {tens} + {units}. First {a} + {tens} = {a + tens}. Then {a + tens} + {units} = {answer}."
        elif family == "subtraction":
            question = f"What is {a} - {b}?"
            answer = str(a - b)
            explanation = f"Split {b} into {tens} + {units}. First {a} - {tens} = {a - tens}. Then {a - tens} - {units} = {answer}."
        else:
            question = f"What is {a} * {b}?"
            answer = str(a * b)
            explanation = f"Use the distributive law: {a} * ({tens} + {units}). The partial products are {a} * {tens} = {a * tens} and {a} * {units} = {a * units}. Add them: {a * tens} + {a * units} = {answer}."
    elif family == "division":
        divisor, quotient = rng.randint(2, 1000), rng.randint(0, 5000)
        remainder = rng.randrange(divisor)
        dividend = divisor * quotient + remainder
        p = {"dividend": dividend, "divisor": divisor}
        question = f"Divide {dividend} by {divisor}, giving the quotient and remainder."
        answer = f"{quotient} remainder {remainder}"
        explanation = f"{divisor} * {quotient} = {divisor * quotient}. Subtract: {dividend} - {divisor * quotient} = {remainder}. The remainder is nonnegative and less than {divisor}, so the quotient is {quotient}."
    elif family.startswith("fraction_"):
        a = Fraction(rng.randint(1, 2000), rng.randint(2, 200))
        b = Fraction(rng.randint(1, 2000), rng.randint(2, 200))
        a, b = sorted((a, b))
        p = {"an": a.numerator, "ad": a.denominator, "bn": b.numerator, "bd": b.denominator}
        if family == "fraction_addition":
            denominator = math.lcm(a.denominator, b.denominator)
            an = a.numerator * (denominator // a.denominator)
            bn = b.numerator * (denominator // b.denominator)
            question = f"Calculate {a} + {b} and simplify."
            answer = str(a + b)
            explanation = f"Use common denominator {denominator}. Rewrite the fractions as {an}/{denominator} and {bn}/{denominator}. Add numerators: {an} + {bn} = {an + bn}. Reduce {an + bn}/{denominator} to {answer}."
        else:
            numerator, denominator = a.numerator * b.numerator, a.denominator * b.denominator
            question = f"Calculate {a} * {b} and simplify."
            answer = str(a * b)
            explanation = f"Multiply numerators: {a.numerator} * {b.numerator} = {numerator}. Multiply denominators: {a.denominator} * {b.denominator} = {denominator}. Reduce {numerator}/{denominator} to {answer}."
    elif family == "percentage":
        percent, amount = rng.randint(1, 100), rng.randint(1, 10000)
        p = {"percent": percent, "amount": amount}
        question = f"What is {percent}% of {amount}? Give an exact answer."
        answer = str(Fraction(percent * amount, 100))
        explanation = f"Percent means per hundred. Multiply {amount} by {percent}/100: ({amount} * {percent})/100 = {percent * amount}/100 = {answer}."
    elif family == "linear_equation":
        a, b, x = rng.randint(2, 50), rng.randint(-500, 500), rng.randint(-1000, 1000)
        c = a * x + b
        p = {"a": a, "b": b, "c": c}
        question = f"Solve {a}*x + ({b}) = {c}."
        answer = str(x)
        explanation = f"Subtract ({b}) from both sides: {a}*x = {c} - ({b}) = {c - b}. Divide both sides by {a}: x = {c - b}/{a} = {x}. Check: {a}*({x}) + ({b}) = {c}."
    elif family in ("rectangle_area", "triangle_area"):
        a, b = rng.randint(1, 5000), rng.randint(1, 5000)
        if family == "rectangle_area":
            a, b = sorted((a, b))
        p = {"a": a, "b": b}
        if family == "rectangle_area":
            question = f"A rectangle is {a} cm long and {b} cm wide. What is its area?"
            answer = f"{a * b} square cm"
            explanation = f"Rectangle area is length * width. Calculate {a} * {b} = {a * b}. The area is measured in square cm."
        else:
            area = Fraction(a * b, 2)
            question = f"A triangle has base {a} cm and perpendicular height {b} cm. What is its area?"
            answer = f"{area} square cm"
            explanation = f"Triangle area is base * height / 2. Calculate ({a} * {b})/2 = {a * b}/2 = {area}. The area is measured in square cm."
    elif family == "probability":
        total = rng.randint(2, 5000)
        favourable = rng.randint(1, total)
        p = {"favourable": favourable, "total": total}
        question = f"There are {total} equally likely outcomes, of which {favourable} are favourable. What is the probability of a favourable outcome?"
        answer = str(Fraction(favourable, total))
        explanation = f"Probability is favourable outcomes / total outcomes. Thus P = {favourable}/{total}. Simplify to {answer}."
    else:
        a, b, c, n = rng.randint(1, 999), rng.randint(-99, 99), rng.randint(-999, 999), rng.randint(2, 12)
        p = {"a": a, "b": b, "c": c, "n": n}
        question = f"Differentiate f(x) = {a}*x^{n} + ({b})*x + ({c})."
        answer = f"{a * n}*x^{n - 1} + ({b})"
        explanation = f"Apply the power rule d(x^n)/dx = n*x^(n-1). The first term becomes {a}*{n}*x^{n - 1} = {a * n}*x^{n - 1}. The derivative of ({b})*x is {b}, and the constant ({c}) has derivative 0."
    key = json.dumps([family, p], sort_keys=True, separators=(",", ":"))
    return {"problem_id": hashlib.sha256(key.encode()).hexdigest(), "family": family,
            "parameters": p, "question": question, "explanation": explanation, "answer": answer}


def prepare_worked_maths(output, train_count=100000, validation_count=10000, seed=42):
    if min(train_count, validation_count) < 1:
        raise ValueError("Maths example counts must be positive")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_dir = output.parent / "maths-sources"
    source_dir.mkdir(exist_ok=True)
    paths = {"train": output, "validation": output.with_name(output.stem + "-valid.txt")}
    records = {split: source_dir / f"{output.stem}-{split}.jsonl" for split in paths}
    counts = {"train": 0, "validation": 0}
    targets = {"train": train_count, "validation": validation_count}
    families = {split: Counter() for split in paths}
    seen, rng, attempt = set(), random.Random(seed), 0
    temporary = {path: path.with_suffix(path.suffix + ".part") for path in (*paths.values(), *records.values())}
    try:
        with ExitStack() as stack:
            texts = {split: stack.enter_context(temporary[path].open("w", encoding="utf-8")) for split, path in paths.items()}
            audits = {split: stack.enter_context(temporary[path].open("w", encoding="utf-8")) for split, path in records.items()}
            while any(counts[split] < targets[split] for split in paths):
                problem = make_problem(rng, FAMILIES[attempt % len(FAMILIES)])
                attempt += 1
                identity = problem["problem_id"]
                if identity in seen:
                    continue
                seen.add(identity)
                # Partition canonical problems before formatting; variants cannot cross splits.
                split = "validation" if int(identity[:8], 16) % 10 == 0 else "train"
                if counts[split] >= targets[split]:
                    continue
                texts[split].write(f"Question: {problem['question']}\nSolution: {problem['explanation']}\nAnswer: {problem['answer']}\n{EOS_TOKEN}\n\n")
                audits[split].write(json.dumps(problem, ensure_ascii=False) + "\n")
                counts[split] += 1
                families[split][problem["family"]] += 1
        for path, pending in temporary.items():
            pending.replace(path)
    finally:
        for pending in temporary.values():
            pending.unlink(missing_ok=True)
    metadata = {"dataset": "Locally generated exact worked maths", "seed": seed,
                "counts": counts, "family_counts": families, "arithmetic": "Exact integers and fractions; no floating-point answers",
                "split": "Hash of canonical problem parameters; disjoint problems, shared task templates",
                "records": {split: str(path) for split, path in records.items()},
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved worked maths: {train_count:,} training and {validation_count:,} validation examples", flush=True)
