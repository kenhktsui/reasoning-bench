import random
random.seed(42)
import math
import hashlib


def generate_short_hash(input_string, length=4):
    input_bytes = input_string.encode('ascii')
    hash_object = hashlib.shake_128(input_bytes)
    hex_hash = hash_object.hexdigest(length)
    return hex_hash


def generate_arithmetic_sequence(start, step, length):
    return [start + i * step for i in range(length)], f"{start} + {step} * x"


def generate_geometric_sequence(start, ratio, length):
    return [start * (ratio ** i) for i in range(length)], f"{start} * {ratio} ** x"


def generate_fibonacci_like_sequence(start1, start2, length):
    sequence = [start1, start2]
    for _ in range(length - 2):
        sequence.append(sequence[-1] + sequence[-2])
    return sequence, f"f(0) = {start1}; f(1) = {start2} ;f(n) = f(n-1) + n(n-1)"


def generate_quadratic_sequence(a, b, c, length):
    return [a * (i ** 2) + b * i + c for i in range(length)], f"{a} * x ** 2 + {b} * x + {c}"


def generate_triangular_sequence(length):
    return [n * (n + 1) // 2 for n in range(1, length + 1)], f"x * (x+1) / 2"


def generate_prime_sequence(length):
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    primes = []
    num = 2
    while len(primes) < length:
        if is_prime(num):
            primes.append(num)
        num += 1
    return primes, "prime"


def generate_power_sequence(base, length):
    return [base ** i for i in range(length)], f"{base} ** x"


def generate_factorial_sequence(length):
    sequence = [1]
    for i in range(1, length):
        sequence.append(sequence[-1] * (i + 1))
    return sequence, "x!"


def generate_alternating_sequence(length):
    return [(-1) ** i * (i) for i in range(1, length)], "(-1) ** x * x"


def generate_monotonic_random(base, length, smallest_step=1, largest_step=100):
    sequence = [base]
    if random.random() < 0.5:
        for i in range(1, length):
            sequence.append(sequence[-1] - random.randint(smallest_step, largest_step))
        return sequence, f"monotonic_decrease_random starts with {base}"
    else:
        for i in range(1, length):
            sequence.append(sequence[-1] + random.randint(smallest_step, largest_step))
        return sequence, f"monotonic_increase_random starts with {base}"


def generate_question(sequence, question_type, sequence_type):
    visible_length = random.randint(5, 8)

    if question_type == "next":
        start_pos = 0
        visible_sequence = sequence[:visible_length]
        answer = sequence[visible_length]
        nth = visible_length + 1  # one-indexed
        question = f"Consider the following sequence: {', '.join(map(str, visible_sequence))}, ...\n\n"
        question += "What is the next number in this sequence?\n"
        question += 'Output your answer in JSON with key "answer". If you are not able to provide, answer "null"'
    elif question_type == "nth":
        start_pos = 0
        visible_sequence = sequence[:visible_length]
        nth = random.randint(visible_length + 1, len(sequence))
        answer = sequence[nth - 1]
        question = f"Consider the following sequence: {', '.join(map(str, visible_sequence))}, ...\n\n"
        question += f"What is the {nth}th number in this sequence?\n"
        question += 'Output your answer in JSON with key "answer". If you are not able to provide, answer "null"'
    elif question_type == "previous":
        start_pos = random.randint(5, 8)
        visible_sequence = sequence[start_pos: start_pos + visible_length]
        nth = start_pos  # one-indexed
        answer = sequence[nth - 1]
        question = f"Consider the following sequence: ..., {', '.join(map(str, visible_sequence))}, ...\n\n"
        question += "What is the previous number in this sequence?\n"
        question += 'Output your answer in JSON with key "answer". If you are not able to provide, answer "null"'
    else:
        raise NotImplementedError

    # override answer for random sequence
    if sequence_type == "monotonic_random":
        answer = None

    # adjust index as the function starts from zero
    if sequence_type in ["quadratic", "arithmetic", "fibonacci", "geometric", "power", "prime"]:
        nth -= 1
        start_pos -= 1

    return question, answer, visible_length, start_pos + 1, nth


def generate_questions(target_num_questions):
    questions = []
    id_set = set()
    for i in range(target_num_questions):
        sequence_type = random.choice(["arithmetic", "geometric", "fibonacci", "quadratic", "triangular",
                                       "prime", "power", "factorial", "alternating", "monotonic_random"])
        question_type = random.choice(["next", "nth", "previous"])

        if sequence_type == "arithmetic":
            start = random.randint(1, 10)
            step = random.randint(2, 10)
            sequence, formula_str = generate_arithmetic_sequence(start, step, 30)
        elif sequence_type == "geometric":
            start = random.randint(1, 5)
            ratio = random.randint(2, 5)
            sequence, formula_str = generate_geometric_sequence(start, ratio, 15)
        elif sequence_type == "fibonacci":
            start1 = random.randint(0, 5)
            start2 = random.randint(1, 5)
            sequence, formula_str = generate_fibonacci_like_sequence(start1, start2, 20)
        elif sequence_type == "quadratic":
            a = random.randint(1, 3)
            b = random.randint(-5, 5)
            c = random.randint(-10, 10)
            sequence, formula_str = generate_quadratic_sequence(a, b, c, 20)
        elif sequence_type == "triangular":
            sequence, formula_str = generate_triangular_sequence(20)
        elif sequence_type == "prime":
            sequence, formula_str = generate_prime_sequence(20)
        elif sequence_type == "power":
            base = random.randint(2, 5)
            sequence, formula_str = generate_power_sequence(base, 15)
        elif sequence_type == "factorial":
            sequence, formula_str = generate_factorial_sequence(12)
        elif sequence_type == "alternating":
            sequence, formula_str = generate_alternating_sequence(20)
        elif sequence_type == "monotonic_random":
            base = random.randint(1, 10)
            sequence, formula_str = generate_monotonic_random(base, 20)
        else:
            raise NotImplementedError

        question, answer, visible_length, start_pos, nth = generate_question(sequence, question_type, sequence_type)
        id_for_question = generate_short_hash(question)
        if id_for_question not in id_set:
            id_set.add(id_for_question)
            questions.append(
                {
                    "id": id_for_question,
                    "question_type": question_type,
                    "sequence_type": sequence_type,
                    "question": question,
                    "answer": answer,
                    "visible_length": visible_length,
                    "start_position": start_pos,
                    "nth_element": nth,
                    "formula": formula_str
                }
            )
    return questions


# Generate questions
if __name__ == "__main__":
    from collections import Counter
    from datasets import Dataset

    dataset = Dataset.from_list(generate_questions(4000))
    print(dataset)
    dataset.push_to_hub("kenhktsui/num_seq_bench", private=True)
    print(Counter(dataset["question_type"]).most_common())
    print(Counter(dataset["sequence_type"]).most_common())
    print(Counter(dataset["visible_length"]).most_common(10))
    print(Counter(dataset["nth_element"]).most_common(10))
    for st in ["arithmetic", "geometric", "fibonacci", "quadratic", "triangular",
               "prime", "power", "factorial", "alternating", "monotonic_random"]:
        print(f"***{st}***")
        print([d for d in dataset if d["sequence_type"] == st][0])
