import numpy as np


def generate_periodic_with_defects(dA, dB, num_periods, defect_prob, dev_A, dev_B, seed,
                                   replace_prob_A, replace_prob_B):
    """Generate periodic structure with defects, thickness variations and material replacement"""
    np.random.seed(seed)
    structure = []

    for _ in range(num_periods):
        # Layer A with possible replacement
        if np.random.random() > defect_prob:
            thickness_A = max(1, dA + np.random.normal(0, dev_A))
            if np.random.random() < replace_prob_A:
                layer_type = 'A_alt'
            else:
                layer_type = 'A'
            structure.append((layer_type, thickness_A))

        # Layer B with possible replacement
        if np.random.random() > defect_prob:
            thickness_B = max(1, dB + np.random.normal(0, dev_B))
            if np.random.random() < replace_prob_B:
                layer_type = 'B_alt'
            else:
                layer_type = 'B'
            structure.append((layer_type, thickness_B))

    return structure


def generate_fibonacci_sequence(order):
    """Generate Fibonacci sequence without defects"""
    if order == 1:
        return ['A']
    if order == 2:
        return ['B']
    fib_prev_prev = ['A']
    fib_prev = ['B']
    for _ in range(2, order + 1):
        fib_curr = fib_prev + fib_prev_prev
        fib_prev_prev = fib_prev
        fib_prev = fib_curr
    return fib_prev


def generate_fibonacci_with_defects(order, dA, dB, defect_prob, dev_A, dev_B, seed,
                                    replace_prob_A, replace_prob_B):
    """Generate Fibonacci structure with defects, thickness variations and material replacement"""
    np.random.seed(seed)
    fib_seq = generate_fibonacci_sequence(order)
    structure = []

    for char in fib_seq:
        if char == 'A':
            if np.random.random() > defect_prob:
                thickness = max(1, dA + np.random.normal(0, dev_A))
                if np.random.random() < replace_prob_A:
                    structure.append(('A_alt', thickness))
                else:
                    structure.append(('A', thickness))
        else:  # 'B'
            if np.random.random() > defect_prob:
                thickness = max(1, dB + np.random.normal(0, dev_B))
                if np.random.random() < replace_prob_B:
                    structure.append(('B_alt', thickness))
                else:
                    structure.append(('B', thickness))

    return structure