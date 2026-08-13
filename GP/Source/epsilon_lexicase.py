def selection(fitnesses):
    import numpy as np
    # Ensure input shape and compute required quantities per specification
    num_total_cases = len(fitnesses[0])
    pop_size = len(fitnesses)

    # Convert to numpy array for efficient operations
    arr = np.array(fitnesses, dtype=float)

    # Compute per-case epsilon using median absolute deviation (MAD)
    medians = np.median(arr, axis=0)
    eps = np.median(np.abs(arr - medians), axis=0)

    # Epsilon-lexicase selection: shuffle cases, filter pool by threshold best+epsilon
    case_indices = np.arange(num_total_cases)
    np.random.shuffle(case_indices)

    pool = np.arange(pop_size)
    for c in case_indices:
        if pool.size <= 1:
            break
        errors_c = arr[pool, c]
        best = np.min(errors_c)
        threshold = best + eps[c]
        mask = errors_c <= threshold
        pool = pool[mask]

    return int(np.random.choice(pool))