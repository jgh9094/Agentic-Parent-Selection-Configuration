import numpy as np

def selection(fitnesses):
    pop_size = len(fitnesses)
    tournament_size = 3
    selected_indices = np.random.choice(pop_size, tournament_size, replace=False)
    # Calculate mean error per individual in the tournament
    mean_errors = np.array([np.mean(fitnesses[i]) for i in selected_indices])
    # Find all individuals tied for the lowest mean error
    best_positions = np.flatnonzero(mean_errors == mean_errors.min())
    # Randomly break ties by picking one of the best individuals
    winner_index = selected_indices[np.random.choice(best_positions)]
    return winner_index