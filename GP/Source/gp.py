#!/usr/bin/env python3
"""
DEAP-based Genetic Programming for Symbolic Regression.

This module implements a tree-based genetic programming system for evolving
symbolic regression solutions using the DEAP framework. It supports custom
parent selection strategies, parallel evaluation, and comprehensive logging.

Compatible with Python 3.13 and DEAP 1.4.3.

Author: Agentic Parent Selection Configuration Project
"""

import argparse
import importlib.util
import logging
import os
import random
from functools import partial
from multiprocessing import Pool
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import ray
from deap import base, creator, gp, tools

if TYPE_CHECKING:
    from multiprocessing.pool import Pool as PoolType

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# EPHEMERAL RANDOM CONSTANT GENERATOR
# =============================================================================

# Module-level random generator for ERC (initialized in run_evolution)
_module_rng: Optional[np.random.Generator] = None

def generate_erc() -> float:
    """
    Generate an ephemeral random constant uniformly sampled between -1 and 1.

    This is a module-level function (not a lambda) to support multiprocessing pickling.
    Uses the module-level numpy random generator for reproducibility.

    Returns:
        Random float in range [-1, 1].
    """
    global _module_rng
    if _module_rng is not None:
        return float(_module_rng.uniform(-1, 1))
    return random.uniform(-1, 1)


# =============================================================================
# PROTECTED OPERATORS (Safe versions for GP primitives)
# =============================================================================

def protected_div(left: float, right: float) -> float:
    """
    Protected division operator that avoids division by zero.

    Args:
        left: Numerator value.
        right: Denominator value.

    Returns:
        Result of division, or 1.0 if denominator is near zero.
    """
    if abs(right) < 1e-10:
        return 1.0
    return left / right

def protected_sqrt(x: float) -> float:
    """
    Protected square root operator that handles negative values.

    Args:
        x: Input value.

    Returns:
        Square root of absolute value of x.
    """
    return np.sqrt(np.abs(x))

def protected_log(x: float) -> float:
    """
    Protected logarithm operator that handles non-positive values.

    Args:
        x: Input value.

    Returns:
        Natural logarithm of absolute value of x (with minimum threshold).
    """
    if abs(x) < 1e-10:
        return 0.0
    return np.log(np.abs(x))

def protected_inv(x: float) -> float:
    """
    Protected inverse (1/x) operator that avoids division by zero.

    Args:
        x: Input value.

    Returns:
        Inverse of x, or 1.0 if x is near zero.
    """
    if abs(x) < 1e-10:
        return 1.0
    return 1.0 / x

def protected_tan(x: float) -> float:
    """
    Protected tangent operator that handles values near asymptotes.

    Args:
        x: Input value in radians.

    Returns:
        Tangent of x, clipped to avoid extreme values.
    """
    result = np.tan(x)
    if np.isinf(result) or np.isnan(result):
        return 0.0
    return np.clip(result, -1e10, 1e10)

def safe_abs(x: float) -> float:
    """
    Safe absolute value operator.

    Args:
        x: Input value.

    Returns:
        Absolute value of x.
    """
    return np.abs(x)

def safe_neg(x: float) -> float:
    """
    Safe negation operator.

    Args:
        x: Input value.

    Returns:
        Negated value of x.
    """
    return -x

def safe_sin(x: float) -> float:
    """
    Safe sine operator.

    Args:
        x: Input value in radians.

    Returns:
        Sine of x.
    """
    return np.sin(x)

def safe_cos(x: float) -> float:
    """
    Safe cosine operator.

    Args:
        x: Input value in radians.

    Returns:
        Cosine of x.
    """
    return np.cos(x)

def safe_max(x: float, y: float) -> float:
    """
    Safe maximum operator.

    Args:
        x: First value.
        y: Second value.

    Returns:
        Maximum of x and y.
    """
    return max(x, y)

def safe_min(x: float, y: float) -> float:
    """
    Safe minimum operator.

    Args:
        x: First value.
        y: Second value.

    Returns:
        Minimum of x and y.
    """
    return min(x, y)

def safe_add(x: float, y: float) -> float:
    """
    Safe addition operator.

    Args:
        x: First value.
        y: Second value.

    Returns:
        Sum of x and y.
    """
    return x + y

def safe_sub(x: float, y: float) -> float:
    """
    Safe subtraction operator.

    Args:
        x: First value.
        y: Second value.

    Returns:
        Difference x - y.
    """
    return x - y

def safe_mul(x: float, y: float) -> float:
    """
    Safe multiplication operator.

    Args:
        x: First value.
        y: Second value.

    Returns:
        Product of x and y.
    """
    return x * y


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_inputs(data_dir: str, split_dir: str, selector_path: str,
                    output_dir: str, seed: int, n_cpus: int) -> None:
    """
    Validate all input arguments and ensure required files exist.

    Args:
        data_dir: Directory containing data.csv file.
        split_dir: Directory containing train/val/test split files.
        selector_path: Path to selector.py file.
        output_dir: Directory for output files.
        seed: Random seed value.
        n_cpus: Number of CPUs for parallelization.

    Raises:
        AssertionError: If any validation check fails.
    """
    # Validate data directory and file
    data_csv_path = os.path.join(data_dir, 'data.csv')
    assert os.path.isdir(data_dir), f"Data directory does not exist: {data_dir}"
    assert os.path.isfile(data_csv_path), f"data.csv not found in: {data_dir}"

    # Validate split directory and files
    assert os.path.isdir(split_dir), f"Split directory does not exist: {split_dir}"
    training_path = os.path.join(split_dir, 'training.npy')
    testing_path = os.path.join(split_dir, 'testing.npy')
    assert os.path.isfile(training_path), f"training.npy not found in: {split_dir}"
    assert os.path.isfile(testing_path), f"testing.npy not found in: {split_dir}"

    # Validate selector file
    assert os.path.isfile(selector_path), f"selector.py not found at: {selector_path}"
    assert selector_path.endswith('.py'), f"Selector must be a Python file: {selector_path}"

    # Validate output directory (create if doesn't exist)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

    # Validate numeric parameters
    assert isinstance(seed, int), f"Seed must be an integer, got: {type(seed)}"
    assert isinstance(n_cpus, int) and n_cpus >= 1, f"n_cpus must be a positive integer, got: {n_cpus}"

    logger.info("All input validations passed.")


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_data(data_dir: str, split_dir: str) -> Tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray,
                                                       List[str]]:
    """
    Load data from CSV and partition into training and testing sets.

    Args:
        data_dir: Directory containing data.csv file.
        split_dir: Directory containing train/test split index files.

    Returns:
        Tuple containing:
            - X_train: Training features
            - y_train: Training targets
            - X_test: Testing features
            - y_test: Testing targets
            - feature_names: List of feature column names
    """
    # Load the CSV data
    data_csv_path = os.path.join(data_dir, 'data.csv')
    df = pd.read_csv(data_csv_path)
    logger.info(f"Loaded data with shape: {df.shape}")

    # Identify target column (labeled 'y') and feature columns
    assert 'y' in df.columns, "Target column 'y' not found in data.csv"
    feature_cols = [col for col in df.columns if col != 'y']

    X = df[feature_cols].values
    y = df['y'].values

    logger.info(f"Number of features: {len(feature_cols)}")
    logger.info(f"Number of samples: {len(y)}")

    # Load split indices
    train_indices = np.load(os.path.join(split_dir, 'training.npy'))
    test_indices = np.load(os.path.join(split_dir, 'testing.npy'))

    logger.info(f"Training samples: {len(train_indices)}")
    logger.info(f"Testing samples: {len(test_indices)}")

    # Partition data
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, y_train, X_test, y_test, feature_cols

def load_selection_function(selector_path: str) -> Callable:
    """
    Dynamically load the selection function from a Python file.

    Args:
        selector_path: Path to the selector.py file containing the 'selection' function.

    Returns:
        The selection function callable.

    Raises:
        AssertionError: If the selection function is not found in the file.
    """
    spec = importlib.util.spec_from_file_location("selector_module", selector_path)
    assert spec is not None, f"Could not load module spec from {selector_path}"
    assert spec.loader is not None, f"Module spec has no loader for {selector_path}"
    selector_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector_module)

    assert hasattr(selector_module, 'selection'), \
        f"'selection' function not found in {selector_path}"

    logger.info(f"Loaded selection function from: {selector_path}")
    return selector_module.selection


# =============================================================================
# GP PRIMITIVE SET CONFIGURATION
# =============================================================================

def create_primitive_set(num_features: int, feature_names: List[str]) -> gp.PrimitiveSetTyped:
    """
    Create the DEAP primitive set with all required operators and terminals.

    The primitive set includes:
    - 14 non-terminals: add, sub, mul, div, sqrt, log, abs, neg, inv, max, min, sin, cos, tan
    - (num_features + 1) terminals: feature variables + ephemeral random constant

    Args:
        num_features: Number of feature columns in the dataset.
        feature_names: Names of the feature columns.

    Returns:
        Configured DEAP PrimitiveSet.
    """
    # Create primitive set with the correct number of input arguments
    pset = gp.PrimitiveSet("MAIN", num_features)

    # Rename arguments to match feature names
    for i, name in enumerate(feature_names):
        pset.renameArguments(**{f"ARG{i}": name})

    # Add binary operators (arity = 2)
    pset.addPrimitive(safe_add, 2, name="add")
    pset.addPrimitive(safe_sub, 2, name="sub")
    pset.addPrimitive(safe_mul, 2, name="mul")
    pset.addPrimitive(protected_div, 2, name="div")
    pset.addPrimitive(safe_max, 2, name="max")
    pset.addPrimitive(safe_min, 2, name="min")

    # Add unary operators (arity = 1)
    pset.addPrimitive(protected_sqrt, 1, name="sqrt")
    pset.addPrimitive(protected_log, 1, name="log")
    pset.addPrimitive(safe_abs, 1, name="abs")
    pset.addPrimitive(safe_neg, 1, name="neg")
    pset.addPrimitive(protected_inv, 1, name="inv")
    pset.addPrimitive(safe_sin, 1, name="sin")
    pset.addPrimitive(safe_cos, 1, name="cos")
    pset.addPrimitive(protected_tan, 1, name="tan")

    # Add ephemeral random constant terminal (uniformly sampled between -1 and 1)
    pset.addEphemeralConstant("ERC", generate_erc)

    logger.info(f"Created primitive set with {len(pset.primitives[pset.ret])} primitives "
                f"and {len(pset.terminals[pset.ret])} terminals")

    return pset


# =============================================================================
# TREE EVALUATION
# =============================================================================

def evaluate_tree(individual: gp.PrimitiveTree,
                  compile_func: Callable,
                  X_train: np.ndarray,
                  y_train: np.ndarray,
                  max_size: int) -> Tuple[Tuple[float, ...], Optional[List[float]]]:
    """
    Evaluate a GP tree on the training data.

    Args:
        individual: The GP tree to evaluate.
        compile_func: Function to compile the tree into callable.
        X_train: Training features.
        y_train: Training targets.
        max_size: Maximum allowed tree size for bloat control.

    Returns:
        Tuple containing:
            - (R^2 score,): Tuple with R^2 performance (for DEAP fitness)
            - List of absolute errors for each training sample

    Returns ((-np.inf,), None) if evaluation fails.
    """
    # Bloat control: reject trees that are too large
    if len(individual) > max_size:
        return ((-np.inf,), None)

    try:
        # Compile the tree
        func = compile_func(individual)

        # Make predictions for each sample
        predictions = np.array([func(*x) for x in X_train])

        # Check for invalid predictions
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            return ((-np.inf,), None)

        # Calculate R^2 (coefficient of determination)
        ss_res = np.sum((y_train - predictions) ** 2)
        ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)

        if ss_tot == 0:
            r2 = 0.0
        else:
            r2 = 1.0 - (ss_res / ss_tot)

        # Calculate absolute errors for each sample (for parent selection)
        absolute_errors = list(np.abs(y_train - predictions))

        return ((r2,), absolute_errors)

    except Exception as e:
        logger.debug(f"Evaluation failed: {e}")
        return ((-np.inf,), None)

def calculate_r2(individual: gp.PrimitiveTree,
                 compile_func: Callable,
                 X: np.ndarray,
                 y: np.ndarray) -> float:
    """
    Calculate R^2 score for a tree on given data.

    Args:
        individual: The GP tree to evaluate.
        compile_func: Function to compile the tree.
        X: Feature data.
        y: Target values.

    Returns:
        R^2 score, or -np.inf if evaluation fails.
    """
    try:
        func = compile_func(individual)
        predictions = np.array([func(*x) for x in X])

        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            return -np.inf

        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - (ss_res / ss_tot)

    except Exception:
        return -np.inf

# =============================================================================
# OFFSPRING GENERATION
# =============================================================================

# Ray remote function for parallel parent selection
@ray.remote
def ray_select_parent(selection_func: Callable,
                      fitnesses_per_sample: List[List[float]],
                      idx: int) -> Tuple[int, int]:
    """
    Ray remote function to perform parent selection.

    Args:
        selection_func: The custom selection function.
        fitnesses_per_sample: List of error lists for each individual.
        idx: Index position where the result should be stored.

    Returns:
        Tuple of (storage_index, selected_parent_index).
    """
    selected_parent_idx = selection_func(fitnesses_per_sample)
    return (idx, selected_parent_idx)


def generate_offspring(population: List,
                       fitnesses_per_sample: List[List[float]],
                       selection_func: Callable,
                       toolbox: base.Toolbox,
                       pop_size: int,
                       cxpb: float,
                       mutpb: float,
                       max_height: int,
                       rng: np.random.Generator) -> List:
    """
    Generate offspring for the next generation.

    The process:
    1. Determine how many offspring via crossover vs mutation only
    2. Select the appropriate number of parents (using Ray for parallelization)
    3. Apply crossover and/or mutation operations

    Args:
        population: Current population of individuals.
        fitnesses_per_sample: List of error lists for each individual.
        selection_func: The custom selection function.
        toolbox: DEAP toolbox with genetic operators.
        pop_size: Target population size.
        cxpb: Crossover probability.
        mutpb: Mutation probability.
        max_height: Maximum tree height for bloat control.
        rng: Numpy random generator for reproducibility.

    Returns:
        List of offspring individuals.
    """
    offspring = []


    # Determine offspring generation method for each individual
    # crossover_rate determines proportion of offspring via crossover
    # mutation_rate determines proportion via mutation only
    n_crossover = list(rng.choice(['m', 'c'], pop_size, p=[mutpb, cxpb])).count('c')
    n_mutation_only = pop_size - n_crossover

    # Select parents for crossover (need 2 parents per offspring)
    n_crossover_parents = 2 * n_crossover
    # Select parents for mutation only (need 1 parent per offspring)
    n_mutation_parents = n_mutation_only

    total_parents_needed = n_crossover_parents + n_mutation_parents

    # Initialize selected_parents with -1 placeholder values
    selected_parents = [-1] * total_parents_needed

    # Put fitnesses_per_sample in Ray's object store once to avoid repeated serialization
    fitnesses_ref = ray.put(fitnesses_per_sample)

    # Submit Ray tasks for parallel parent selection
    pending_refs = []
    for i in range(total_parents_needed):
        ref = ray_select_parent.remote(selection_func, fitnesses_ref, i)
        pending_refs.append(ref)

    # Process Ray tasks as they complete using ray.wait
    while pending_refs:
        done_refs, pending_refs = ray.wait(pending_refs, num_returns=1)
        for done_ref in done_refs:
            storage_idx, parent_idx = ray.get(done_ref)
            selected_parents[storage_idx] = toolbox.clone(population[parent_idx])

    # Assert that all parents have been selected (no -1 values remain)
    assert all(p != -1 for p in selected_parents), \
        "Error: Not all parents were selected. Some positions still contain -1."

    parent_idx = 0

    # Available mutation operators
    mutation_operators = [toolbox.mutUniform, toolbox.mutNodeReplacement,
                          toolbox.mutShrink, toolbox.mutInsert]

    # Generate offspring via crossover
    for _ in range(n_crossover):
        parent1 = selected_parents[parent_idx]
        parent2 = selected_parents[parent_idx + 1]
        parent_idx += 2

        # Apply crossover using DEAP's one-point crossover for GP trees
        child1, child2 = toolbox.mate(parent1, parent2)

        # Delete fitness values as individuals have been modified
        del child1.fitness.values
        del child2.fitness.values

        # Use child1 as the offspring (randomly could also use child2)
        child = child1

        # Potentially apply mutation to crossover offspring
        if rng.random() < mutpb:
            # Randomly select a mutation operator
            mutate_op = mutation_operators[int(rng.integers(0, len(mutation_operators)))]
            child, = mutate_op(child)
            del child.fitness.values

        # Apply height limit
        if child.height > max_height:
            # If child is too tall, use the original parent
            child = toolbox.clone(population[int(rng.integers(0, len(population)))])

        offspring.append(child)

    # Generate offspring via mutation only
    for i in range(n_mutation_only):
        parent = selected_parents[parent_idx]
        parent_idx += 1

        # Randomly select a mutation operator
        mutate_op = mutation_operators[int(rng.integers(0, len(mutation_operators)))]
        child, = mutate_op(parent)
        del child.fitness.values

        # Apply height limit
        if child.height > max_height:
            # If child is too tall, use the original parent
            child = toolbox.clone(population[int(rng.integers(0, len(population)))])

        offspring.append(child)

    return offspring


# =============================================================================
# POST-HOC ANALYSIS
# =============================================================================

def post_hoc_analysis(population: List,
                      population_r2: List[float],
                      toolbox: base.Toolbox,
                      X_test: np.ndarray, y_test: np.ndarray,
                      rng: np.random.Generator) -> Tuple[Optional[gp.PrimitiveTree], float, float]:
    """
    Perform post-hoc analysis to select the final solution.

    Process:
    1. Find trees tied for the best training R^2
    2. Randomly select one if multiple are tied
    3. Evaluate on test set

    Args:
        population: Final population of individuals.
        population_r2: Training R^2 for each individual.
        toolbox: DEAP toolbox with compile function.
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        rng: Random number generator.

    Returns:
        Tuple of (best_tree, test_r2, train_r2).
    """
    # Step 1: Find solutions tied for best training R^2
    valid_indices = [i for i, r2 in enumerate(population_r2) if r2 > -np.inf]

    if not valid_indices:
        logger.error("No valid solutions found in final population!")
        return None, -np.inf, -np.inf

    valid_r2 = [population_r2[i] for i in valid_indices]
    max_train_r2 = max(valid_r2)

    # Find all solutions tied for the best training R^2
    tied_candidates = [valid_indices[i] for i, r2 in enumerate(valid_r2)
                       if r2 == max_train_r2]

    logger.info(f"Training R^2 max: {max_train_r2:.6f}")
    logger.info(f"Solutions tied for best training R^2: {len(tied_candidates)}")

    # Step 2: Select final candidate (randomly if multiple tied)
    if len(tied_candidates) > 1:
        choice_idx = int(rng.integers(0, len(tied_candidates)))
        final_idx = tied_candidates[choice_idx]
        logger.info("Multiple tied solutions; randomly selected one.")
    else:
        final_idx = tied_candidates[0]

    final_tree = population[final_idx]
    final_train_r2 = population_r2[final_idx]

    # Step 3: Evaluate on test set
    final_test_r2 = calculate_r2(final_tree, toolbox.compile, X_test, y_test)

    logger.info(f"Final solution - Train R^2: {final_train_r2:.6f}, "
                f"Test R^2: {final_test_r2:.6f}")

    return final_tree, final_test_r2, final_train_r2


# =============================================================================
# OUTPUT SAVING
# =============================================================================

def save_outputs(output_dir: str,
                 tree: gp.PrimitiveTree,
                 test_r2: float,
                 train_r2: float) -> None:
    """
    Save the final outputs to the specified directory.

    Saves three files:
    1. final_tree.txt - Human-readable tree expression
    2. test_r2.txt - Test R^2 score
    3. training_r2.txt - Training R^2 score

    Args:
        output_dir: Directory to save outputs.
        tree: The final GP tree solution.
        test_r2: Test R^2 performance.
        train_r2: Training R^2 performance.
    """
    # Save tree expression
    tree_path = os.path.join(output_dir, 'final_tree.txt')
    with open(tree_path, 'w') as f:
        f.write(str(tree))
    logger.info(f"Saved tree expression to: {tree_path}")

    # Save test R^2
    test_path = os.path.join(output_dir, 'test_r2.txt')
    with open(test_path, 'w') as f:
        f.write(f"{test_r2}\n")
    logger.info(f"Saved test R^2 to: {test_path}")

    # Save training R^2
    train_path = os.path.join(output_dir, 'training_r2.txt')
    with open(train_path, 'w') as f:
        f.write(f"{train_r2}\n")
    logger.info(f"Saved training R^2 to: {train_path}")


# =============================================================================
# MAIN EVOLUTIONARY LOOP
# =============================================================================

def run_evolution(data_dir: str,
                  split_dir: str,
                  selector_path: str,
                  seed: int,
                  output_dir: str,
                  n_cpus: int,
                  pop_size: int = 1000,
                  n_generations: int = 100,
                  cxpb: float = 0.8,
                  mutpb: float = 0.2,
                  max_height: int = 50,
                  max_size: int = 500) -> None:
    """
    Run the complete evolutionary GP process.

    Args:
        data_dir: Directory containing data.csv.
        split_dir: Directory with train/val/test split files.
        selector_path: Path to selector.py with selection function.
        seed: Random seed for reproducibility.
        output_dir: Directory to save outputs.
        n_cpus: Number of CPUs for parallelization.
        pop_size: Population size (default 1000).
        n_generations: Number of generations (default 100).
        cxpb: Crossover probability (default 0.8).
        mutpb: Mutation probability (default 0.2).
        max_height: Maximum tree height (default 50).
        max_size: Maximum tree size (default 500).
    """
    global toolbox  # Need global for multiprocessing
    global _module_rng  # Module-level rng for ERC generation

    # Set random seeds using numpy's modern random generator
    rng = np.random.default_rng(seed)
    _module_rng = rng  # Set module-level rng for ERC generation

    # Also seed standard random for DEAP internal operations
    random.seed(seed)

    # Initialize Ray for parallel parent selection
    if not ray.is_initialized():
        ray.init(num_cpus=n_cpus, ignore_reinit_error=True)
        logger.info(f"Initialized Ray with {n_cpus} CPUs")

    logger.info(f"Starting GP evolution with seed: {seed}")
    logger.info(f"Population size: {pop_size}, Generations: {n_generations}")
    logger.info(f"Crossover rate: {cxpb}, Mutation rate: {mutpb}")
    logger.info(f"Max height: {max_height}, Max size: {max_size}")
    logger.info(f"Using {n_cpus} CPU(s) for parallelization")

    # Load data
    X_train, y_train, X_test, y_test, feature_names = load_data(data_dir, split_dir)
    num_features = len(feature_names)

    # Load selection function
    selection_func = load_selection_function(selector_path)

    # Create primitive set
    pset = create_primitive_set(num_features, feature_names)

    # Define fitness and individual types
    # Using FitnessMax since we want to maximize R^2
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    # Create toolbox
    toolbox = base.Toolbox()

    # Register tree generation methods
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=6)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    # Register genetic operators
    # Crossover: One-point crossover for GP trees (standard for symbolic regression)
    toolbox.register("mate", gp.cxOnePoint)

    # Mutation: Register all mutation types (one will be randomly selected per mutation)
    # Options: uniform mutation (replaces subtree), node replacement, shrink, insert
    toolbox.register("mutUniform", gp.mutUniform, expr=toolbox.expr, pset=pset)
    toolbox.register("mutNodeReplacement", gp.mutNodeReplacement, pset=pset)
    toolbox.register("mutShrink", gp.mutShrink)
    toolbox.register("mutInsert", gp.mutInsert, pset=pset)

    mutation_operators = ["mutUniform", "mutNodeReplacement", "mutShrink", "mutInsert"]
    logger.info(f"Available mutation operators: {mutation_operators}")

    # Decorate with height limit
    toolbox.decorate("mate", gp.staticLimit(key=lambda ind: ind.height, max_value=max_height))
    toolbox.decorate("mutUniform", gp.staticLimit(key=lambda ind: ind.height, max_value=max_height))
    toolbox.decorate("mutNodeReplacement", gp.staticLimit(key=lambda ind: ind.height, max_value=max_height))
    toolbox.decorate("mutShrink", gp.staticLimit(key=lambda ind: ind.height, max_value=max_height))
    toolbox.decorate("mutInsert", gp.staticLimit(key=lambda ind: ind.height, max_value=max_height))

    # Initialize population
    population = toolbox.population(n=pop_size)
    logger.info(f"Initialized population with {len(population)} individuals")

    # Archive to store all successfully evaluated solutions and their training R^2
    archive = []  # List of (individual, r2) tuples

    # Set up multiprocessing pool
    pool = Pool(processes=n_cpus) if n_cpus > 1 else None

    try:
        # Main evolutionary loop
        for gen in range(n_generations):
            logger.info(f"=== Generation {gen + 1}/{n_generations} ===")

            # Step 1: Tree Evaluation
            eval_func = partial(evaluate_tree,
                                compile_func=toolbox.compile,
                                X_train=X_train,
                                y_train=y_train,
                                max_size=max_size)

            if pool is not None:
                results = pool.map(eval_func, population)
            else:
                results = list(map(eval_func, population))

            # Process evaluation results
            valid_population = []
            valid_errors = []
            population_r2 = []

            for ind, (fitness, errors) in zip(population, results):
                if errors is not None:  # Valid evaluation
                    ind.fitness.values = fitness
                    valid_population.append(ind)
                    valid_errors.append(errors)
                    population_r2.append(fitness[0])

                    # Add to archive
                    archive.append((toolbox.clone(ind), fitness[0]))

            # Remove failed solutions from population
            n_removed = len(population) - len(valid_population)
            if n_removed > 0:
                logger.info(f"Removed {n_removed} failed solutions")

            population = valid_population

            if len(population) == 0:
                logger.error("All solutions failed evaluation! Stopping evolution.")
                break



            # Log statistics for only solutions with positive R^2
            fitnesses = [ind.fitness.values[0] for ind in population if ind.fitness.values[0] > 0.0]
            logger.info(f"Number of solutions with positive R^2: {len(fitnesses)}")
            logger.info(f"Population size: {len(population)}")
            logger.info(f"Best R^2: {max(fitnesses):.6f}")
            logger.info(f"Mean R^2: {np.mean(fitnesses):.6f}")
            logger.info(f"Std R^2: {np.std(fitnesses):.6f}")

            # Skip offspring generation on the last generation
            if gen == n_generations - 1:
                break

            # Step 2 & 3: Parent Selection and Offspring Generation
            offspring = generate_offspring(
                population=population,
                fitnesses_per_sample=valid_errors,
                selection_func=selection_func,
                toolbox=toolbox,
                pop_size=pop_size,
                cxpb=cxpb,
                mutpb=mutpb,
                max_height=max_height,
                rng=rng
            )

            # No replacement strategy: offspring becomes the new population
            population = offspring

        logger.info("Evolution complete!")

        # Extract all unique solutions and their R^2 scores from archive
        # Use the final population for post-hoc analysis
        final_population = []
        final_r2_scores = []

        # Get unique solutions from archive (using string representation as key)
        seen = set()
        for ind, r2 in archive:
            ind_str = str(ind)
            if ind_str not in seen and r2 > -np.inf:
                seen.add(ind_str)
                final_population.append(ind)
                final_r2_scores.append(r2)

        logger.info(f"Total unique solutions in archive: {len(final_population)}")

        # Post-hoc Analysis
        if len(final_population) > 0:
            best_tree, test_r2, train_r2 = post_hoc_analysis(
                population=final_population,
                population_r2=final_r2_scores,
                toolbox=toolbox,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                rng=rng
            )

            if best_tree is not None:
                # Save outputs
                save_outputs(output_dir, best_tree, test_r2, train_r2)
            else:
                logger.error("No valid solution found for output!")
        else:
            logger.error("No solutions available for post-hoc analysis!")

    finally:
        if pool is not None:
            pool.close()
            pool.join()
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown complete")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description='DEAP-based Genetic Programming for Symbolic Regression',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing data.csv file'
    )

    parser.add_argument(
        '--split_dir',
        type=str,
        required=True,
        help='Directory containing training.npy and testing.npy files'
    )

    parser.add_argument(
        '--selector_path',
        type=str,
        required=True,
        help='Path to selector.py file containing the selection function'
    )

    parser.add_argument(
        '--seed',
        type=int,
        required=True,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save output files'
    )

    parser.add_argument(
        '--n_cpus',
        type=int,
        required=True,
        help='Number of CPUs for parallelization'
    )

    # Optional evolutionary parameters (with defaults)
    parser.add_argument(
        '--pop_size',
        type=int,
        default=1000,
        help='Population size'
    )

    parser.add_argument(
        '--n_generations',
        type=int,
        default=100,
        help='Number of generations'
    )

    parser.add_argument(
        '--cxpb',
        type=float,
        default=0.8,
        help='Crossover probability'
    )

    parser.add_argument(
        '--mutpb',
        type=float,
        default=0.2,
        help='Mutation probability'
    )

    parser.add_argument(
        '--max_height',
        type=int,
        default=40,
        help='Maximum tree height for bloat control'
    )

    parser.add_argument(
        '--max_size',
        type=int,
        default=500,
        help='Maximum tree size for bloat control'
    )

    return parser.parse_args()

def main() -> None:
    """
    Main entry point for the GP symbolic regression system.
    """
    args = parse_arguments()

    # Validate inputs
    validate_inputs(
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        selector_path=args.selector_path,
        output_dir=args.output_dir,
        seed=args.seed,
        n_cpus=args.n_cpus
    )

    # Run evolution
    run_evolution(
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        selector_path=args.selector_path,
        seed=args.seed,
        output_dir=args.output_dir,
        n_cpus=args.n_cpus,
        pop_size=args.pop_size,
        n_generations=args.n_generations,
        cxpb=args.cxpb,
        mutpb=args.mutpb,
        max_height=args.max_height,
        max_size=args.max_size
    )

if __name__ == "__main__":
    main()
