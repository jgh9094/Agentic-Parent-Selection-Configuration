#!/usr/bin/env python3
"""
DEAP-based Genetic Programming for Symbolic Regression.

This module implements a tree-based genetic programming system for evolving
symbolic regression solutions using the DEAP framework. It supports custom
parent selection strategies, parallel evaluation, and comprehensive logging.

Compatible with Python 3.13 and DEAP 1.4.3.

Parallelization Optimizations (Ray):
- Batched tree evaluation: Processes multiple trees per Ray task to reduce overhead
- Batched parent selection: Selects multiple parents per Ray task
- Efficient result collection: Uses ray.wait() with adaptive num_returns for responsive processing
- Object store optimization: Reuses data references to minimize serialization
- Dynamic batch sizing: Automatically adjusts batch size based on CPU count and workload

Author: Agentic Parent Selection Configuration Project
"""

import argparse
import importlib.util
import logging
import os
import random
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import ray
from deap import base, creator, gp, tools
import time

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
# RAY CONFIGURATION
# =============================================================================

# Batch size for Ray task submission (tune based on population size and CPU count)
EVALUATION_BATCH_SIZE = 50
SELECTION_BATCH_SIZE = 100

def calculate_optimal_batch_size(total_items: int, n_cpus: int, min_batch_size: int, max_batch_size: int) -> int:
    """
    Calculate optimal batch size for Ray parallelization.

    Args:
        total_items: Total number of items to process.
        n_cpus: Number of available CPUs.
        min_batch_size: Minimum batch size (avoid too much overhead).
        max_batch_size: Maximum batch size (avoid unbalanced work distribution).

    Returns:
        Optimal batch size.
    """
    if total_items <= n_cpus:
        # If items <= CPUs, use batch size of 1 to maximize parallelism
        return 1

    # Aim for roughly 2-4 batches per CPU for load balancing
    ideal_batch_size = max(1, total_items // (n_cpus * 3))

    # Clamp to min/max bounds
    return max(min_batch_size, min(ideal_batch_size, max_batch_size))


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
    validation_path = os.path.join(split_dir, 'validation.npy')
    testing_path = os.path.join(split_dir, 'testing.npy')
    assert os.path.isfile(training_path), f"training.npy not found in: {split_dir}"
    assert os.path.isfile(validation_path), f"validation.npy not found in: {split_dir}"
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
                                                       np.ndarray, np.ndarray,
                                                       List[str]]:
    """
    Load data from CSV and partition into training, validation, and testing sets.

    Args:
        data_dir: Directory containing data.csv file.
        split_dir: Directory containing train/val/test split index files.

    Returns:
        Tuple containing:
            - X_train: Training features
            - y_train: Training targets
            - X_val: Validation features
            - y_val: Validation targets
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
    val_indices = np.load(os.path.join(split_dir, 'validation.npy'))
    test_indices = np.load(os.path.join(split_dir, 'testing.npy'))

    logger.info(f"Training samples: {len(train_indices)}")
    logger.info(f"Validation samples: {len(val_indices)}")
    logger.info(f"Testing samples: {len(test_indices)}")

    # Partition data
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

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

@ray.remote
def ray_evaluate_trees_batch(individuals: List[gp.PrimitiveTree],
                             pset: gp.PrimitiveSet,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             max_size: int,
                             start_idx: int) -> List[Tuple[int, Tuple[Tuple[float, ...], Optional[List[float]]]]]:
    """
    Ray remote function to evaluate a batch of GP trees on the training data.

    Batching reduces Ray overhead by evaluating multiple trees per task.

    Args:
        individuals: List of GP trees to evaluate.
        pset: Primitive set for compiling trees (serializes better than compile_func).
        X_train: Training features.
        y_train: Training targets.
        max_size: Maximum allowed tree size for bloat control.
        start_idx: Starting index for this batch.

    Returns:
        List of (index, evaluation_result) tuples.
    """
    # Compile function locally in the worker
    compile_func = lambda ind: gp.compile(ind, pset)

    results = []
    for i, individual in enumerate(individuals):
        idx = start_idx + i
        result = evaluate_tree(individual, compile_func, X_train, y_train, max_size)
        results.append((idx, result))

    return results


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
            - (MSE,): Tuple with mean squared error (for DEAP fitness)
            - List of squared errors for each training sample

    Returns ((np.inf,), None) if evaluation fails.
    """
    # Bloat control: reject trees that are too large
    if len(individual) > max_size:
        logger.debug(f"Tree rejected: size {len(individual)} > max_size {max_size}")
        return ((np.inf,), None)

    try:
        # Compile the tree
        func = compile_func(individual)

        # Make predictions for each sample
        predictions = np.array([func(*x) for x in X_train])

        # Check for invalid predictions
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            logger.debug(f"Tree rejected: invalid predictions (NaN or Inf)")
            return ((np.inf,), None)

        # Calculate squared errors for each sample
        squared_errors = (y_train - predictions) ** 2

        # Calculate MSE (mean squared error) from squared errors
        mse = np.mean(squared_errors)

        # Convert squared errors to list for parent selection
        squared_errors_list = list(squared_errors)

        return ((mse,), squared_errors_list)

    except Exception as e:
        logger.debug(f"Evaluation failed: {e}")
        return ((np.inf,), None)

def calculate_mse(individual: gp.PrimitiveTree,
                  compile_func: Callable,
                  X: np.ndarray,
                  y: np.ndarray) -> float:
    """
    Calculate MSE for a tree on given data.

    Args:
        individual: The GP tree to evaluate.
        compile_func: Function to compile the tree.
        X: Feature data.
        y: Target values.

    Returns:
        MSE, or np.inf if evaluation fails.
    """
    try:
        func = compile_func(individual)
        predictions = np.array([func(*x) for x in X])

        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            return np.inf

        mse = np.mean((y - predictions) ** 2)

        return mse

    except Exception:
        return np.inf

@ray.remote
def ray_calculate_mse_batch(individuals: List[gp.PrimitiveTree],
                            pset: gp.PrimitiveSet,
                            X: np.ndarray,
                            y: np.ndarray,
                            start_idx: int) -> List[Tuple[int, float]]:
    """
    Ray remote function to calculate MSE for a batch of GP trees on given data.

    Batching reduces Ray overhead by evaluating multiple trees per task.

    Args:
        individuals: List of GP trees to evaluate.
        pset: Primitive set for compiling trees.
        X: Feature data.
        y: Target values.
        start_idx: Starting index for this batch.

    Returns:
        List of (index, mse) tuples.
    """
    # Compile function locally in the worker
    compile_func = lambda ind: gp.compile(ind, pset)

    results = []
    for i, individual in enumerate(individuals):
        idx = start_idx + i
        mse = calculate_mse(individual, compile_func, X, y)
        results.append((idx, mse))

    return results

# =============================================================================
# OFFSPRING GENERATION
# =============================================================================

@ray.remote
def ray_select_parents_batch(selection_func: Callable,
                             fitnesses_per_sample: List[List[float]],
                             n_parents: int) -> List[int]:
    """
    Ray remote function to perform batch parent selection.

    Batching reduces Ray overhead by selecting multiple parents per task.

    Args:
        selection_func: The custom selection function.
        fitnesses_per_sample: List of error lists for each individual.
        n_parents: Number of parents to select in this batch.
        seed: Random seed for this batch (for reproducibility).

    Returns:
        List of selected parent indices.
    """

    selected_parents = []
    for _ in range(n_parents):
        parent_idx = selection_func(fitnesses_per_sample)
        selected_parents.append(parent_idx)

    return selected_parents


def generate_offspring(population: List,
                       fitnesses_per_sample: List[List[float]],
                       selection_func: Callable,
                       toolbox: base.Toolbox,
                       pop_size: int,
                       cxpb: float,
                       mutpb: float,
                       max_height: int,
                       rng: np.random.Generator,
                       n_cpus: int) -> List:
    """
    Generate offspring for the next generation.

    The process:
    1. Determine how many offspring via crossover vs mutation only
    2. Select the appropriate number of parents (using batched Ray for parallelization)
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
    n_crossover = list(rng.choice(['m', 'c'], pop_size, p=[mutpb, cxpb])).count('c')
    n_mutation_only = pop_size - n_crossover

    # Select parents for crossover (need 2 parents per offspring)
    n_crossover_parents = 2 * n_crossover
    # Select parents for mutation only (need 1 parent per offspring)
    n_mutation_parents = n_mutation_only
    # Total parents needed
    total_parents_needed = n_crossover_parents + n_mutation_parents

    if total_parents_needed == 0:
        return offspring

    # Put fitnesses_per_sample in Ray's object store once to avoid repeated serialization
    fitnesses_ref = ray.put(fitnesses_per_sample)

    # Calculate optimal batch size for parent selection
    batch_size = calculate_optimal_batch_size(
        total_items=total_parents_needed,
        n_cpus=n_cpus,
        min_batch_size=10,
        max_batch_size=SELECTION_BATCH_SIZE
    )
    n_batches = (total_parents_needed + batch_size - 1) // batch_size

    logger.debug(f"Parent selection: {total_parents_needed} parents, {n_batches} batches of ~{batch_size}")

    pending_refs = []
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_parents_needed)
        n_parents_in_batch = batch_end - batch_start


        ref = ray_select_parents_batch.remote(
            selection_func,
            fitnesses_ref,
            n_parents_in_batch,
        )
        pending_refs.append(ref)

    # Collect all results at once (more efficient than incremental ray.wait)
    batch_results = ray.get(pending_refs)

    # Flatten batch results and clone selected individuals
    selected_parent_indices = []
    for batch in batch_results:
        selected_parent_indices.extend(batch)

    selected_parents = [toolbox.clone(population[idx]) for idx in selected_parent_indices]

    assert len(selected_parents) == total_parents_needed, \
        f"Error: Expected {total_parents_needed} parents, got {len(selected_parents)}"

    parent_idx = 0

    # Available mutation operators
    mutation_operators = [toolbox.mutUniform, toolbox.mutNodeReplacement,
                          toolbox.mutShrink, toolbox.mutInsert]

    # Generate offspring via crossover
    for _ in range(n_crossover):
        parent1 = selected_parents[parent_idx]
        parent2 = selected_parents[parent_idx + 1]
        parent_idx += 2

        # Try up to 50 times to generate a valid offspring
        child = None
        for _ in range(50):
            # Clone parents for this attempt
            p1_copy = toolbox.clone(parent1)
            p2_copy = toolbox.clone(parent2)

            # Apply crossover using DEAP's one-point crossover for GP trees
            child1, child2 = toolbox.mate(p1_copy, p2_copy)

            # Delete fitness values as individuals have been modified
            del child1.fitness.values
            del child2.fitness.values

            # Use child1 as the offspring (randomly could also use child2)
            child_candidate = child1

            # Potentially apply mutation to crossover offspring
            if rng.random() < mutpb:
                # Randomly select a mutation operator
                mutate_op = mutation_operators[int(rng.integers(0, len(mutation_operators)))]
                child_candidate, = mutate_op(child_candidate)
                del child_candidate.fitness.values

            # Check height limit
            if child_candidate.height <= max_height:
                child = child_candidate
                break

        # If all attempts failed, randomly return one of the selected parents
        if child is None:
            child = toolbox.clone(parent1 if rng.random() < 0.5 else parent2)

        offspring.append(child)

    # Generate offspring via mutation only
    for _ in range(n_mutation_only):
        parent = selected_parents[parent_idx]
        parent_idx += 1

        # Try up to 50 times to generate a valid offspring
        child = None
        for _ in range(50):
            # Clone parent for this attempt
            parent_copy = toolbox.clone(parent)

            # Randomly select a mutation operator
            mutate_op = mutation_operators[int(rng.integers(0, len(mutation_operators)))]
            child_candidate, = mutate_op(parent_copy)
            del child_candidate.fitness.values

            # Check height limit
            if child_candidate.height <= max_height:
                child = child_candidate
                break

        # If all attempts failed, return the selected parent
        if child is None:
            child = toolbox.clone(parent)

        offspring.append(child)

    return offspring


# =============================================================================
# POST-HOC ANALYSIS
# =============================================================================

def post_hoc_analysis(population: List,
                      population_mse: List[float],
                      toolbox: base.Toolbox,
                      X_val: np.ndarray,
                      y_val: np.ndarray,
                      X_test: np.ndarray,
                      y_test: np.ndarray,
                      pset: gp.PrimitiveSet,
                      n_cpus: int,
                      rng: np.random.Generator) -> Tuple[Optional[gp.PrimitiveTree], float, float, float]:
    """
    Perform post-hoc analysis to select the final solution.

    Process:
    1. Evaluate all valid models on validation set (parallelized with Ray)
    2. Find solutions with minimum MSE on validation set
    3. Randomly select one if multiple are tied
    4. Evaluate selected tree on test set

    Args:
        population: Final population of individuals.
        population_mse: Training MSE for each individual.
        toolbox: DEAP toolbox with compile function.
        X_val, y_val: Validation data.
        X_test, y_test: Test data.
        pset: Primitive set for compiling trees.
        n_cpus: Number of CPUs for parallelization.
        rng: Random number generator.

    Returns:
        Tuple of (best_tree, test_mse, validation_mse, train_mse).
    """
    # Step 1: Find valid solutions and evaluate them on validation set
    valid_indices = [i for i, mse in enumerate(population_mse) if mse < np.inf]

    if not valid_indices:
        logger.error("No valid solutions found in final population!")
        return None, np.inf, np.inf, np.inf

    # Evaluate all valid solutions on validation set using Ray parallelization
    valid_population = [population[idx] for idx in valid_indices]
    n_valid = len(valid_population)

    logger.info(f"Evaluating {n_valid} valid solutions on validation set using Ray...")

    # Put validation data in Ray's object store
    X_val_ref = ray.put(X_val)
    y_val_ref = ray.put(y_val)
    pset_ref = ray.put(pset)

    # Calculate optimal batch size
    batch_size = calculate_optimal_batch_size(
        total_items=n_valid,
        n_cpus=n_cpus,
        min_batch_size=10,
        max_batch_size=EVALUATION_BATCH_SIZE
    )
    n_batches = (n_valid + batch_size - 1) // batch_size

    logger.debug(f"Validation evaluation: {n_valid} individuals, {n_batches} batches of ~{batch_size}")

    # Submit batched Ray tasks
    pending_refs = []
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_valid)
        batch_individuals = valid_population[batch_start:batch_end]

        ref = ray_calculate_mse_batch.remote(
            batch_individuals,
            pset_ref,
            X_val_ref,
            y_val_ref,
            batch_start
        )
        pending_refs.append(ref)

    # Collect all results
    validation_mses = [None] * n_valid
    batch_results = ray.get(pending_refs)

    for batch in batch_results:
        for idx, mse in batch:
            validation_mses[idx] = mse

    logger.info(f"Validation evaluation complete.")

    # Find minimum validation MSE
    min_val_mse = min(validation_mses)

    # Find all solutions tied for the best validation MSE
    tied_candidates = [valid_indices[i] for i, mse in enumerate(validation_mses)
                       if mse == min_val_mse]

    logger.info(f"Validation MSE min: {min_val_mse:.6f}")
    logger.info(f"Solutions tied for best validation MSE: {len(tied_candidates)}")

    # Step 2: Select final candidate (randomly if multiple tied)
    if len(tied_candidates) > 1:
        choice_idx = int(rng.integers(0, len(tied_candidates)))
        final_idx = tied_candidates[choice_idx]
        logger.info("Multiple tied solutions; randomly selected one.")
    else:
        choice_idx = 0
        final_idx = tied_candidates[0]

    final_tree = population[final_idx]
    final_train_mse = population_mse[final_idx]
    final_val_mse = min_val_mse  # All tied candidates have the same validation MSE

    # Step 3: Evaluate on test set
    final_test_mse = calculate_mse(final_tree, toolbox.compile, X_test, y_test)

    logger.info(f"Final solution - Train MSE: {final_train_mse:.6f}, "
                f"Val MSE: {final_val_mse:.6f}, Test MSE: {final_test_mse:.6f}")

    return final_tree, final_test_mse, final_val_mse, final_train_mse


# =============================================================================
# OUTPUT SAVING
# =============================================================================

def save_outputs(output_dir: str,
                 tree: gp.PrimitiveTree,
                 test_mse: float,
                 val_mse: float,
                 train_mse: float) -> None:
    """
    Save the final outputs to the specified directory.

    Saves four files:
    1. final_tree.txt - Human-readable tree expression
    2. test_mse.txt - Test MSE score
    3. validation_mse.txt - Validation MSE score
    4. training_mse.txt - Training MSE score

    Args:
        output_dir: Directory to save outputs.
        tree: The final GP tree solution.
        test_mse: Test MSE performance.
        val_mse: Validation MSE performance.
        train_mse: Training MSE performance.
    """
    # Save tree expression
    tree_path = os.path.join(output_dir, 'final_tree.txt')
    with open(tree_path, 'w') as f:
        f.write(str(tree))
    logger.info(f"Saved tree expression to: {tree_path}")

    # Save test MSE
    test_path = os.path.join(output_dir, 'test_mse.txt')
    with open(test_path, 'w') as f:
        f.write(f"{test_mse}\n")
    logger.info(f"Saved test MSE to: {test_path}")

    # Save validation MSE
    val_path = os.path.join(output_dir, 'validation_mse.txt')
    with open(val_path, 'w') as f:
        f.write(f"{val_mse}\n")
    logger.info(f"Saved validation MSE to: {val_path}")

    # Save training MSE
    train_path = os.path.join(output_dir, 'training_mse.txt')
    with open(train_path, 'w') as f:
        f.write(f"{train_mse}\n")
    logger.info(f"Saved training MSE to: {train_path}")


# =============================================================================
# MAIN EVOLUTIONARY LOOP
# =============================================================================

def run_evolution(data_dir: str,
                  split_dir: str,
                  selector_path: str,
                  seed: int,
                  output_dir: str,
                  n_cpus: int,
                  pop_size: int = 200,
                  n_generations: int = 50,
                  cxpb: float = 0.8,
                  mutpb: float = 0.2,
                  max_height: int = 17,
                  max_size: int = 100) -> None:
    """
    Run the complete evolutionary GP process.

    Args:
        data_dir: Directory containing data.csv.
        split_dir: Directory with train/val/test split files.
        selector_path: Path to selector.py with selection function.
        seed: Random seed for reproducibility.
        output_dir: Directory to save outputs.
        n_cpus: Number of CPUs for parallelization.
        pop_size: Population size (default 200).
        n_generations: Number of generations (default 50).
        cxpb: Crossover probability (default 0.8).
        mutpb: Mutation probability (default 0.2).
        max_height: Maximum tree height (default 20).
        max_size: Maximum tree size (default 200).
    """
    global _module_rng  # Module-level rng for ERC generation

    # Set random seeds using numpy's modern random generator
    rng = np.random.default_rng(seed)
    _module_rng = rng  # Set module-level rng for ERC generation

    # Also seed standard random for DEAP internal operations
    random.seed(seed)

    # Initialize Ray for parallelization
    if not ray.is_initialized():
        try:
            ray.init(
                num_cpus=n_cpus,
                ignore_reinit_error=True,
                logging_level=logging.ERROR,  # Reduce Ray logging verbosity
                _system_config={
                    "max_io_workers": min(n_cpus, 4),  # Limit I/O workers
                }
            )
            logger.info(f"Initialized Ray with {n_cpus} CPUs")
            logger.info(f"Ray cluster resources: {ray.cluster_resources()}")
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            raise

    logger.info(f"Starting GP evolution with seed: {seed}")
    logger.info(f"Population size: {pop_size}, Generations: {n_generations}")
    logger.info(f"Crossover rate: {cxpb}, Mutation rate: {mutpb}")
    logger.info(f"Max height: {max_height}, Max size: {max_size}")
    logger.info(f"Using {n_cpus} CPU(s) for parallelization")

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_data(data_dir, split_dir)
    num_features = len(feature_names)

    # Load selection function
    selection_func = load_selection_function(selector_path)

    # Create primitive set
    pset = create_primitive_set(num_features, feature_names)

    # Define fitness and individual types
    # Using FitnessMin since we want to minimize MSE
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    # Create toolbox
    toolbox = base.Toolbox()

    # Register tree generation methods
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=0, max_=4)
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

    # Archive to store all successfully evaluated solutions and their training MSE
    archive = []  # List of (individual, mse) tuples

    # Put training data and primitive set in Ray's object store once
    X_train_ref = ray.put(X_train)
    y_train_ref = ray.put(y_train)
    pset_ref = ray.put(pset)

    try:
        # Main evolutionary loop
        for gen in range(n_generations):
            logger.info(f"=== Generation {gen + 1}/{n_generations} ===")

            # Step 1: Tree Evaluation using batched Ray tasks
            batch_size = calculate_optimal_batch_size(
                total_items=len(population),
                n_cpus=n_cpus,
                min_batch_size=10,
                max_batch_size=EVALUATION_BATCH_SIZE
            )
            n_batches = (len(population) + batch_size - 1) // batch_size

            logger.debug(f"Evaluating {len(population)} individuals in {n_batches} batches of ~{batch_size}")

            pending_refs = []
            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(population))
                batch_individuals = population[batch_start:batch_end]

                ref = ray_evaluate_trees_batch.remote(
                    batch_individuals,
                    pset_ref,
                    X_train_ref,
                    y_train_ref,
                    max_size,
                    batch_start
                )
                pending_refs.append(ref)

            # Use ray.wait to collect results as they complete
            # Process in batches for efficiency while maintaining responsiveness
            results = [None] * len(population)
            completed_batches = 0
            while pending_refs:
                # Wait for up to 25% of remaining tasks or at least 1
                num_to_wait = max(1, len(pending_refs) // 4)
                done_refs, pending_refs = ray.wait(pending_refs, num_returns=num_to_wait, timeout=None)

                for done_ref in done_refs:
                    batch_result = ray.get(done_ref)
                    for idx, result in batch_result:
                        results[idx] = result
                    completed_batches += 1

                # Optional: Log progress for long evaluations
                if completed_batches % max(1, n_batches // 4) == 0 and completed_batches < n_batches:
                    logger.debug(f"Evaluation progress: {completed_batches}/{n_batches} batches completed")

            # Process evaluation results
            valid_population = []
            valid_errors = []
            population_mse = []

            for ind, (fitness, errors) in zip(population, results):
                if errors is not None:  # Valid evaluation
                    ind.fitness.values = fitness
                    valid_population.append(ind)
                    valid_errors.append(errors)
                    population_mse.append(fitness[0])

                    # Add to archive
                    archive.append((toolbox.clone(ind), fitness[0]))
                else:
                    logger.debug(f"Individual failed: fitness={fitness}, errors={errors}")

            # Remove failed solutions from population
            n_removed = len(population) - len(valid_population)
            if n_removed > 0:
                logger.info(f"Removed {n_removed} failed solutions")

            population = valid_population

            if len(population) == 0:
                logger.error("All solutions failed evaluation! Stopping evolution.")
                break

            # Log statistics for valid solutions
            fitnesses = [ind.fitness.values[0] for ind in population if ind.fitness.values[0] < np.inf]
            if len(fitnesses) > 0:
                logger.info(f"Population size: {len(population)}, Valid solutions: {len(fitnesses)}")
                logger.info(f"MSE - Best: {min(fitnesses):.6f}, Mean: {np.mean(fitnesses):.6f}, Std: {np.std(fitnesses):.6f}")
            else:
                logger.warning(f"Population size: {len(population)}, but no valid solutions!")

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
                rng=rng,
                n_cpus=n_cpus
            )

            # No replacement strategy: offspring becomes the new population
            population = offspring

        logger.info("Evolution complete!")

        # Extract all unique solutions and their MSE scores from archive
        # Use dict to deduplicate solutions (keeps best MSE for each unique tree structure)
        unique_solutions = {}
        for ind, mse in archive:
            if mse < np.inf:
                ind_str = str(ind)
                # Keep the best MSE (lowest) for each unique tree structure
                if ind_str not in unique_solutions or mse < unique_solutions[ind_str][1]:
                    unique_solutions[ind_str] = (ind, mse)

        final_population = [ind for ind, _ in unique_solutions.values()]
        final_mse_scores = [mse for _, mse in unique_solutions.values()]

        logger.info(f"Total unique solutions in archive: {len(final_population)}")

        # Post-hoc Analysis
        if len(final_population) > 0:
            best_tree, test_mse, val_mse, train_mse = post_hoc_analysis(
                population=final_population,
                population_mse=final_mse_scores,
                toolbox=toolbox,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                pset=pset,
                n_cpus=n_cpus,
                rng=rng
            )

            if best_tree is not None:
                # Save outputs
                save_outputs(output_dir, best_tree, test_mse, val_mse, train_mse)
            else:
                logger.error("No valid solution found for output!")
        else:
            logger.error("No solutions available for post-hoc analysis!")

    finally:
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
    start_time = time.time()
    run_evolution(
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        selector_path=args.selector_path,
        seed=args.seed,
        output_dir=args.output_dir,
        n_cpus=args.n_cpus
    )
    print(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    main()
