# Custom XGBoost Implementation (From Scratch)

This repository contains a from scratch implementation of XGBoost in Python, designed for classification and regression tasks with a primary focus on fraud detection style, large scale, and imbalanced datasets.

## Overview

The implementation provides a complete gradient boosting framework built on custom decision trees. It supports standard objectives for classification and regression, integrates early stopping and regularization, and is designed to scale efficiently on medium to large tabular datasets using NumPy based vectorized operations.

The repository is intended for:
- Understanding the internal mechanics of XGBoost style algorithms
- Experimenting with performance optimizations in gradient boosting

## Core Components

- **Gradient Boosting Engine**  
  Sequential ensemble learning with configurable learning rates, tree depth, and regularization.

- **Custom Decision Trees**  
  Tree construction implemented from first principles, supporting both regression and classification objectives.


## Optimizations Included

The implementation incorporates several optimizations without relying on external boosting libraries:

- **Sparsity Aware Splitting**  
  Explicit handling of missing and sparse feature values during split selection.

- **Histogram Based (Binned) Training**  
  Continuous features are discretized into bins to reduce memory usage and accelerate split evaluation.

- **Early Stopping and Regularization**  
  Validation based stopping criteria and regularization terms to control overfitting.

- **Parallel Execution**  
  Selected training and evaluation stages are parallelized for improved CPU utilization.

- **Hyperparameter Optimization Support**  
  Integrated workflow for automated hyperparameter search using Optuna.


## Usage

The repository supports the following workflows:
- Training a gradient boosting model with fixed hyperparameters
- Running automated hyperparameter optimization
- Benchmarking performance across multiple datasets

Execution is handled through Python scripts and a simple CLI interface. Users can configure model parameters, optimization settings, and analysis modes through script arguments.

## Dataset Assumptions

Datasets are expected in CSV format, with the target variable in the first column and features in subsequent columns. Both dense and sparse feature representations are supported.

## Dependencies

The project relies on standard Python scientific computing libraries, along with Optuna for hyperparameter optimization and joblib for parallel execution.
