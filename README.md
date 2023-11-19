# Gradient-Descent-Optimization-Implementation-and-Visualization

## Overview

This repository contains Python code for implementing steepest-descent method to optimize a given objective function and visualizing optimization paths with animated contours. Demonstrates two strategies: fixed and optimal step sizes. Includes Fibonacci search for step size and data saved with Pickle.

## Files

### 1. Gradient_Descent.py

- Implementation of the steepest-descent method using two different step size strategies: optimal step size and fixed step size.
- The `f` function defines the objective function.
- The `Fibonacci` function calculates Fibonacci numbers, and `Fib` returns a list of Fibonacci numbers.
- The `GD_` and `GD` functions implement steepest descent with optimal and fixed step sizes, respectively.
- The script also includes a main block that demonstrates the usage of the implemented methods and saves the optimization path and results to a Pickle file.

### 2. main.py

- Animated visualization of the steepest descent paths using Matplotlib and NumPy.
- Reads the Pickle file generated by `Gradient_Descent.py` to obtain the optimization paths.
- The `create_animation` function generates an animated plot of the optimization process.
- The main block loads the data from the Pickle file, prepares the data for visualization, and saves the animation as a GIF.

## Requirements

- NumPy
- NumDiffTools
- Matplotlib

## Usage

1. Ensure that the required dependencies are installed by running:

   ```bash
   pip install numpy numdifftools matplotlib
2. Run `Gradient_Descent.py` to perform the steepest-descent optimization and save the results to a Pickle file:

   ```bash
   python Gradient_Descent.py
3. Run `main.py` to visualize the steepest-descent paths and generate an animated GIF.
   
   ```bash
   python main.py

