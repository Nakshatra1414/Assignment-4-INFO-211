"""
INFO-B 211 Assignment 4
Scipy Modules Implementation

Student Name: Nakshatra
Course: INFO-B 211
Assignment: Scipy Programming Assignment

Description:
This program demonstrates the use of the following Scipy modules:

1. scipy.constants
2. scipy.stats
3. scipy.integrate
4. scipy.interpolate
5. scipy.linalg

The program loads an NBA dataset and performs statistical analysis
along with mathematical computations using Scipy.
"""

# -----------------------------
# Import Required Libraries
# -----------------------------

import pandas as pd
import numpy as np

from scipy import stats
from scipy import integrate
from scipy import interpolate
from scipy import linalg
from scipy import constants


# -----------------------------
# Function: Load Dataset
# -----------------------------

def load_dataset(filename):
    """
    Loads CSV dataset safely.

    Parameters:
        filename (str): Name of dataset file

    Returns:
        DataFrame or None
    """

    try:
        # encoding latin1 avoids decoding errors
        df = pd.read_csv(
            filename,
            encoding='latin1',
            on_bad_lines='skip'
        )

        print("\nDataset Loaded Successfully\n")

        return df

    except FileNotFoundError:

        print("\nERROR: Dataset file not found.")
        print("Make sure nba.csv is in the same folder.\n")

        return None

    except Exception as e:

        print("\nERROR loading dataset:")
        print(e)

        return None


# -----------------------------
# Function: Show Dataset Info
# -----------------------------

def show_dataset_info(df):
    """
    Displays dataset preview and structure.
    """

    print("First 5 Rows of Dataset:\n")

    print(df.head())

    print("\nNumber of Rows:", df.shape[0])
    print("Number of Columns:", df.shape[1])

    print("\nColumn Names:\n")

    print(df.columns)


# -----------------------------
# Function: Scipy Constants
# -----------------------------

def scipy_constants_demo():
    """
    Demonstrates scipy.constants module.
    """

    print("\n==============================")
    print("SCIPY CONSTANTS MODULE")
    print("==============================\n")

    print("Value of Pi =", constants.pi)

    print("Speed of Light =", constants.c, "m/s")

    print("Gravitational Acceleration =", constants.g, "m/s^2")

    print("Boltzmann Constant =", constants.Boltzmann)


# -----------------------------
# Function: Statistics Module
# -----------------------------

def scipy_statistics_demo(df):
    """
    Demonstrates scipy.stats module.
    """

    print("\n==============================")
    print("SCIPY STATISTICS MODULE")
    print("==============================\n")

    try:

        # Automatically choose numeric column
        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns

        column = numeric_columns[0]

        data = df[column].dropna()

        print("Selected Numeric Column:", column)

        print("\nMean =", np.mean(data))

        print("Median =", np.median(data))

        mode_value = stats.mode(data, keepdims=True)

        print("Mode =", mode_value.mode[0])

        print("Standard Deviation =", np.std(data))

    except Exception as e:

        print("Statistics Error:", e)


# -----------------------------
# Function: Integration Module
# -----------------------------

def scipy_integration_demo():
    """
    Demonstrates scipy.integrate module.
    """

    print("\n==============================")
    print("SCIPY INTEGRATION MODULE")
    print("==============================\n")

    try:

        # Integrate x^2 from 0 to 5

        result, error = integrate.quad(
            lambda x: x**2,
            0,
            5
        )

        print("Integral of x^2 from 0 to 5 =", result)

        print("Estimated Error =", error)

    except Exception as e:

        print("Integration Error:", e)


# -----------------------------
# Function: Interpolation Module
# -----------------------------

def scipy_interpolation_demo():
    """
    Demonstrates scipy.interpolate module.
    """

    print("\n==============================")
    print("SCIPY INTERPOLATION MODULE")
    print("==============================\n")

    try:

        x = np.array([0,1,2,3,4])
        y = x**2

        interpolation_function = interpolate.interp1d(x,y)

        value = interpolation_function(2.5)

        print("Interpolated value at 2.5 =", value)

    except Exception as e:

        print("Interpolation Error:", e)


# -----------------------------
# Function: Linear Algebra Module
# -----------------------------

def scipy_linear_algebra_demo():
    """
    Demonstrates scipy.linalg module.
    """

    print("\n==============================")
    print("SCIPY LINEAR ALGEBRA MODULE")
    print("==============================\n")

    try:

        A = np.array([
            [1,2],
            [3,4]
        ])

        B = np.array([5,6])

        solution = linalg.solve(A,B)

        print("Matrix A:\n", A)

        print("\nMatrix B:\n", B)

        print("\nSolution of Ax=B:\n", solution)

    except Exception as e:

        print("Linear Algebra Error:", e)


# -----------------------------
# Main Program
# -----------------------------

def main():

    print("\n===================================")
    print("INFO-B 211 SCIPY ASSIGNMENT")
    print("===================================")

    dataset = load_dataset("nba.csv")

    if dataset is None:
        return

    show_dataset_info(dataset)

    scipy_constants_demo()

    scipy_statistics_demo(dataset)

    scipy_integration_demo()

    scipy_interpolation_demo()

    scipy_linear_algebra_demo()

    print("\n===================================")
    print("Assignment Completed Successfully")
    print("===================================\n")


# Run Program

if __name__ == "__main__":

    main()