
#This file will act as the entry point to your project, prompting students to choose a task:

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")
scale_factor = 1000.0

# Scale the training set's label.
train_df["median_house_value"] /= scale_factor

# Scale the test set's label
test_df["median_house_value"] /= scale_factor

class context:
    train_df = train_df
    test_df = test_df
    scale_factor = scale_factor
    pd = pd
    np = np
    tf = tf
    plt = plt

def main():
    print("Welcome to the Machine Learning Project!")
    print("Please select a task to work on:")
    tasks = {
        "1": "Task 1 -- Experiment with the validation split",
        "2": "Task 2 -- Determine why the loss curves differ",
        "3": "Task 3 -- Fix the problem",
        "4": "Task 4 -- Use the Test Dataset to Evaluate Your Model's Performance"
    }

    for key, task in tasks.items():
        print(f"({key}) : {task}")

    choice = input("Enter the number corresponding to the task: ")

    if choice in tasks:
        task_module = f"tasks.task{choice}"
        try:
            task = __import__(task_module, fromlist=["*"])
            task.run(context)  # Assuming each task file has a run() function.
        except ImportError:
            print(f"Error: Could not find {tasks[choice]} module.")
    else:
        print("Invalid choice. Please run the program again and select a valid task.")

if __name__ == "__main__":
    main()
