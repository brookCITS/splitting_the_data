# Validation and Test Sets

## Description

You'll create the validation set by dividing the downloaded training set into two parts:

- a smaller training set
- a validation set

## Objective

- Split a training set into a smaller training set and a validation set.
- Analyze deltas between training set and validation set results.
- Test the trained model with a test set to determine whether your trained model is overfitting.
- etect and fix a common training problem.

## Project Setup

### Prerequisites

- Python 3.8+
- ML libraries: TenseroFlow, Pandas, NumPy, PyPlotLib
- Unit Test: Unittest, Ward

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/brookCITS/splitting_the_data.git
   cd splitting_the_data
   ```
2. **Run the setup script**: The setup script will create a virtual environment, activate it, and install the required packages.

   ```bash
   . ./setup.sh
   ```

3. **Activate the virtual environment (if not automatically activated)**:

- On MacOS/Linux:
  ```bash
  source venv/bin/activate
  ```
- On Windows:
  ```bash
  venv\Scripts\activate
  ```

## High-Level Overview of the Tasks

- **Task1**: Experiment with two or three different values of validation_split. Do different values of validation_split fix the problem?

- **Task2**: Examine the values in the data frame at different indecies.

- **Task3**: Shuffle the data in the data frame before the splitting it.

- **Task4**: Compare the root mean squared error of the model when evaluated on each of the datases.

## Running the Tests

The project uses Ward for testing. To run all the tests, execute the following command:

```bash
ward -p tests/
```
