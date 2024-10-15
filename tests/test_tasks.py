import unittest
import numpy as np
from ..main import  context# Import the context from main.py
from ..tasks import task1, task2, task3, task4

class TestTasks(unittest.TestCase):

    def test_task_answers(self):
        """Check that each task has a non-empty answer."""
        tasks = [task1, task2, task3, task4]
        for i, task in enumerate(tasks, 1):
            with self.subTest(task=f"Task {i}"):
                result = task.test()
                question = result.get("question", "").strip()
                answer = result.get("answer", "").strip()
                self.assertTrue(answer, f"Task {i} has an empty answer for the question: {question}")
                print(f"Task {i}: Answer provided: '{answer}'.")

    def test_task3_model_performance(self):
        """Check that the model in Task 3 has a reasonable RMSE."""
        result = task3.run(context)

        # Let's assume we extract the RMSE from the model's evaluation on the test set.
        test_rmse_threshold = 86.0  # Example threshold, adjust as needed.
        
        # The result from Task 3's `run` should include RMSE or other performance metrics.
        # Here, we assume `results` contains the test RMSE.
        if "results" in result:
            rmse = result["results"]["root_mean_squared_error"]
            self.assertLess(rmse, test_rmse_threshold, 
                f"Model's RMSE {rmse} exceeds threshold {test_rmse_threshold}.")
            print(f"Model's RMSE is {rmse}, which is within the acceptable range.")
        else:
            self.fail("Task 3 did not return 'results' with model evaluation.")

    def test_feedback(self):
        """Provide feedback based on the task answers."""
        feedback = []

        # Check Task 1 feedback
        result1 = task1.test()
        if "no" in result1["answer"]:
            feedback.append("Good job in observing how different validation splits didn't impact training!")

        # Task 2 feedback example
        result2 = task2.test()
        if "not similar" in result2["answer"].lower():
            feedback.append("You've correctly identified the difference between the training and validation sets.")

        # Task 3 feedback example
        result3 = task3.test()
        if "Shuffling" in result3["answer"].lower():
            feedback.append("Nice! Shuffling the training data helps create a more balanced training and validation split.")

        # Task 4 feedback example
        result4 = task4.test()
        if "similar" in result4["answer"].lower():
            feedback.append("Great! Ideally, the error metrics should be close across training, validation, and test sets.")

        print("\n--- Feedback for Students ---")
        for line in feedback:
            print(line)
        print("\nKeep up the good work!")

if __name__ == "__main__":
    unittest.main()
