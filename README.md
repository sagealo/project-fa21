# CS 170 Project Fall 2021

Take a look at the project spec before you get started!

Requirements:

Python 3.6+

Files:

- `parse.py`: functions to read/write inputs and outputs
- `solver.py`: where you should be writing your code to solve inputs
- `Task.py`: contains a class that is useful for processing inputs

When writing inputs/outputs:

- Make sure you use the functions `write_input_file` and `write_output_file` provided
- Run the functions `read_input_file` and `read_output_file` to validate your files before submitting!
- These are the functions run by the autograder to validate submissions

Algorithm:

- First, we started off with a greedy algorithm which used different heuristics to decide which task to choose at a given time. Some heuristics include:
  - Favoring little free time - Pick a task as close to its deadline as possible
  - Favor high profit per minute - Pick a task with highest profit / duration
  - Favor early deadline - Pick a task with early deadline
  - And more!
- These worked at first but then we decided to create a DP algortithm to improve our tasks:
  - DP Subproblem: f(i, t) => maximum profit scheduling igloo 0 through i with a final deadline of t
  - This algorithm resulted in a ~3% increase in profit for a majority of out schedules
