from parse import read_input_file, write_output_file, read_output_file
from math import e
import os
import Task
import random


# Each task(igloo) has:
#     - task_id (int): task id of the Task
#         !-> self.get_task_id()
#     - deadline (int): deadline of the Task
#         !-> self.get_deadline()
#     - duration (int): duration of the Task
#         !-> self.get_duration()
#     - perfect_benefit (float): the benefit recieved from completing the Task anytime before (or on) the deadline
#         !-> self.get_max_benefit()
# A task's profits decay exponentially according to the function:
#     - p_i * (e ** (-0.0170 * s_i)),
#         where p_i is the perfect_benefit and s_i is the number of minutes late the task was completed
#     !-> self.get_late_benefit(minutes_late)

# Goal: We wish to maximize the total profit/benefit!


def solve(tasks):
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list of igloos in order of polishing
    """
    """sorted_tasks = sorted(tasks, key=lambda task: task.get_deadline())

    task = sorted_tasks[0]"""

    curr_time = 0
    result = []

    while curr_time < 1440:
        best = None
        max = float('-inf')

        for task in tasks:
            if curr_time + task.get_duration() <= 1440:
                val = heuristic(task, curr_time)
                if val > max:
                    max = val
                    best = task

        if best == None:
            return result

        curr_time += best.get_duration()
        result.append(best.get_task_id())
        tasks.remove(best)

    best = simulated_annealing(result)

    return best

def heuristic(task, time):
    return heuristic_sage1(task, time)

def heuristic_sage1(task, time):
    # This function gives more weight to functions with better profit over time and real_value / time
    return (task.get_max_benefit() / task.get_duration()) * (get_real_value(time, task) / task.get_duration())

def heuristic_sage2(task, time):
    # This function gives more weight to tasks which have an earlier deadline
    return (1440 - task.get_deadline()) * get_real_value(time, task)

def get_real_value(time, task):
    deadline = task.get_deadline()
    time_late = time + task.get_duration() - deadline

    return task.get_late_benefit(time_late)


def output_profit(tasks):
    profit = 0
    time = 0
    for task in tasks:
        val = get_real_value(time, task)
        profit += val
        time += task.get_duration()

    return profit

def overwrite_if_better(output, best_output):
    new_profit = output_profit(output)
    max_profit = output_profit(best_output)
    if new_profit > max_profit:
        write_output_file(output_path, [task.get_task_id() for task in output])
        print("BETTER: ","Increase in profit: ", new_profit - max_profit, output_path)

def ids_to_task_objects(ids, input_path):
    scheduled_tasks = []
    all_tasks = read_input_file(input_path)
    for id in ids:
        scheduled_tasks.append(all_tasks[id - 1])

    return scheduled_tasks


#simulated annealing attempt "https://medium.com/swlh/how-to-implement-simulated-annealing-algorithm-in-python-ab196c2f56a0"
def simulated_annealing(initial_state):
    """Peforms simulated annealing to find a solution"""
    initial_temp = 90
    final_temp = .1
    alpha = 0.01

    current_temp = initial_temp

    # Start by initializing the current state with the initial state
    current_state = initial_state
    solution = current_state

    while current_temp > final_temp:
        neighbor = random.choice(get_neighbors())

        # Check if neighbor is best so far
        cost_diff = get_cost(self.current_state) - get_cost(neighbor)

        # if the new solution is better, accept it
        if cost_diff > 0:
            solution = neighbor
        # if the new solution is not better, accept it with a probability of e^(-cost/temp)
        else:
            if random.uniform(0, 1) < math.exp(-cost_diff / current_temp):
                solution = neighbor
        # decrement the temperature
        current_temp -= alpha

    return solution

def get_cost(state):
    """Calculates cost of the argument state for your solution."""
    return output_profit(state)

def get_neighbors(tasks, state):
    """Returns neighbors of the argument state for your solution."""

    num_tasks = len(state)

    rand_task = random.randint(1, num_tasks - 1)

    task_to_replace = state[rand_task]

    task_to_replace_duration = task_to_replace.get_duration()

    neighbors = []

    #replace a task with a task we havent used yet

    for task in tasks:
        if task not in state:
            new_state = state.copy()
            new_state[rand_task] = task
            if valid(new_state):
                neighbors.append(new_state)

    #swap the order of 2 tasks

    rand_task2 = random.randint(1, num_tasks - 1)

    task1 = state[rand_task]
    task2 = state[rand_task2]

    new_state = state.copy()

    new_state[rand_task] = task2
    new_state[rand_task2] = task1

    neighbors.append(new_state)

    return neighbors


def valid(permutation):
    total = 1440
    time = 0
    for task in permutation:
        time+= task.get_duration()

    if time <= total:
        return True
    else:
        return False






# Here's an example of how to run your solver.
if __name__ == '__main__':
     for dir in os.listdir('inputs/'):
         for input_path in os.listdir('inputs/' + dir):
             abs_path = 'inputs/' + dir + '/' + input_path
             output_path = 'outputs/' + dir + '/' + input_path[:-3] + '.out'
             # print(abs_path)
             tasks = read_input_file(abs_path)
             output = solve(tasks)
             best_output = read_output_file(output_path)

             output = ids_to_task_objects(output, abs_path)
             best_output =ids_to_task_objects(best_output, abs_path)

             overwrite_if_better(output, best_output)
