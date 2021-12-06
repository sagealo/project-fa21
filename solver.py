from parse import read_input_file, write_output_file, read_output_file
from math import e
import os
import Task
import random
import heapq as heap


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
# NOTE: To prepare submission run `python3 prepare_submission.py outputs/ submission.json`


def solve(tasks, input_path):
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
        result.append(best)
        tasks.remove(best)

    best = simulated_annealing(result, tasks, input_path)

    return best


def heapify_tasks(tasks):
    """Returns list of tuples (start_time, task) sorted by latest possible start time"""
    start_times = []
    for task in tasks:
        start_time = max(0, task.get_deadline() - task.get_duration)
        pair = (start_time, task)
        heap.heappush(start_times, pair)
    return start_times

def output_profit(tasks):
    profit = 0
    time = 0
    for task in tasks:
        val = get_real_value(time, task)
        profit += val
        time += task.get_duration()

    return profit

def solve_brute_force(tasks, curr_time, curr_igloo, result):
    # Either use the task or we don't use the task
    # Pass in tasks sorted by deadline
    if curr_time >= 1440 or curr_igloo >= len(tasks):
        return result
    else:
        use_igloo = result
        if tasks[curr_igloo].get_duration() + curr_time <= 1440:
            use_igloo = solve_brute_force(tasks, curr_time + tasks[curr_igloo].get_duration(), curr_igloo + 1, result + [tasks[curr_igloo]])
        skip_igloo = solve_brute_force(tasks, curr_time, curr_igloo + 1, result)
        if (output_profit(use_igloo) > output_profit(skip_igloo)):
            return use_igloo
        else: 
            return skip_igloo


def heuristic(task, time):
    return favor_little_free_time(task, time)

def favor_profit_over_time(task, time):
    # This function gives more weight to functions with better profit over time and real_value / time
    return (task.get_max_benefit() / task.get_duration()) * (get_real_value(time, task) / task.get_duration())

def favor_early_deadline(task, time):
    # This function gives more weight to tasks which have an earlier deadline
    return ((1440 - task.get_deadline()) + 1) * get_real_value(time, task)

def favor_little_free_time(task, time):
    deadline = task.get_deadline()                                                  # Deadline of task
    time_after_task = time + task.get_duration()                                    # The time after the task is completed
    free_time = deadline - time_after_task                                          # The amount of time between the end of the task and the deadline
    profit = favor_profit_over_time(task, time) * favor_early_deadline(task, time)  # The profit/hueristic function (NOTE: Experiment with this value)
    constant = 1.38                                                                 # The exponential decay value i.e. the later the bigger the free time the worse it is. (NOTE: Experiment with this value)
    if free_time >= 0:  
        return (1440 - (free_time ** constant)) * profit
    else: 
        return profit

def get_real_value(time, task):
    deadline = task.get_deadline()
    time_late = time + task.get_duration() - deadline

    return task.get_late_benefit(time_late)

def overwrite_if_better(output, best_output):
    new_profit = output_profit(output)
    max_profit = output_profit(best_output)
    if new_profit > max_profit:
        write_output_file(output_path, [task.get_task_id() for task in output])
        print("BETTER: ","Percent increase in profit: ", ((new_profit - max_profit) / max_profit) * 100, '%', output_path)

def ids_to_task_objects(ids, input_path):
    scheduled_tasks = []
    all_tasks = read_input_file(input_path)
    for id in ids:
        scheduled_tasks.append(all_tasks[id - 1])

    return scheduled_tasks


#simulated annealing attempt "https://medium.com/swlh/how-to-implement-simulated-annealing-algorithm-in-python-ab196c2f56a0"
def simulated_annealing(initial_state, tasks, input_path):
    """Peforms simulated annealing to find a solution"""
    initial_temp = 90
    final_temp = .1
    alpha = 0.01

    current_temp = initial_temp

    # Start by initializing the current state with the initial state
    current_state = initial_state
    solution = current_state

    while current_temp > final_temp:

        neighbor = random.choice(get_neighbors(current_state, tasks, input_path))

        # Check if neighbor is best so far
        cost_diff = get_cost(current_state) - get_cost(neighbor)

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

def get_neighbors(state, tasks, input_path):
    """Returns neighbors of the argument state for your solution."""

    num_tasks = len(state)

    rand_task = random.randint(1, num_tasks - 1)

    neighbors = []

    #replace a task with a task we havent used yet
    for task in tasks:
        if task not in state:
            new_state = state.copy()
            new_state[rand_task] = task
            if valid(new_state, input_path):
                neighbors.append(new_state)

    #remove a task
    for task in state:
        new_state = state.copy()
        new_state.remove(task)
        neighbors.append(new_state)

    return neighbors


def valid(permutation, input_path):

    total = 1440
    time = 0
    for task in permutation:
        """print(type(task))"""
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
             output = solve(tasks, abs_path)
             print(output_profit(output))
             print("reached")
             best_output = read_output_file(output_path)



             #output = ids_to_task_objects(output, abs_path)
             best_output =ids_to_task_objects(best_output, abs_path)
             print(output_profit(best_output))

             overwrite_if_better(output, best_output)
             print("reached2")
