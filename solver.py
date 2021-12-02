from parse import read_input_file, write_output_file, read_output_file
from math import e
import os
import Task


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
        max = 0

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

    return result


def heuristic(task, time):
    val = get_real_value(time, task)
    duration = task.get_duration()
    time_left = 1440 - time
    time_late = time + task.get_duration() - task.get_deadline()
    deadline = task.get_deadline()
    profit = task.get_max_benefit()
    # rate = profit / (duration * max(1, time_late))
    rate = profit / time_left
    return rate

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
    print("New: ", new_profit, "Max: ", max_profit)
    if new_profit > max_profit:
        write_output_file(output_path, [task.get_task_id() for task in output])
        print("BETTER: ","Increase in profit: ", new_profit - max_profit, output_path)

def ids_to_task_objects(ids, input_path):
    scheduled_tasks = []
    all_tasks = read_input_file(input_path)
    for id in ids:
        scheduled_tasks.append(all_tasks[id - 1])

    return scheduled_tasks

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


