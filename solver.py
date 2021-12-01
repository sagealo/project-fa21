from parse import read_input_file, write_output_file
from math import e
import os


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
            if curr_time + task.get_duration()<= 1440:
                val = heuristic(task, curr_time)
                if val > max:
                    max = val
                    best = task

        if best == None:
            return result

        curr_time += task.get_duration()
        result.append(task.get_task_id())
        tasks.remove(task)

    return result


def heuristic(task, timestep):

    val = get_real_value(timestep, task)

    duration = task.get_duration()

    rate = val / duration

    return rate


def get_real_value(time, task):
    deadline = task.get_deadline()

    time_late = time - deadline

    return task.get_late_benefit(time_late)



# Here's an example of how to run your solver.
if __name__ == '__main__':
     for dir in os.listdir('inputs/'):
         for input_path in os.listdir('inputs/' + dir):
             abs_path = 'inputs/' + dir + '/' + input_path
             output_path = 'outputs/' + input_path[:-3] + '.out'
             print(abs_path)
             tasks = read_input_file(abs_path)
             output = solve(tasks)
             write_output_file(output_path, output)
