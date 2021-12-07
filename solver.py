from parse import read_input_file, write_output_file, read_output_file
from math import e
import os
import Task
import random
import heapq as heap
from queue import PriorityQueue


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

    return result

def solve_out_of_order(tasks):
    schedule = [False for _ in range(1440)]
    h = []
    used = []

    while True:
        best = (None, 0)
        max = float('-inf')
        for task in tasks:
            if task not in used:
                if possible_to_schedule(task, schedule):
                    val = best_we_can_do(task, schedule)
                    if val[0] > max:
                        max = val[0]
                        best = (task, val[1])

        if best[0] == None:
            break;
        else:
            heap.heappush(h, (best[1], best[0]))
            schedule_task(best[0], best[1], schedule)
            used.append(best[0])

    ordering = []
    for entry in h:
        print(entry)
    while h:
        task_to_add = heap.heappop(h)
        print("task_to_add: ")
        print(task_to_add)
        ordering.append(task_to_add[1].get_task_id())

    return ordering

def possible_to_schedule(task, schedule):
    """given a task and a schedule, determines if there is a long enough time slot to fit """
    max_period = 0
    curr_period = 0
    ctr = 0
    while ctr < len(schedule):
        if schedule[ctr] == False:
            curr_period += 1
        else:
            if curr_period > max_period:
                max_period = curr_period
            curr_period = 0
        ctr += 1
        """print("curr_period: " + str(curr_period))
        print("max_period: " + str(max_period))"""

    if curr_period > max_period:
        max_period = curr_period

    if max_period >= task.get_duration():
        return True
    else:
        return False

def best_we_can_do(task, schedule):
    """given a task and a schedule, determines the latest point we can schedule it s.t. it has maximum profit,
    returns the following tuple (max_proft, latest point we can schedule it at to produce max possible profit)"""
    best = None
    max = float('-inf')
    start_time = -1
    for minute in range(len(schedule)):
        if can_schedule_at_time(task, minute, schedule):
            val = get_real_value(minute, task)
            if val >= max:
                best = task
                max = val
                start_time = minute

    return (max, start_time)


def can_schedule_at_time(task, start_time, schedule):
    ctr = start_time
    if start_time + task.get_duration() > 1440:
        return False
    while ctr < start_time + task.get_duration():
        if not schedule[ctr] == False:
            return False
        ctr += 1

    return True

def schedule_task(task, start_time, schedule):
    "given a task and a start time, schedule that task at the given start time"
    print("reached")
    print("start time: " + str(start_time))
    print("task: " + str(task))
    ctr = start_time

    while ctr < start_time + task.get_duration():
        schedule[ctr] = True
        ctr += 1

    return



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
def pseudorandom_heuristic(task, time):
    randint = random.randrange(0, 1)
    if randint == 0:
        return favor_profit_over_time(task, time)
    elif randint == 1:
        return favor_early_deadline(task, time)
    else:
        return favor_little_free_time(task, time)

def pseudorandom_combination_heuristic(task, time):
    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)
    r3 = random.uniform(0, 1)
    return (r1 * favor_early_deadline(task, time)) + (r2 * favor_profit_over_time(task, time)) + (r3 * favor_little_free_time(task, time))

def heuristic(task, time):
    return simple_rate(task, time) * favor_early_deadline(task, time)

def favor_profit_over_time(task, time):
    # This function gives more weight to functions with better profit over time and real_value / time
    return (task.get_max_benefit() / task.get_duration()) * (get_real_value(time, task) / task.get_duration())

def simple_rate(task, time):
    return (get_real_value(time, task) / task.get_duration())

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

    print("new_profit: " + str(new_profit))
    print("max_profit: " + str(max_profit))
    if new_profit > max_profit:
        write_output_file(output_path, [task.get_task_id() for task in output])
        print("BETTER: ","Percent increase in profit: ", ((new_profit - max_profit) / max_profit) * 100, '%', output_path)

def ids_to_task_objects(ids, input_path):
    scheduled_tasks = []
    all_tasks = read_input_file(input_path)
    for id in ids:
        scheduled_tasks.append(all_tasks[id - 1])

    return scheduled_tasks

# Here's an example of how to run your solver.
if __name__ == '__main__':
    for i in range(15):
        for dir in os.listdir('inputs/'):
            for input_path in os.listdir('inputs/' + dir):
                abs_path = 'inputs/' + dir + '/' + input_path
                output_path = 'outputs/' + dir + '/' + input_path[:-3] + '.out'
                # print(abs_path)
                tasks = read_input_file(abs_path)
                output = solve(tasks)
                best_output = read_output_file(output_path)
                output = ids_to_task_objects(output, abs_path)
                best_output = ids_to_task_objects(best_output, abs_path)

                overwrite_if_better(output, best_output)
