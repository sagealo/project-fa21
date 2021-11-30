from parse import read_input_file, write_output_file
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
    pass


# Here's an example of how to run your solver.
# if __name__ == '__main__':
#     for input_path in os.listdir('inputs/'):
#         output_path = 'outputs/' + input_path[:-3] + '.out'
#         tasks = read_input_file(input_path)
#         output = solve(tasks)
#         write_output_file(output_path, output)