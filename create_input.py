import random

def create_input(n, file):
    # Define the max values for t, d, and p
    t_max = 1440
    d_max = 60
    p_max = 100
    # Open the file
    f = open(file, 'w')
    # Write n to the file
    f.write(str(n) + '\n')
    # Create random with seed of the desired input size
    random.seed(n)
    # Loop through n and generate the values
    for i in range(1, n + 1):
        curr_igloo = str(i) + ' '
        curr_t = random.randrange(1, t_max + 1)
        curr_d = random.randrange(1, d_max + 1)
        curr_p = random.uniform(1, p_max)
        curr_igloo += str(curr_t) + ' ' + str(curr_d) + ' ' + str(format(curr_p, ".3f")) + '\n'
        f.write(curr_igloo)

create_input(79, 'inputs/100.in')
create_input(124, 'inputs/150.in')
create_input(198, 'inputs/200.in')
