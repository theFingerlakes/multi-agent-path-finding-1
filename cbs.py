import time as timer
import heapq
import random
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost
import copy


def detect_collision(path1, path2):
    ##############################
    # Task 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.
    len1 = len(path1)
    len2 = len(path2)
    max_path = max(len1, len2)
    for i in range(max_path):
        location1_1 = get_location(path1, i)
        location2_1 = get_location(path2, i)
        location1_2 = get_location(path1, i-1)
        location2_2 = get_location(path2, i-1)
        if location1_1 == location2_1:
            res = {'loc':[location1_1], 'timestep': i}
            return res
        elif location1_2 == location2_1 and location2_2 == location1_1:
            res = {'loc': [location1_2, location1_1], 'timestep': i}
            return res
    return None



def detect_collisions(paths):
    ##############################
    # Task 3.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.
    res = []
    len_paths = len(paths)
    for x in range(len_paths):
        for y in range(x+1, len_paths):
            collision = detect_collision(paths[x], paths[y])
            if collision != None:
                res.append({'a1': x, 'a2': y, 'loc': collision['loc'], 'timestep': collision['timestep']})
    return res
    

def standard_splitting(collision):
    ##############################
    # Task 3.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep
    len_collision = len(collision['loc'])

    if len_collision == 1:
        res = [{'agent': collision['a1'], 'loc': collision['loc'], 'timestep': collision['timestep']},
                {'agent': collision['a2'], 'loc': collision['loc'], 'timestep': collision['timestep']}]
        return res
    elif len_collision == 2:
        res = [{'agent': collision['a1'], 'loc': collision['loc'], 'timestep': collision['timestep']},
                {'agent': collision['a2'], 'loc': [collision['loc'][1], collision['loc'][0]], 'timestep': collision['timestep']}]
        return res
    return None


def disjoint_splitting(collision):
    ##############################
    # Task 4.1: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint enforces one agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the same agent to be at the
    #                            same location at the timestep.
    #           Edge collision: the first constraint enforces one agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the same agent to traverse the
    #                          specified edge at the specified timestep
    #           Choose the agent randomly
    test = random.randint(0, 1)
    len_collision = len(collision['loc'])
    
    if test == 0:
        res = [
                {'agent': collision['a1'], 'loc': collision['loc'], 'timestep': collision['timestep'], 'positive': 0},
                {'agent': collision['a1'], 'loc': collision['loc'], 'timestep': collision['timestep'], 'positive': 1}]
        return res

    else:
        if len_collision == 1:
            res = [
                {'agent': collision['a2'], 'loc': collision['loc'], 'timestep': collision['timestep'], 'positive': 0},
                {'agent': collision['a2'], 'loc': collision['loc'], 'timestep': collision['timestep'], 'positive': 1}]
            return res

        else:
            res = [
                {'agent': collision['a2'], 'loc': [collision['loc'][1], collision['loc'][0]], 'timestep': collision['timestep'], 'positive': 0},
                {'agent': collision['a2'], 'loc': [collision['loc'][1], collision['loc'][0]], 'timestep': collision['timestep'], 'positive': 1}]
        return res


def paths_violate_constraint(constraint, paths):
    res = []
    len_constraint_loc = len(constraint['loc'])
    pathLen = len(paths)
    for i in range(pathLen):
        if i == constraint['agent']:
            continue
        curr = get_location(paths[i], constraint['timestep'])
        prev = get_location(paths[i], constraint['timestep'] - 1)
        if len_constraint_loc == 1: # Vertex constraint
            if constraint['loc'][0] == curr:
                res.append(i)
        else: # Edge constraint
            if constraint['loc'][0] == prev or constraint['loc'][1] == curr or constraint['loc'] == [curr, prev]:
                res.append(i)
    return res


class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def find_solution(self, disjoint=True):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        # Task 3.1: Testing
        print(root['collisions'])

        # Task 3.2: Testing
        for collision in root['collisions']:
            print(standard_splitting(collision))

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node()
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit

        openListLen = len(self.open_list)
        while openListLen > 0:
            node = self.pop_node()

            if len(node['collisions']) == 0:
                self.print_results(node)
                return node['paths']

            collision = node['collisions'][0]
            if disjoint:
                constraints = disjoint_splitting(collision)
            else:
                constraints = standard_splitting(collision)
            for constraint in constraints:
                newNode = {'cost':0, 'constraints': [], 'paths': [], 'collisions': []}
                newNode['constraints'] = copy.deepcopy(node['constraints'])
                newNode['constraints'].append(constraint)
                newNode['paths'] = copy.deepcopy(node['paths'])
                agent = constraint['agent']
                path = a_star(self.my_map, self.starts[agent], self.goals[agent], self.heuristics[agent], agent, newNode['constraints'])
                if path is not None:
                    newNode['paths'][agent] = path
                    newNode['collisions'] = detect_collisions(newNode['paths'])
                    newNode['cost'] = get_sum_of_cost(newNode['paths'])
                    if disjoint:
                        viopath = paths_violate_constraint(constraint, newNode['paths']) if constraint['positive'] else []
                        if len(viopath) > 0:
                            continue
                        self.push_node(newNode)
                    else:
                        self.push_node(newNode)

        raise BaseException('No solutions')
        #self.print_results(root)
        #return root['paths']


    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
