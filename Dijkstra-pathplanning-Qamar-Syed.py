import numpy as np
import cv2
from queue import PriorityQueue

# set up start and goal node here
x_start = 6
y_start = 6
x_goal = 100
y_goal = 185

# initialize clearances and graph dimensions and action set
xdim = 400
ydim = 250
clearance = 5
action_set = {(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)}

# class holds the equations for the obstacles
class Map:
    def __init__(self):
        # for the triangular shape, used slopes and made the lines and then shifted them up and down 5 to add the
        # clearance, the shape isn't well suited for scaling with a constant clearance instead of scale factor
        # but this should work
        self.tri_1 = lambda x, y: (x-36) * (180-185) / (80-36) + 185 <= y <= (x - 36) * (210 - 185) / (115 - 36) + 185 + clearance and y >= (x - 80) * (210 - 180) / (115 - 80) + 180 - clearance
        self.tri_2 = lambda x, y: (x-36) * (180-185) / (80-36) + 185 >= y >= (x - 36) * (100 - 185) / (105 - 36) + 185 - clearance and y <= (x-80) * (100-180) / (105-80) + 180 + clearance
        # just added clearance to radius for circle equation
        self.circle = lambda x, y: pow(y - 185, 2) + pow(x - 300, 2) <= pow(40+clearance, 2) and 260-clearance <= x <= 340+clearance
        # added clearance to side lengths for clearance for hexagon, assumed equal sides
        self.hex_1 = lambda x, y: (200 - ((70+clearance) / 2)) <= x <= 200 and (x - (200 - (70+clearance)/2)) * (-(70+clearance) * np.tan(np.pi / 6) / 2) / ((70+clearance)/2) + 100 - (70+clearance) * np.tan(np.pi / 6) / 2 <= y <= (x - (200 - (70+clearance) / 2)) * ((70+clearance) * np.tan(np.pi / 6) / 2) / ((70+clearance)/2) + 100 + (70+clearance) * np.tan(np.pi / 6) / 2
        self.hex_2 = lambda x, y: 200 <= x <= (200 + (70+clearance) / 2) and (x - 200) * ((70+clearance) * np.tan(np.pi / 6) / 2) / ((70+clearance)/2) + 100 - (70+clearance) * np.tan(np.pi / 6) <= y <= (x - 200) * (-(70+clearance) * np.tan(np.pi / 6) / 2) / ((70+clearance)/2) + 100 + (70+clearance) * np.tan(np.pi / 6)
        # added clearance to borders
        self.quad = lambda x, y: x <= 0 + clearance or y <= 0 + clearance or x >= xdim - clearance or y >= ydim - clearance

    def is_obstacle(self, coords):
        # check if a coordinate is in an obstacle space
        return self.circle(coords[0], coords[1]) or self.quad(coords[0], coords[1]) or self.tri_1(coords[0], coords[1]) or self.tri_2(coords[0], coords[1]) or self.hex_1(coords[0], coords[1]) or self.hex_2(coords[0], coords[1])

# node class to hold coordinates, parent node, and current calculated cost
class Node:
    def __init__(self, value, parent=None):
        self.coords = value
        self.value = value
        self.x = value[0]
        self.y = value[1]
        self.parent = parent
        self.dist = np.inf

    # altered eq and hash function to allow nodes to be properly implemented as dictionary key and in priority queue
    def __eq__(self, other):
        if self.coords == other.coords:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.coords)

    def __str__(self):
        return str(self.coords)

    # move functions based on an angle from the origin in intervals of 45 degrees
    def move_0(self):
        new_val = (self.value[0] + 1, self.value[1])
        return Node(new_val, self)

    def move_45(self):
        new_val = (self.value[0] + 1, self.value[1] + 1)
        return Node(new_val, self)

    def move_90(self):
        new_val = (self.value[0], self.value[1] + 1)
        return Node(new_val, self)

    def move_135(self):
        new_val = (self.value[0] - 1, self.value[1] + 1)
        return Node(new_val, self)

    def move_180(self):
        new_val = (self.value[0] - 1, self.value[1])
        return Node(new_val, self)

    def move_225(self):
        new_val = (self.value[0] - 1, self.value[1] - 1)
        return Node(new_val, self)

    def move_270(self):
        new_val = (self.value[0], self.value[1] - 1)
        return Node(new_val, self)

    def move_315(self):
        new_val = (self.value[0] + 1, self.value[1] - 1)
        return Node(new_val, self)

    # function to generate path from a node all the way to the start
    def gen_path(self):
        traceback = []
        counter = self
        while counter.parent:
            traceback.append(counter)
            counter = counter.parent
        traceback.append(counter)
        traceback.reverse()
        return traceback


# Graph class starts with empty dictionary
# key is a node, value holds list with all the associated edges
# holds obstacle map inside to check
class Graph:
    def __init__(self, map_0):
        self.graph = {}
        self.map = map_0

    # function to use to search for the surroundings for a new node and generate graph
    # adds all move function result nodes to the edges list unless they are part of an obstacle
    def gen_nodes(self, node):
        edges = [(node.move_0(), 1), (node.move_45(), 1.4), (node.move_90(), 1), (node.move_135(), 1.4),
                 (node.move_180(), 1), (node.move_225(), 1.4), (node.move_270(), 1), (node.move_315(), 1.4)]

        edges = list(filter(lambda val: not self.map.is_obstacle(val[0].coords), edges))

        self.graph[node] = edges

# initialize data structures for algorithm
q = PriorityQueue()
object_space = Map()
graph = Graph(object_space)


def dijkstras():
    # make start node and check if start and goal are in valid space, initialize cost dictionary
    start = Node((x_start, y_start))
    if object_space.is_obstacle(start.coords) or object_space.is_obstacle((x_goal, y_goal)):
        print("Start or goal in obstacle space")
        return None
    final_dist = {}

    # start the priority queue with the start node
    start.dist = 0
    q.put((0, 0, start))

    # counter just kept for the priority queue as it couldn't compare nodes and needed an intermediary value
    j = 1
    # loop while the priority queue has a node and pop it out and get the distance and node
    while not q.empty():
        curr_dist, k, curr_node = q.get()

        # if the node has already been popped before, ignore it
        # instead of altering the entries into the q, I just have the node ignored if they have already been popped
        # if not seen, set its distance and the node as a key value in the final distance dictionary
        if curr_node in final_dist:
            continue
        final_dist[curr_node] = curr_dist

        # return node if it is goal node
        if curr_node.coords == (x_goal, y_goal):
            return curr_node

        # generate edges and nodes around the current node
        graph.gen_nodes(curr_node)

        # loop through all the adjacent nodes and check distance value
        # if lower than current, update the value in the node and add the new distance and the node to the q
        for neighbor, cost in graph.graph[curr_node]:

            new_dist = curr_dist + cost
            if new_dist < neighbor.dist:
                neighbor.dist = new_dist
                q.put((new_dist, j, neighbor))
                j += 1
    # return Not found if node was not found for some reason
    print("Not found")
    return None

# function called to call the algorithm and visualize it
def visual():
    goal = dijkstras()
    # print another statement and return None if invalid algorithm output
    if not goal:
        print("Invalid, select new nodes")
        return None

    # set up array as completely black image for now, will update later
    frame = np.zeros((ydim+1, xdim+1, 3), np.uint8)

    # check every block and turn white if it is an obstacle
    for x in range(0, xdim+1):
        for y in range(0, ydim+1):
            if object_space.is_obstacle((x, y)):
                frame[ydim-y, x] = (255, 255, 255)

    # use function to generate path from start to goal
    traceback = goal.gen_path()

    # start a video file to write to
    # these settings worked for me on Ubuntu 20.04 to make a video
    output = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 120, (xdim+1, ydim+1))

    # change blocks to yellow as they were searched by the algorithm
    # adds a frame to the video only every 20 blocks to save computing time
    i = 0
    for node in graph.graph.keys():
        x1, y1 = node.coords
        frame[ydim - y1, x1] = (0, 255, 255)
        i += 1
        if i % 20 == 0:
            output.write(frame)

    # colors the path line at the end and adds two seconds worth of frames
    for node in traceback:
        x1, y1 = node.coords
        frame[ydim-y1, x1] = (255, 0, 0)
    for i in range(0, 240):
        output.write(frame)
    output.release()

visual()
