#!/usr/bin/env python3

import rospy
import heapq
import numpy as np
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point
from nav_msgs.msg import GridCells

class AStarPlanner:
    # Setup
    def __init__(self):
        rospy.init_node('astar_planner')

        # map state
        self.map = None
        self.map_info = None # rez, origin, w, h
        self.start = None
        self.goal = None

        # Subscribe to created map
        rospy.Subscriber('/map', OccupancyGrid,self.map_cb)
        # Subscribe to initial pose
        rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.start_cb)
        # Subscribe to move_base
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_cb)

        # Publishers - called in rviz (part 3)
        self.pub_frontier = rospy.Publisher('/frontier_cells', GridCells, queue_size=10)
        self.pub_expanded = rospy.Publisher('/expanded_cells', GridCells, queue_size=10)
        self.pub_path_cells = rospy.Publisher('/path_cells', GridCells, queue_size=10)
        self.pub_nav_path = rospy.Publisher('/planned_path', Path, queue_size=10)

        rospy.spin()

    #------------------Callback functions----------------
    # map
    def map_cb(self, msg):
        # get info
        self.map_info = msg.info
        # save info
        self.map = np.array(msg.data).reshape((msg.info.height, msg.info.width))

    # called when 2d pose estimate is used
    def start_cb(self, msg):
        self.start = self.world_to_grid(
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )

    # 2d nav goal
    def goal_cb(self, msg):
        self.goal = self.world_to_grid(msg.pose.position.x, msg.pose.position.y)

        if self.start is not None and self.map is not None:
            self.astar()


    #--------------Coordinate Functions---------------
    # meters --> grid
    def world_to_grid(self, wx, wy):
        # resolution
        res = self.map_info.resolution
        # origin
        origin = self.map_info.origin.position
        # convert
        col = int((wx - origin.x) / res)
        row = int((wy - origin.y) / res)
        return (col, row)
    # grid --> meters
    def grid_to_world(self, col, row):
        res = self.map_info.resolution
        origin = self.map_info.origin.position
        wx = origin.x + (col + 0.5) * res
        wy = origin.y + (row + 0.5) * res
        return (wx, wy)
    # check if neighbor cell is free
    def is_free(self, col, row):
        w = self.map_info.width
        h = self.map_info.height
        # check bounds
        if col < 0 or row < 0 or col >= w or row >= h:
            return False
        return self.map[row][col] == 0

    def make_gridcells(self, cells):
        #create empty grid
        msg = GridCells()
        # tell rviz grid is in map
        msg.header.frame_id = 'map'
        msg.header.stamp = rospy.Time.now()
        # size of cells (m)
        msg.cell_width = self.map_info.resolution
        msg.cell_height = self.map_info.resolution
        for (col, row) in cells:
            # convert
            wx, wy = self.grid_to_world(col, row)
            p = Point()
            p.x, p.y, p.z = wx, wy, 0.0
            # add point
            msg.cells.append(p)
        return msg

    
    #------------------- Main A* algorithm----------------------
    def astar(self):
        # setup
        start, goal = self.start, self.goal
        # heuristic - euclidean
        def euclidean(a, b):
            return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

        # priority queue
        open_heap = []
        heapq.heappush(open_heap, (euclidean(start, goal), 0.0, start))

        # cells for calcs
        came_from = {start: None}
        g_score = {start: 0.0}
        frontier = set([start])
        expanded = set()

        # All possible neighbor locations
        neighbors = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        rate = rospy.Rate(20)

        # main loop
        while open_heap:
            # pop lowest scored node
            f, g, current = heapq.heappop(open_heap)
            if current in expanded:
                continue
            
            frontier.discard(current)
            expanded.add(current)

            # check if node is goal - done
            if current == goal:
                rospy.loginfo("Goal found!")
                path = self.reconstruct_path(came_from, goal)
                self.publish_results(path)
                return

            # goal not found - check neighbors
            for dc, dr in neighbors:
                nb = (current[0]+dc, current[1]+dr)
                if not self.is_free(*nb) or nb in expanded:
                    continue

                step = np.sqrt(2) if dc != 0 and dr != 0 else 1.0
                tg   = g + step

                if tg < g_score.get(nb, float('inf')):
                    came_from[nb] = current
                    g_score[nb]   = tg
                    heapq.heappush(open_heap, (tg + euclidean(nb, goal), tg, nb))
                    frontier.add(nb)
            # Show the search
            self.pub_frontier.publish(self.make_gridcells(list(frontier)))
            self.pub_expanded.publish(self.make_gridcells(list(expanded)))
            rate.sleep()

    # Reconstruct path
    def reconstruct_path(self, came_from, goal):
        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = came_from[node]
        path.reverse()
        return self.optimize_path(path)

    # remove points on straight line (redundant)
    def optimize_path(self, path):
        if len(path) < 3:
            return path
        optimized = [path[0]]
        for i in range(1, len(path) - 1):
            prev = optimized[-1]
            current = path[i]
            nxt = path[i+1]
            # get vectors
            d1 = (current[0]-prev[0], current[1]-prev[1])
            d2 = (nxt[0]-current[0],  nxt[1]-current[1])
            # ross product
            cross = d1[0]*d2[1] - d1[1]*d2[0]
            # if =0 --> colinear
            # if != 0 --> robot is turning
            if cross != 0:
                optimized.append(current)
        optimized.append(path[-1])
        return optimized

    # publish results
    def publish_results(self, path):
        self.pub_path_cells.publish(self.make_gridcells(path))

        nav_path = Path()
        nav_path.header.frame_id = 'map'
        nav_path.header.stamp    = rospy.Time.now()

        for (col, row) in path:
            wx, wy = self.grid_to_world(col, row)
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0
            nav_path.poses.append(pose)
        self.pub_nav_path.publish(nav_path)


if __name__ == '__main__':
    AStarPlanner()