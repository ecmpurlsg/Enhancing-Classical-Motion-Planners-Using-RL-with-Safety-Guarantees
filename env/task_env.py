#!/usr/bin/env python3

import rospy
import numpy as np
from numpy import inf
from collections import deque

import robot_env

class TaskEnv(robot_env.RobEnv):
    """
    Minimal environment containing only what's required by:
      • reset()
      • step_cdrl_short_lidar()
    and their direct dependencies.

    Assumes robot_env.RobEnv provides:
      - clear_min_scan_buffer_size(), clear_odom_buffer_size(), clear_costmap()
      - get_scan(), get_odom(), get_gp(), send_vel(), send_goal(), init_robot()
      - (optional) other ROS-plumbing already initialized in your app
    """

    def __init__(self):
        # Timing
        self.rate_task = rospy.Rate(10)

        # State flags / counters
        self.clear_costmap_counter = 0
        self.status_move_base = 0
        self.counter = -1

        # Goals & poses (fill these from your code before calling reset())
        self.list_init_pose = []   # e.g., [Pose(), Pose(), ...]
        self.list_goals = []       # e.g., [Pose(), Pose(), ...]
        self.init_pose = None
        self.goal = None
        self.wp_goal = None



    # =========================
    # ======  RESET  ==========
    # =========================
    def reset(self):
        """
        Minimal reset that:
          1) Rotates through `list_init_pose`/`list_goals` if provided
          2) Initializes env variables and robot pose
          3) Clears costmap & sends a new goal
          4) Seeds waypoint goal
          5) Returns zeroed observation compatible with step_cdrl_short_lidar
        """
        self.counter += 1

        if len(self.list_init_pose) > 0:
            idx = self.counter % len(self.list_init_pose)
            self.init_pose = self.list_init_pose[idx]
            if len(self.list_goals) == len(self.list_init_pose):
                self.goal = self.list_goals[idx]

        self._init_env_variables()

        if self.init_pose is not None:
            self._set_init_pose()

        self.clear_costmap()

        if self.goal is not None:
            self._send_goal()

        self.set_wp_goal()

        # Return zeros matching (velocity, angle, lidar_stack(k=4), reward, done)
        k = 4
        return np.zeros(2), np.zeros(2), np.zeros(720 * k), 0.0, 0

    # ===============================
    # ======  STEP  =================   
    # ===============================
    def step(self, action):
        k = 4  # number of historical LiDAR frames to stack

        # Clear buffers / periodic costmap clear
        self.clear_min_scan_buffer_size()
        self.clear_odom_buffer_size()
        self.clear_costmap_counter += 1
        if self.clear_costmap_counter == 10:
            self.clear_costmap()
            self.clear_costmap_counter = 0

        # Init step vars
        is_done = 0
        velocity = np.zeros(2, dtype=np.float32)
        angle = np.zeros(2, dtype=np.float32)

        # Initial waypoint (for progress computation)
        _, init_point, wp = self.get_dist_and_wp_v2()
        if isinstance(wp, bool):
            return np.zeros(2), np.zeros(2), np.zeros(720 * k), -self.dist_to_goal() * 1.0, 1

        # Apply action and wait a bit
        self.send_vel(action)
        for _ in range(5):
            self.rate_task.sleep()

        # Progress toward waypoint
        travelled_dist = self.get_dist_traveled(init_point, wp)

        # Next observation
        lidar = self.get_stacked_scan(k)
        tries, angle_atan, PSI, gp, gp_len, wp = self.try_get_path_v2()  # get new local goal
        if tries == 10:
            return np.zeros(2), np.zeros(2), lidar, -self.dist_to_goal() * 1.0, 1
        if isinstance(wp, bool):
            return np.zeros(2), np.zeros(2), np.zeros(720 * k), -self.dist_to_goal() * 1.0, 1

        angle[0] = np.sin(angle_atan)
        angle[1] = np.cos(angle_atan)

        odom = self.get_odom()
        velocity[0] = odom.twist.twist.linear.x
        velocity[1] = odom.twist.twist.angular.z

        # Reward & termination
        reward = self.reward(lidar, travelled_dist)
        is_done = self.goal_reached_dist()

        if is_done == 1:
            reward = 10
        if self.status_move_base == 4:  # aborted
            is_done = 1
            reward = -1
        if self.status_move_base == 3:  # goal reached
            is_done = 1
            reward = 10

        if is_done:
            print(f"Distance to goal: {self.dist_to_goal()}")

        return velocity, angle, lidar, reward, is_done

    # ===========================
    # ===== Reset helpers =======
    # ===========================
    def _init_env_variables(self):
        self.status_move_base = 0

    def _set_init_pose(self):
        # Provided by robot_env.PtdrlRobEnv
        self.init_robot(self.init_pose)

    def _send_goal(self):
        # Provided by robot_env.PtdrlRobEnv
        self.send_goal(self.goal)

    def set_wp_goal(self):
        X, Y = self.get_robot_position()
        self.wp_goal = np.array([X, Y], dtype=np.float32)

    # ============================================
    # ======  Direct helpers (shared)  ===========
    # ============================================
    def get_stacked_scan(self, k: int = 4, max_range: float = 3.5, flatten: bool = True):
        """
        Return current laser scan together with the last k history frames.
        Ordering is oldest → newest along axis 0.
        """
        scan = self.get_scan()
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        ranges[ranges == inf] = max_range

        if not hasattr(self, "_lidar_buf"):
            # store k+1 frames total; prime buffer with the first scan
            self._lidar_buf = deque(maxlen=k + 1)
            for _ in range(k + 1):
                self._lidar_buf.append(ranges.copy())

        # Append newest (deque drops oldest automatically)
        self._lidar_buf.append(ranges)

        stacked = np.stack(self._lidar_buf, axis=0)  # (k+1, 720)
        return stacked.reshape(-1) if flatten else stacked

    def try_get_path_v2(self, max_tries: int = 20):
        """Attempt to fetch local-goal angle and related path info."""
        angle_atan, PSI, gp, gp_len, wp = self.get_local_goal_angle_v2()
        tries = 0
        for _ in range(max_tries):
            tries += 1
            if isinstance(gp, bool) and gp is False:
                self.rate_task.sleep()
                angle_atan, PSI, gp, gp_len, wp = self.get_local_goal_angle_v2()
            else:
                break
            self.rate_task.sleep()
        return tries, angle_atan, PSI, gp, gp_len, wp

    def get_local_goal_angle_v2(self):
        """
        Choose a local goal (waypoint) along the global path based on distance
        and heading change, then return its bearing (egocentric).
        """
        gp, X, Y, PSI = self.get_global_path_and_robot_status()
        if isinstance(gp, bool) and gp is False:
            return 0, 0, False, 0, 0

        egocentric = True
        lg = np.zeros(2, dtype=np.float32)
        itr = 0

        los = 3.0
        min_los = 0.5
        if len(gp) > 0:
            lg_flag = 0
            for wp in gp:
                itr += 1
                dist = np.sqrt(np.sum((np.asarray(wp) - np.array([0.0, 0.0])) ** 2))
                angle = 0.0
                if dist > min_los:
                    mid = int(itr / 2)
                    angle = self.calculate_angle(gp[mid], wp - gp[mid])
                if (dist > los) or (30 < angle < 150):
                    lg = wp if egocentric else self.transform_lg(wp, X, Y, PSI)
                    lg_flag = 1
                    break
            if lg_flag == 0:
                lg = gp[-1] if egocentric else self.transform_lg(gp[-1], X, Y, PSI)

        local_goal = np.array([np.arctan2(lg[1], lg[0])], dtype=np.float32)
        return local_goal, PSI, gp, itr, np.array(lg, dtype=np.float32)

    def calculate_angle(self, v1, v2):
        v1 = np.asarray(v1, dtype=np.float32)
        v2 = np.asarray(v2, dtype=np.float32)
        dot = float(np.dot(v1, v2))
        mag = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if mag == 0.0:
            return 0.0
        cos_t = np.clip(dot / mag, -1.0, 1.0)
        return np.degrees(np.arccos(cos_t))

    def get_global_path_and_robot_status(self):
        gp = self.get_gp()
        if (isinstance(gp, bool) and gp is False) or isinstance(gp, int):
            return False, False, False, False

        X, Y, Z, PSI, qt = self.get_robot_status()
        gp = self.transform_gp(gp, X, Y, PSI)
        return gp.T, X, Y, PSI

    def transform_gp(self, gp, X, Y, PSI):
        R_r2i = np.array([[np.cos(PSI), -np.sin(PSI), X],
                          [np.sin(PSI),  np.cos(PSI), Y],
                          [0,            0,           1]])
        R_i2r = np.linalg.inv(R_r2i)
        pi = np.concatenate([gp, np.ones_like(gp[:, :1])], axis=-1)
        pr = R_i2r @ pi.T
        return np.asarray(pr[:2, :])

    def transform_lg(self, wp, X, Y, PSI):
        R_r2i = np.array([[np.cos(PSI), -np.sin(PSI), X],
                          [np.sin(PSI),  np.cos(PSI), Y],
                          [0,            0,           1]])
        R_i2r = np.linalg.inv(R_r2i)
        pi = np.array([[wp[0]], [wp[1]], [1]])
        pr = R_i2r @ pi
        return np.array([pr[0, 0], pr[1, 0]], dtype=np.float32)

    def get_robot_status(self):
        odom = self.get_odom()
        q1 = odom.pose.pose.orientation.x
        q2 = odom.pose.pose.orientation.y
        q3 = odom.pose.pose.orientation.z
        q0 = odom.pose.pose.orientation.w
        X = odom.pose.pose.position.x
        Y = odom.pose.pose.position.y
        Z = odom.pose.pose.position.z
        PSI = np.arctan2(2 * (q0 * q3 + q1 * q2), (1 - 2 * (q2 ** 2 + q3 ** 2)))
        qt = (q1, q2, q3, q0)
        return X, Y, Z, PSI, qt

    def get_dist_and_wp_v2(self):
        """Return (wp_error, current_position, chosen_waypoint)."""
        max_tries = 20
        los = 3.0
        min_los = 0.5

        gp = None
        for _ in range(max_tries):
            gp = self.get_gp()
            if (isinstance(gp, bool) and gp is False) or isinstance(gp, int):
                gp = self.get_gp()
            else:
                break
        if (isinstance(gp, bool) and gp is False) or isinstance(gp, int):
            return False, False, False

        X, Y = self.get_robot_position()
        wpp = np.zeros(2, dtype=np.float32)
        dist = 0.0
        for itr, wp in enumerate(gp, start=1):
            wpp = wp
            dist = self.get_dist(np.array([X, Y], dtype=np.float32), np.asarray(wp, dtype=np.float32))
            ang = 0.0
            if dist > min_los:
                ang = self.calculate_angle(gp[int(itr / 2)] - gp[0], wp - gp[int(itr / 2)])
            if (dist > los) or (30 < ang < 150):
                break

        wp_error = (dist / los) + 0.2
        return wp_error, np.array([X, Y], dtype=np.float32), np.array(wpp, dtype=np.float32)

    def get_dist(self, a, b):
        d = a - b
        return float(np.sqrt(np.sum(d * d)))

    def get_dist_traveled(self, init_point, wp):
        X, Y = self.get_robot_position()
        final_point = np.array([X, Y], dtype=np.float32)
        init_dist = self.get_dist(init_point, wp)
        final_dist = self.get_dist(final_point, wp)
        return init_dist - final_dist

    def goal_reached_dist(self):
        """Return 1 if within 1.0 m of goal, otherwise 0."""
        xy = self._goal_xy()
        if xy is None:
            return 0
        min_dist = 1.0
        X, Y = self.get_robot_position()
        current = np.array([X, Y], dtype=np.float32)
        g = np.array(xy, dtype=np.float32)
        return 1 if self.get_dist(current, g) < min_dist else 0

    def get_robot_position(self):
        odom = self.get_odom()
        return odom.pose.pose.position.x, odom.pose.pose.position.y

    def _goal_xy(self):
        """Safely extract (x, y) from self.goal for both Pose-like or tuple-like goals."""
        if self.goal is None:
            return None
        try:
            return float(self.goal.position.x), float(self.goal.position.y)
        except AttributeError:
            try:
                return float(self.goal[0]), float(self.goal[1])
            except Exception:
                return None

    def dist_to_goal(self):
        xy = self._goal_xy()
        if xy is None:
            return np.inf
        odom = self.get_odom()
        dx = odom.pose.pose.position.x - xy[0]
        dy = odom.pose.pose.position.y - xy[1]
        return float(np.sqrt(dx * dx + dy * dy))

    def reward(self, lidar, travelled_dist):
        """
        Per-step reward:
            r = −0.1 + travelled_dist + min_range − 1.0
            extra −2.0 if stuck & very close (<0.15 m)
        """
        min_range = float(np.min(lidar))
        r_collision = min(min_range - 1.0, 0.0)  # ≤ 0 when obstacle < 1.0 m
        reward = -0.1 + travelled_dist + r_collision
        if abs(travelled_dist) < 0.01 and min_range < 0.15:
            reward -= 2.0
        return reward
