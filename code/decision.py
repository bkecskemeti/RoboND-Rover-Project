import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra 
from supporting_functions import to_polar_coords, closest_point_idx, get_path, neighbourhood, closest_element

# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Check preconditions and set state if necessary
    if Rover.mission_completed():
        Rover.switch_mode('mission_complete')

    elif Rover.is_stuck():
        # If stuck ie no distance is covered despite steering, do some random movements and turns
        Rover.switch_mode('perturb')

    elif Rover.total_time - Rover.goal_last_seen < Rover.goal_faith:
        # If a goal is believed to lie ahead, switch mode
        Rover.switch_mode('approach')

    elif (Rover.nav_angles is None) or (len(Rover.nav_angles) < Rover.stop_forward):
        # If not enough navigable pixels ahead, then stop
        Rover.switch_mode('stop')

    # State dependent actions and state transitions
    if Rover.mode == 'mission_complete':
        # If all samples are collected and we are back where we started, just stop and wait for eternity
        Rover.throttle = 0
        Rover.steer = 0
        Rover.brake = Rover.brake_set if Rover.vel > 0.1 else 0

    elif Rover.mode == 'forward': 
        # Just wander forward towards an average navigable angle for a while (when eg. there's no plan)

        # When velocity is below max, then throttle, else coast
        Rover.throttle = Rover.throttle_set if Rover.vel < Rover.max_vel else 0
        # Steer to average navigable angle clipped to the range +/- 15
        Rover.steer = np.clip(np.mean(Rover.nav_angles) * 180/np.pi, -15, 15)            
        Rover.brake = 0  

        if Rover.total_time - Rover.wander_start > (60.0 if Rover.total_time < 90.0 else 20.0):
            Rover.switch_mode('path')

    elif Rover.mode == 'path':
        # Make plans about which path to follow
        try_plan(Rover)

        if Rover.follow_path is None:
            # Planning unsuccessful, start wandering
            Rover.switch_mode('forward')

        else:
            # There is a plan, follow it
            pos = np.array(Rover.pos, dtype=float)
            # Remove part of path already close enough
            Rover.follow_path = np.delete(Rover.follow_path, neighbourhood(Rover.follow_path, pos, 3.0), axis=0)

            if len(Rover.follow_path) > 0:
                # There are still points left to reach, steer towards the next point
                target_vector = Rover.follow_path[0] - pos
                dist, angle = to_polar_coords(target_vector[0], target_vector[1])

                target_yaw = (angle * 180 / np.pi)
                target_yaw = target_yaw if target_yaw >= 0 else target_yaw + 360.0

                target_steer = closest_element(Rover.nav_angles * 180 / np.pi, target_yaw - Rover.yaw)
                target_vel = Rover.max_vel if abs(target_steer) < 60 else 0

                Rover.steer = np.clip(target_steer, -25, 25)
                Rover.throttle = Rover.throttle_set if Rover.vel < target_vel else 0
                Rover.brake = Rover.brake_set if Rover.vel > target_vel * 1.1 else 0

            else:
                # Set goals reached near our current position
                Rover.set_reached(np.array(Rover.pos, dtype = float))
                # Plan completed, wander again for a while
                Rover.switch_mode('forward')

    elif Rover.mode == 'approach':
        # When we are approaching a goal, steer towards it and stop smoothly when reached
        slow_dist = 20.0
        stop_dist = 10.0

        if Rover.total_time - Rover.goal_last_seen > Rover.goal_faith:
            # Give up if goal is not seen for a while
            Rover.throttle, Rover.brake, Rover.steer = 0, 0, 0
            Rover.switch_mode('forward')

        elif Rover.goal_distance < 0:
            # Otherwise, coast if goal is not seen for some reason at the moment
            Rover.throttle, Rover.brake, Rover.steer = 0, 0, 0

        else:
            # If we see the goal, steer towards it and decrease velocity
            target_velocity = Rover.max_vel * (min(Rover.goal_distance, slow_dist) - stop_dist) / (slow_dist - stop_dist)

            if Rover.vel > target_velocity:
                Rover.throttle = 0
                Rover.brake = 2 * Rover.throttle_set
            else:
                Rover.throttle = Rover.throttle_set
                Rover.brake = 0

            if Rover.near_sample:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set

            # Steer towards goal
            Rover.steer = Rover.goal_angle * 180/np.pi

    elif Rover.mode == 'perturb':
        # If the rover is stuck, do some turning and backwards movements
        if Rover.total_time - Rover.perturb_start > Rover.perturb_turn_time + Rover.perturb_back_time:
            Rover.switch_mode('forward')
        elif Rover.total_time - Rover.perturb_start > Rover.perturb_turn_time:
            Rover.throttle = -2 * Rover.throttle_set
            Rover.brake = 0
            Rover.steer = 0
        else:
            Rover.throttle = 0
            Rover.brake = Rover.brake_set
            Rover.steer = 30 * Rover.perturb_turn_direction

    elif Rover.mode == 'stop':
        # If there is nothing ahead, stop, and turn around
        if Rover.vel > 0.2:
            # If we're in stop mode but still moving keep braking
            Rover.throttle = 0
            Rover.brake = Rover.brake_set
            Rover.steer = 0
        else:
            # Now we're stopped and we have vision data to see if there's a path forward
            if len(Rover.nav_angles) < Rover.go_forward:
                Rover.throttle = 0
                Rover.brake = 0
                Rover.steer = -15
            
            if len(Rover.nav_angles) >= Rover.go_forward:
                # If we're stopped but see sufficient navigable terrain in front then switch mode
                Rover.switch_mode('path')
        
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        # If in a state where want to pickup a rock send pickup command
        Rover.send_pickup = True
        # Set goals reached near our current position
        Rover.set_reached(np.array(Rover.pos, dtype = float))
    
    return Rover


def try_plan(Rover):
    if (Rover.follow_path is None) or len(Rover.follow_path) == 0:
        # Find the shortest path in the world map from the current position to a selected goal

        current_pos = np.array(Rover.pos, dtype=float)

        # Construct graph
        navigable_points = np.transpose(np.nonzero(Rover.navigable > 0)) + [0.5, 0.5]
        unexplored_points = np.transpose(Rover.unexplored.nonzero()) + [0.5, 0.5]
        # Nodes are the navigable points and the unexplored points
        graph_nodes = np.concatenate((navigable_points, unexplored_points))
        # Use Manhattan distance, no edges between unexplored points
        dist_matrix = squareform(pdist(graph_nodes, metric='cityblock') < 1.5)
        unexplored_idx = range(len(navigable_points), len(graph_nodes))
        dist_matrix[unexplored_idx, unexplored_idx] = False

        graph = csr_matrix(dist_matrix)

        source_node = closest_point_idx(navigable_points, current_pos)
        distances, predecessors = dijkstra(graph, directed=False, indices=source_node, return_predecessors=True)

        opt_goal = Rover.get_first_non_reached_goal()

        if Rover.samples_collected == Rover.samples_to_find:
            # If all samples are found, return to the starting position
            goal_pos = np.array(Rover.start_pos, dtype=float)
            target_node = closest_point_idx(navigable_points, goal_pos)
        elif opt_goal is not None:
            # Go back to any samples found but not picked up for some reason
            goal_pos = opt_goal
            target_node = closest_point_idx(navigable_points, goal_pos)
        else:
            # Go to the closest unexplored node
            target_node = unexplored_idx[np.argmin(distances[unexplored_idx])]
            
        Rover.follow_path = get_path(graph_nodes, source_node, target_node, predecessors)

