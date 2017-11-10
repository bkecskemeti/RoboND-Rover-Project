import numpy as np


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:

        if Rover.is_stuck():
            print('WE SEEM TO BE STUCK!')
            Rover.mode = 'perturb'

        if Rover.mode == 'forward': 
            
            if Rover.total_time - Rover.goal_last_seen < Rover.goal_faith:
                # If a goal is believed to lie ahead, switch mode
                Rover.mode = 'approach'
            
            else:
                # Calculate target angle if area ahead looks good
                # Prefer unexplored areas, if not enough, then any navigable
                target_angles = None
                if (Rover.unknown_angles is not None) and (len(Rover.unknown_angles) >= Rover.stop_forward):
                    target_angles = Rover.unknown_angles
                elif len(Rover.nav_angles) >= Rover.stop_forward:
                    target_angles = Rover.nav_angles

                if target_angles is not None:
                    # When velocity is below max, then throttle, else coast
                    # Steer to average angle clipped to the range +/- 15
                    Rover.throttle = Rover.throttle_set if Rover.vel < Rover.max_vel else 0
                    Rover.steer = np.clip(np.mean(target_angles * 180/np.pi), -15, 15)            
                    Rover.brake = 0  
                else:
                    # If not enough navigable pixels ahead, change to stop mode
                    Rover.throttle = 0
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'

        # When we are approaching a goal, steer towards it and stop smoothly when reached
        elif Rover.mode == 'approach':
            
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:  

                slow_dist = 20.0
                stop_dist = 10.0

                if Rover.total_time - Rover.goal_last_seen > Rover.goal_faith:
                    # Give up if goal is not seen for a while
                    Rover.throttle, Rover.brake, Rover.steer = 0, 0, 0
                    Rover.mode = 'forward'
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

            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'

        # Do some random turning if the rover is stuck
        elif Rover.mode == 'perturb':
            print('PERTURBING!', Rover.perturb_max)
            Rover.throttle = 0
            Rover.brake = Rover.brake_set if Rover.vel > 0.2 else 0
            Rover.steer = -30
            Rover.counter += 1
            if Rover.counter >= Rover.perturb_max:
                Rover.counter = 0
                Rover.throttle = -2 * Rover.throttle_set
                Rover.mode = 'forward'

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'

    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    return Rover

