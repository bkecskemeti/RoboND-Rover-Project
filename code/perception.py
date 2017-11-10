import numpy as np
import cv2
from supporting_functions import get_coords
from supporting_functions import to_polar_coords

# Identify pixels above and/or below given thresholds
def color_thresh(img, lo=(0, 0, 0), hi=(255, 255, 255)):
    color_select = np.zeros_like(img[:,:,0])
    mask = (img[:,:,0] >= lo[0]) & (img[:,:,1] >= lo[1]) & (img[:,:,2] >= lo[2]) & \
           (img[:,:,0] <= hi[0]) & (img[:,:,1] <= hi[1]) & (img[:,:,2] <= hi[2])
    color_select[mask] = 1
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel

# The inverse of rover_coords
def img_coords(img, x_pixel, y_pixel):
    ypos = (img.shape[0] - x_pixel).astype(int)
    xpos = (img.shape[1]/2 - y_pixel).astype(int)
    return xpos, ypos

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(xpix_tran.astype(int), 0, world_size - 1)
    y_pix_world = np.clip(ypix_tran.astype(int), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):    
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    return warped

# Helper class to encompass information about the rover and the world
class Perspective:
    def __init__(self, xpos, ypos, yaw, world_size, dst_size, img_shape):
        # position and orientation of the rover
        self.xpos = xpos
        self.ypos = ypos
        self.yaw = yaw
        # needed for transforming to world coordinates
        self.world_size = world_size
        self.scale = 2 * dst_size
        # needed for perspective transform
        bottom_offset = 6
        self.source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
        self.destination = np.float32([[img_shape[1]/2 - dst_size, img_shape[0] - bottom_offset],
                                       [img_shape[1]/2 + dst_size, img_shape[0] - bottom_offset],
                                       [img_shape[1]/2 + dst_size, img_shape[0] - 2*dst_size - bottom_offset], 
                                       [img_shape[1]/2 - dst_size, img_shape[0] - 2*dst_size - bottom_offset]])
        # don't trust camera pixels too close to the horizon
        self.trust_area = ((0.0, -70.0), (70.0, 70))
        # we will use this enlarged box to keep track of unexplored areas 
        self.mask_area = ((-10.0, -90.0), (80.0, 90))

# Helper class to calculate information about a subset of the terrain
class TerrainSet():
    def __init__(self, persp, camera_thresh_img):
        # apply transformation to rover coordinates
        self.threshed = camera_thresh_img
        self.x_rover, self.y_rover = rover_coords(perspect_transform(self.threshed, persp.source, persp.destination))
        # clip points far away close to the horizon (they're too far anyway and they mess up fidelity)
        self.x_rover = np.clip(self.x_rover.astype(int), persp.trust_area[0][0], persp.trust_area[1][0])
        self.y_rover = np.clip(self.y_rover.astype(int), persp.trust_area[0][1], persp.trust_area[1][1])
        self.x_img, self.y_img = img_coords(camera_thresh_img, self.x_rover, self.y_rover)
        self.warped = np.zeros_like(camera_thresh_img)
        self.warped[self.y_img, self.x_img] = 1
        self.x_world, self.y_world = pix_to_world(self.x_rover, self.y_rover, persp.xpos, persp.ypos, persp.yaw, persp.world_size, persp.scale)

# Helper class to calculate a mask over a given area relative to the rover
class TerrainMask():
    def __init__(self, persp, rover_area):
        self.x_rover, self.y_rover = get_coords(rover_area, 1.0)
        self.x_world, self.y_world = pix_to_world(self.x_rover, self.y_rover,
                                                  persp.xpos, persp.ypos, persp.yaw,
                                                  persp.world_size, persp.scale)

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):

    img = Rover.img

    dst_size = 5

    rock_thresh_lo, rock_thresh_hi = (126, 106, 0), (255, 233, 84)
    navigable_thresh_lo = (170, 170, 170)

    persp = Perspective(xpos = Rover.pos[0], ypos = Rover.pos[1], yaw = Rover.yaw,
                        world_size = Rover.worldmap.shape[0], dst_size = dst_size, img_shape = img.shape)
    
    # Navigable terrain 
    nav_terrain = TerrainSet(persp, color_thresh(img, lo=navigable_thresh_lo))
    
    # Obstacles 
    obs_terrain = TerrainSet(persp, color_thresh(img, hi=navigable_thresh_lo))

    # Rock samples
    rock_terrain = TerrainSet(persp, color_thresh(img, lo=rock_thresh_lo, hi=rock_thresh_hi)) 

    if (rock_terrain.warped.any()):
        rock_dist, rock_ang = to_polar_coords(rock_terrain.x_rover, rock_terrain.y_rover)
        rock_idx = np.argmin(rock_dist)
        rock_x, rock_y = rock_terrain.x_world[rock_idx], rock_terrain.y_world[rock_idx]
        Rover.goal_distance, Rover.goal_angle, Rover.goal_last_seen = rock_dist[rock_idx], rock_ang[rock_idx], Rover.total_time
        Rover.worldmap[rock_y, rock_x, 1] = 255
        print("ROCK IS IN SIGHT!", Rover.goal_distance, Rover.goal_angle)
    else:
        Rover.goal_distance, Rover.goal_angle = -1.0, 0.0
        print("CANT SEE ROCK!")

    # Calculate unexplored area
    # These masks are used to get a small unexplored area around the current view, to reduce the number of points in the graph.
    current_view = TerrainSet(persp, np.ones_like(img[:,:,0]))
    extended_view = TerrainMask(persp, persp.mask_area)

    Rover.explored[current_view.y_world, current_view.x_world] = 1
    Rover.unexplored[extended_view.y_world, extended_view.x_world] = 1
    Rover.unexplored = (Rover.unexplored > 0) & (Rover.explored == 0)

    # Polar coordinates for decision step
    dists, angles = to_polar_coords(nav_terrain.x_rover, nav_terrain.y_rover)

    # Update the Rover worldmap
    Rover.worldmap[obs_terrain.y_world, obs_terrain.x_world, 0] += 1
    Rover.worldmap[nav_terrain.y_world, nav_terrain.x_world, 2] += 10

    # Update image to be displayed o the side
    Rover.vision_image[:, :, 0] = obs_terrain.warped * 255
    Rover.vision_image[:, :, 1] = rock_terrain.warped * 255
    Rover.vision_image[:, :, 2] = nav_terrain.warped * 255
    
    # Output for decision step
    Rover.nav_angles = angles

    return Rover

