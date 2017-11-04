import numpy as np
import cv2

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


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

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
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped

def get_transform(img, dst_size):
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                              [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                              [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], 
                              [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset]])
    return source, destination

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):

    img = Rover.img
    
    dst_size = 5 

    world_size, scale = Rover.worldmap.shape[0], 2 * dst_size

    source, destination = get_transform(img, dst_size)
    
    # Thresholds
    rock_thresh_lo, rock_thresh_hi = (126, 106, 0), (255, 216, 84)
    navigable_thresh_lo = (161, 161, 161)

    # Navigable terrain 
    nav_thresh = color_thresh(img, lo=navigable_thresh_lo)
    nav_warped = perspect_transform(nav_thresh, source, destination)
    nav_x_rover, nav_y_rover = rover_coords(nav_warped)
    nav_x_world, nav_y_world = pix_to_world(nav_x_rover, nav_y_rover, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
       
    # Obstacles 
    obs_thresh = color_thresh(img, hi=navigable_thresh_lo)
    obs_warped = perspect_transform(obs_thresh, source, destination)
    obs_x_rover, obs_y_rover = rover_coords(obs_warped)
    obs_x_world, obs_y_world = pix_to_world(obs_x_rover, obs_y_rover, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)

    # Rock samples
    rock_thresh = color_thresh(img, lo=rock_thresh_lo, hi=rock_thresh_hi)
    rock_warped = perspect_transform(rock_thresh, source, destination)
    rock_x_rover, rock_y_rover = rover_coords(rock_warped)
    rock_x_world, rock_y_world = pix_to_world(rock_x_rover, rock_y_rover, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)

    if (rock_warped.any()):
        rock_dist, rock_ang = to_polar_coords(rock_x_rover, rock_y_rover)
        rock_idx = np.argmin(rock_dist)
        rock_x, rock_y = rock_x_world[rock_idx], rock_y_world[rock_idx]
        Rover.worldmap[rock_y, rock_x, 1] = 255

    dists, angles = to_polar_coords(nav_x_rover, nav_y_rover)

    # Update the Rover

    Rover.worldmap[obs_y_world, obs_x_world, 0] += 1
    Rover.worldmap[nav_y_world, nav_x_world, 2] += 10

    Rover.vision_image[:, :, 0] = obs_warped * 255
    Rover.vision_image[:, :, 1] = rock_warped * 255
    Rover.vision_image[:, :, 2] = nav_warped * 255
    
    Rover.nav_angles = angles

    return Rover

