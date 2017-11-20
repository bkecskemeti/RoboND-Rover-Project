## Project: Search and Sample Return

---

### Notebook Analysis

The notebook for the most part is conceptually the same as in the lectures.
The only meaningful addition is the planning part, where the shortest
path is calculated for path planning (more on this later).

#### 1. Image analysis

Note, that I decided to first do the color threshold, then the perspective
transform, to take into account the field of view (points outside the field
of view of the camera should neither assumed to be navigable, nor obstacle).

The plot below shows navigable terrain, obstacles, and rocks:
* row 1 - original images
* row 2 - thresholded images
* row 3 - perpective transform to map

![Image analysis example][./output/warped_threshed_fig.jpg]

Color thresholds for rocks were identified from 6 sample images obtained
manually, see: `/calibration_images/rock{1-6}_pixels.png`

#### 2. Extending process_image(...)

The original code for process_image(..) was very dirty so I cleaned up quite
a bit:

* `def arrange_images(num_columns, *imgs)`: to separate the task of formatting the
output image from generating the images, I defined this function which can put
arbitrary images in a grid layout defined by the number of columns

* `class Perspective`: note, that essentially the same has to be done for any terrain
type (navigable, obstacle, rock). This class contains all data needed to be able to
do the perspective transform and word coordinate mapping (x, y coordinates, yaw etc).

* `class TerrainSet()`: given a `Perspective` and a thresholded image, calculates
the warped images in both rover and world coordinates.

After this, I can nicely get everything we need for any kind of terrain, eg.:
```
nav_terrain = TerrainSet(persp, color_thresh(img, lo=navigable_thresh_lo))
```

#### 3. Tracking unexplored areas

For path planning purposes, I will need to know where are the yet unexplored areas.
The idea is, that, the Rover should choose to go to areas where there is a chance that
it finds territory it has not seen yet (more on this later).

I could assume, that every point in the 200x200 worldmap is unexplored, that however would
yield 40K points most of which are useless. Therefore, I decided to use two masks (given in
Rover coordinates)
* `trust_area:` the area which counts as explored, eg. navigable, obstacle, or rock. I reduced
 the area to be considerably below the horizon, as pixels close to the horizon map to the
 worldmap inaccurately and seemed to mess up fidelity.
* `mask_area` = an area slightly bigger than the Rover's field of view. `mask_area - trust_area`
will be the area just outside the field of view which we will add as unexplored points.

This is how one frame of the final image looks:
* raw camera image
* vision image: obstacles (red), navigable terrain (blue)
* map overlaid to ground truth
* Rover worldmap: obstacles (red), unexplored (blue), navigable (green)

![Example frame][./output/test_image_writeup.jpg]

#### 4. Path planning

I use the Dijksra algorithm from the scipy package to calculate the shortest path between
two points in the Rover's worldmap. The graph is defined as the nodes given by the navigable
and unexplored points, with edge weights between adjacent pixels given by the Manhattan distance.

![Example shortest path][./output/dijkstra.png]

I know this is very rudimentary, as the Rover is moving continuously and such paths are therefore
quite unnatural (involve unnecessary sharp turns).

How exactly this will be used in autonomous driving will be explained in the next section.

### Autonomous Navigation and Mapping

#### 1. `perception_step()`

The perception step is the same as explained above and provided in the python notebook, except:
* Rover.nav_angles is calculated as the difference of all navigable angles and the obstancle angles that are *close*:

```
obs_angles_ahead = obs_angles[np.nonzero(obs_dists < 10.0)]
Rover.nav_angles = np.setdiff1d(angles, obs_angles_ahead)
```

* When rocks are identified in the camera image, their position (angle and distance) is calculated and a goal
is set for the Rover to approach.

#### 2. `decision_step()`

The decision step comes with a number of new modes compared to the bare-bones version provided.

1. *mission_complete* mode: all samples are collected and the Rover is back to closer than 3 metres
from its starting position. Stop and do nothing in this mode.
2. *forward* mode: wandering towards the mean navigable angle. After some time has passed in this state,
the Rover will change to path mode.
3. *path* mode: planning and following a path. Planning means selecting a target coordinate in the worldmap.
If there is a sample that is discovered but not picked up, we target that location. If we collected all samples,
the target is the middle of the map. Otherwise, the target is the closest unexplored pixel that is reachable.
If there is no reachable unexplored pixel either, then the planning was unsuccessful and the Rover returns
to forward mode. If the planning was successful, the result is a path (list of worldmap pixels) to follow.
When following the path, the reached pixels are discarded and the Rover is steered towards the next unreached
pixel of the path. When all pixels are reached, the Rover returns to forward mode.
4. *approach* mode: 
5. *perturb* mode:
6. *stop* mode:

#### 3. Experimental Results 

#### 4. Known problems and ideas for improvement 

Known issues:
* Shadows: the mountains cast shadows on the ground and sometimes they are not recognized as navigable terrain
* Small obstacles: the Rover sometimes drives into obstacles if they are small in `forward` because the mean
angle will lie just in the obstacle
* False rocks: there are some strange mountain walls with dotted texture, and some pixels from some angles
can rarely identify as false positive for rock samples
* Path planning issues: sometimes the planning algorithm produces strange results eg. targeting an area
 of just one single unexplored pixel, effectively driving itself to the wall
* Issues approaching samples: sometimes if a sample is too close to a wall, I do not approach them from an
optimal angle, the Rover might struggle to get close enough to pick it up
* Paths too close to walls: the planning algorithm often results in paths too close to walls (because they are the
shortest) and depending on the terrain this can hinder the movement of the Rover
* Disappearing planes: this is more an issue with the simulator, sometimes driving into an obstacle means the camera
will pass through the surface of the rock and sees the area behind it

Possible improvements:
* Improved path planning algorithm to take into account the velocity and direction of the Rover to smoothen the path
* Do not plan paths close to obstacles
* Come up with a way to navigate through an area of smaller, closer obstacles
* Approach samples more optimally


