import numpy as np
from perception_utils import parse_query_obj
from plan_utils import get_empty_affordance_map, set_voxel_by_radius, cm2index

# Query: a point 10cm in front of [10, 15, 60].
affordance_map = get_empty_affordance_map()
# 10cm in front of so we add to x-axis
x = 10 + cm2index(10, 'x')
y = 15
z = 60
affordance_map[x, y, z] = 1
ret_val = affordance_map

# Query: a point on the right side of the table.
affordance_map = get_empty_affordance_map()
table = parse_query_obj('table')
(min_x, min_y, min_z), (max_x, max_y, max_z) = table.aabb
center_x, center_y, center_z = table.position
# right side so y = max_y
x = center_x
y = max_y
z = center_z
affordance_map[x, y, z] = 1
ret_val = affordance_map

# Query: a point 20cm on top of the container.
affordance_map = get_empty_affordance_map()
container = parse_query_obj('container')
(min_x, min_y, min_z), (max_x, max_y, max_z) = container.aabb
center_x, center_y, center_z = container.position
# 20cm on top of so we add to z-axis
x = center_x
y = center_y
z = max_z + cm2index(20, 'z')
affordance_map[x, y, z] = 1
ret_val = affordance_map

# Query: a point 1cm to the left of the brown block.
affordance_map = get_empty_affordance_map()
brown_block = parse_query_obj('brown block')
(min_x, min_y, min_z), (max_x, max_y, max_z) = brown_block.aabb
center_x, center_y, center_z = brown_block.position
# 1cm to the left of so we subtract from y-axis
x = center_x
y = min_y - cm2index(1, 'y')
z = center_z
affordance_map[x, y, z] = 1
ret_val = affordance_map

# Query: anywhere within 20cm of the right most block.
affordance_map = get_empty_affordance_map()
right_most_block = parse_query_obj('the right most block')
set_voxel_by_radius(affordance_map, right_most_block.position, radius_cm=20, value=1)

# Query: a point on the back side of the table.
affordance_map = get_empty_affordance_map()
table = parse_query_obj('table')
(min_x, min_y, min_z), (max_x, max_y, max_z) = table.aabb
center_x, center_y, center_z = table.position
# back side so x = min_x
x = min_x
y = center_y
z = center_z
affordance_map[x, y, z] = 1
ret_val = affordance_map

# Query: a point on the front right corner of the table.
affordance_map = get_empty_affordance_map()
table = parse_query_obj('table')
(min_x, min_y, min_z), (max_x, max_y, max_z) = table.aabb
center_x, center_y, center_z = table.position
# front right corner so x = max_x and y = max_y
x = max_x
y = max_y
z = center_z
affordance_map[x, y, z] = 1
ret_val = affordance_map

# Query: a point 30cm into the topmost drawer handle.
affordance_map = get_empty_affordance_map()
top_handle = parse_query_obj('topmost drawer handle')
# negative normal because we are moving into the handle.
moving_dir = -top_handle.normal
affordance_xyz = top_handle.position + cm2index(30, moving_dir)
affordance_map[affordance_xyz[0], affordance_xyz[1], affordance_xyz[2]] = 1
ret_val = affordance_map

# Query: a point 5cm above the blue block.
affordance_map = get_empty_affordance_map()
blue_block = parse_query_obj('blue block')
(min_x, min_y, min_z), (max_x, max_y, max_z) = blue_block.aabb
center_x, center_y, center_z = blue_block.position
# 5cm above so we add to z-axis
x = center_x
y = center_y
z = max_z + cm2index(5, 'z')
affordance_map[x, y, z] = 1
ret_val = affordance_map

# Query: a point 20cm away from the leftmost block.
affordance_map = get_empty_affordance_map()
leftmost_block = parse_query_obj('leftmost block')
# positive normal because we are moving away from the block.
moving_dir = leftmost_block.normal
affordance_xyz = leftmost_block.position + cm2index(20, moving_dir)
affordance_map[affordance_xyz[0], affordance_xyz[1], affordance_xyz[2]] = 1
ret_val = affordance_map

# Query: a point 4cm to the left of and 10cm on top of the tray that contains the lemon.
affordance_map = get_empty_affordance_map()
tray_with_lemon = parse_query_obj('tray that contains the lemon')
(min_x, min_y, min_z), (max_x, max_y, max_z) = tray_with_lemon.aabb
center_x, center_y, center_z = tray_with_lemon.position
# 4cm to the left of so we subtract from y-axis, and 10cm on top of so we add to z-axis
x = center_x
y = min_y - cm2index(4, 'y')
z = max_z + cm2index(10, 'z')
affordance_map[x, y, z] = 1
ret_val = affordance_map

# Query: a point 10cm to the right of [45 49 66], and 5cm above it.
affordance_map = get_empty_affordance_map()
# 10cm to the right of so we add to y-axis, and 5cm above it so we add to z-axis
x = 45
y = 49 + cm2index(10, 'y')
z = 66 + cm2index(5, 'z')
affordance_map[x, y, z] = 1
ret_val = affordance_map

# Query: the blue circle.
affordance_map = get_empty_affordance_map()
blue_circle = parse_query_obj('blue circle')
affordance_map = blue_circle.occupancy_map
ret_val = affordance_map

# Query: a point at the center of the blue circle.
affordance_map = get_empty_affordance_map()
blue_circle = parse_query_obj('blue circle')
x, y, z = blue_block.position
affordance_map[x, y, z] = 1
ret_val = affordance_map

# Query: a point 10cm above and 5cm to the left of the yellow bowl.
affordance_map = get_empty_affordance_map()
yellow_bowl = parse_query_obj('yellow bowl')
(min_x, min_y, min_z), (max_x, max_y, max_z) = yellow_bowl.aabb
center_x, center_y, center_z = yellow_bowl.position
# 10cm above so we add to z-axis, and 5cm to the left of so we subtract from y-axis
x = center_x
y = min_y - cm2index(5, 'y')
z = max_z + cm2index(10, 'z')
affordance_map[x, y, z] = 1
ret_val = affordance_mapS