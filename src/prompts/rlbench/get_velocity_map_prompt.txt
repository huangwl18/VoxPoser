import numpy as np
from plan_utils import get_empty_velocity_map, set_voxel_by_radius, cm2index
from perception_utils import parse_query_obj

# Query: faster when on the right side of the table and slower when on the left side of the table.
velocity_map = get_empty_velocity_map()
table = parse_query_obj('table')
center_x, center_y, center_z = table.position
# faster on right side so 1.5 when y > center_y, slower on left side so 0.5 when y < center_y
velocity_map[:, center_y:, :] = 1.5
velocity_map[:, :center_y, :] = 0.5
ret_val = velocity_map

# Query: slow down by a quarter.
velocity_map = get_empty_velocity_map()
velocity_map[:] = 0.75
ret_val = velocity_map

# Query: slow down by a half when you're near anything fragile (objects: ['block', 'fork', 'mug', 'bowl', 'chips']).
velocity_map = get_empty_velocity_map()
mug = parse_query_obj('mug')
set_voxel_by_radius(velocity_map, mug.position, radius_cm=10, value=0.5)
bowl = parse_query_obj('bowl')
set_voxel_by_radius(velocity_map, bowl.position, radius_cm=10, value=0.5)
ret_val = velocity_map

# Query: quarter of the speed when within 9cm from the yellow line.
velocity_map = get_empty_velocity_map()
yellow_line = parse_query_obj('yellow_line')
set_voxel_by_radius(velocity_map, yellow_line.position, radius_cm=9, value=0.25)
ret_val = velocity_map