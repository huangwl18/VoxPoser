import numpy as np
from plan_utils import get_empty_rotation_map, set_voxel_by_radius, cm2index, vec2quat
from perception_utils import parse_query_obj
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import qmult, qinverse

# Query: face the support surface of the bowl.
rotation_map = get_empty_rotation_map()
bowl = parse_query_obj('bowl')
target_rotation = vec2quat(-bowl.normal)
rotation_map[:, :, :] = target_rotation
ret_val = rotation_map

# Query: face the table when within 30cm from table center.
rotation_map = get_empty_rotation_map()
table = parse_query_obj('table')
table_center = table.position
target_rotation = vec2quat(-table.normal)
set_voxel_by_radius(rotation_map, table_center, radius_cm=30, value=target_rotation)
ret_val = rotation_map

# Query: face the blue bowl.
rotation_map = get_empty_rotation_map()
blue_bowl = parse_query_obj('brown block')
target_rotation = vec2quat(-blue_bowl.normal)
rotation_map[:, :, :] = target_rotation
ret_val = rotation_map

# Query: turn clockwise by 45 degrees when at the center of the beer cap.
rotation_map = get_empty_rotation_map()
beer_cap = parse_query_obj('beer cap')
(x, y, z) = beer_cap.position
curr_rotation = rotation_map[x, y, z]
rotation_delta = euler2quat(0, 0, np.pi / 4)
rotation_map[x, y, z] = qmult(curr_rotation, rotation_delta)
ret_val = rotation_map

# Query: turn counter-clockwise by 30 degrees.
rotation_map = get_empty_rotation_map()
curr_rotation = rotation_map[0, 0, 0]
rotation_delta = euler2quat(0, 0, -np.pi / 6)
rotation_map[:, :, :] = qmult(curr_rotation, rotation_delta)
ret_val = rotation_map

# Query: rotate the gripper to be 45 degrees slanted relative to the plate.
rotation_map = get_empty_rotation_map()
plate = parse_query_obj('plate')
face_plate_quat = vec2quat(-plate.normal)
# rotate 45 degrees around the x-axis
rotation_delta = euler2quat(-np.pi / 4, 0, 0)
target_rotation = qmult(face_plate_quat, rotation_delta)
rotation_map[:, :, :] = target_rotation
ret_val = rotation_map