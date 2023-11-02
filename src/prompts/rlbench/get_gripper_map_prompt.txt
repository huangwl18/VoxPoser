import numpy as np
from perception_utils import parse_query_obj
from plan_utils import get_empty_gripper_map, set_voxel_by_radius, cm2index

# Query: open everywhere except 1cm around the green block.
gripper_map = get_empty_gripper_map()
# open everywhere
gripper_map[:, :, :] = 1
# close when 1cm around the green block
green_block = parse_query_obj('green block')
set_voxel_by_radius(gripper_map, green_block.position, radius_cm=1, value=0)
ret_val = gripper_map

# Query: close everywhere but open when on top of the back left corner of the table.
gripper_map = get_empty_gripper_map()
# close everywhere
gripper_map[:, :, :] = 0
# open when on top of the back left corner of the table
table = parse_query_obj('table')
(min_x, min_y, min_z), (max_x, max_y, max_z) = table.aabb
center_x, center_y, center_z = table.position
# back so x = min_x, left so y = min_y, top so we add to z
x = min_x
y = min_y
z = max_z + cm2index(15, 'z')
set_voxel_by_radius(gripper_map, (x, y, z), radius_cm=10, value=1)
ret_val = gripper_map

# Query: always open except when you are on the right side of the table.
gripper_map = get_empty_gripper_map()
# always open
gripper_map[:, :, :] = 1
# close when you are on the right side of the table
table = parse_query_obj('table')
center_x, center_y, center_z = table.position
# right side so y is greater than center_y
gripper_map[:, center_y:, :] = 0

# Query: always close except when you are on the back side of the table.
gripper_map = get_empty_gripper_map()
# always close
gripper_map[:, :, :] = 0
# open when you are on the back side of the table
table = parse_query_obj('table')
center_x, center_y, center_z = table.position
# back side so x is less than center_x
gripper_map[:center_x, :, :] = 1
ret_val = gripper_map