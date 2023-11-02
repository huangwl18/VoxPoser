import numpy as np
from perception_utils import parse_query_obj
from plan_utils import get_empty_avoidance_map, set_voxel_by_radius, cm2index

# Query: 10cm from the bowl.
avoidance_map = get_empty_avoidance_map()
bowl = parse_query_obj('bowl')
set_voxel_by_radius(avoidance_map, bowl.position, radius_cm=10, value=1)
ret_val = avoidance_map

# Query: 20cm near the mug.
avoidance_map = get_empty_avoidance_map()
mug = parse_query_obj('mug')
set_voxel_by_radius(avoidance_map, mug.position, radius_cm=20, value=1)
ret_val = avoidance_map

# Query: 20cm around the mug and 10cm around the bowl.
avoidance_map = get_empty_avoidance_map()
mug = parse_query_obj('mug')
set_voxel_by_radius(avoidance_map, mug.position, radius_cm=20, value=1)
bowl = parse_query_obj('bowl')
set_voxel_by_radius(avoidance_map, bowl.position, radius_cm=10, value=1)
ret_val = avoidance_map

# Query: 10cm from anything fragile.
avoidance_map = get_empty_avoidance_map()
fragile_objects = parse_query_obj('anything fragile')
for obj in fragile_objects:
    set_voxel_by_radius(avoidance_map, obj.position, radius_cm=10, value=1)
ret_val = avoidance_map

# Query: 10cm from the blue circle.
avoidance_map = get_empty_avoidance_map()
blue_circle = parse_query_obj('blue circle')
set_voxel_by_radius(avoidance_map, blue_circle.position, radius_cm=10, value=1)
ret_val = avoidance_map