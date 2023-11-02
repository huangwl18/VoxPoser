import numpy as np
from env_utils import execute, reset_to_default_pose
from perception_utils import parse_query_obj
from plan_utils import get_affordance_map, get_avoidance_map, get_velocity_map, get_rotation_map, get_gripper_map

# Query: move ee forward for 10cm.
movable = parse_query_obj('gripper')
affordance_map = get_affordance_map(f'a point 10cm in front of {movable.position}')
execute(movable, affordance_map)

# Query: go back to default.
reset_to_default_pose()

# Query: move the gripper behind the bowl, and slow down when near the bowl.
movable = parse_query_obj('gripper')
affordance_map = get_affordance_map('a point 15cm behind the bowl')
avoidance_map = get_avoidance_map('10cm near the bowl')
velocity_map = get_velocity_map('slow down when near the bowl')
execute(movable, affordance_map=affordance_map, avoidance_map=avoidance_map, velocity_map=velocity_map)

# Query: move to the back side of the table while staying at least 5cm from the blue block.
movable = parse_query_obj('gripper')
affordance_map = get_affordance_map('a point on the back side of the table')
avoidance_map = get_avoidance_map('5cm from the blue block')
execute(movable, affordance_map=affordance_map, avoidance_map=avoidance_map)

# Query: move to the top of the plate and face the plate.
movable = parse_query_obj('gripper')
affordance_map = get_affordance_map('a point 10cm above the plate')
rotation_map = get_rotation_map('face the plate')
execute(movable, affordance_map=affordance_map, rotation_map=rotation_map)

# Query: drop the toy inside container.
movable = parse_query_obj('gripper')
affordance_map = get_affordance_map('a point 15cm above the container')
gripper_map = get_gripper_map('close everywhere but open when on top of the container')
execute(movable, affordance_map=affordance_map, gripper_map=gripper_map)

# Query: push close the topmost drawer.
movable = parse_query_obj('topmost drawer handle')
affordance_map = get_affordance_map('a point 30cm into the topmost drawer handle')
execute(movable, affordance_map=affordance_map)

# Query: push the second to the left block along the red line.
movable = parse_query_obj('second to the left block')
affordance_map = get_affordance_map('the red line')
execute(movable, affordance_map=affordance_map)

# Query: grasp the blue block from the table at a quarter of the speed.
movable = parse_query_obj('gripper')
affordance_map = get_affordance_map('a point at the center of blue block')
velocity_map = get_velocity_map('quarter of the speed')
gripper_map = get_gripper_map('open everywhere except 1cm around the blue block')
execute(movable, affordance_map=affordance_map, velocity_map=velocity_map, gripper_map=gripper_map)

# Query: move to the left of the brown block.
movable = parse_query_obj('gripper')
affordance_map = get_affordance_map('a point 10cm to the left of the brown block')
execute(movable, affordance_map=affordance_map)

# Query: move to the top of the tray that contains the lemon.
movable = parse_query_obj('gripper')
affordance_map = get_affordance_map('a point 10cm above the tray that contains the lemon')
execute(movable, affordance_map=affordance_map)

# Query: close drawer by 5cm.
movable = parse_query_obj('drawer handle')
affordance_map = get_affordance_map('a point 5cm into the drawer handle')
execute(movable, affordance_map=affordance_map)

# Query: move to 5cm on top of the soda can, at 0.5x speed when within 20cm of the wooden mug, and keep at least 15cm away from the wooden mug.
movable = parse_query_obj('gripper')
affordance_map = get_affordance_map('a point 5cm above the soda can')
avoidance_map = get_avoidance_map('15cm from the wooden mug')
velocity_map = get_velocity_map('0.5x speed when within 20cm of the wooden mug')
execute(movable, affordance_map=affordance_map, avoidance_map=avoidance_map, velocity_map=velocity_map)

# Query: wipe the red dot but avoid the blue block.
movable = parse_query_obj('gripper')
affordance_map = get_affordance_map('the red dot')
avoidance_map = get_avoidance_map('10cm from the blue block')
execute(movable, affordance_map=affordance_map, avoidance_map=avoidance_map)

# Query: grasp the mug from the shelf.
movable = parse_query_obj('gripper')
affordance_map = get_affordance_map('a point at the center of the mug handle')
gripper_map = get_gripper_map('open everywhere except 1cm around the mug handle')
execute(movable, affordance_map=affordance_map, gripper_map=gripper_map)

# Query: move to 10cm on top of the soup bowl, and 5cm to the left of the soup bowl, while away from the glass, at 0.75x speed.
movable = parse_query_obj('gripper')
affordance_map = get_affordance_map('a point 10cm above and 5cm to the left of the soup bowl')
avoidance_map = get_avoidance_map('10cm from the glass')
velocity_map = get_velocity_map('0.75x speed')
execute(movable, affordance_map=affordance_map, avoidance_map=avoidance_map, velocity_map=velocity_map)

# Query: open gripper.
movable = parse_query_obj('gripper')
gripper_map = get_gripper_map('open everywhere')
execute(movable, gripper_map=gripper_map)

# Query: turn counter-clockwise by 180 degrees.
movable = parse_query_obj('gripper')
rotation_map = get_rotation_map('turn counter-clockwise by 180 degrees')
execute(movable, rotation_map=rotation_map)

# Query: sweep all particles to the left side of the table.
particles = parse_query_obj('particles')
for particle in particles:
    movable = particle
    affordance_map = get_affordance_map('a point on the left side of the table')
    execute(particle, affordance_map=affordance_map)

# Query: grasp the bottom drawer handle while moving at 0.5x speed.
movable = parse_query_obj('gripper')
affordance_map = get_affordance_map('a point at the center of the bottom drawer handle')
velocity_map = get_velocity_map('0.5x speed')
rotation_map = get_rotation_map('face the bottom drawer handle')
gripper_map = get_gripper_map('open everywhere except 1cm around the bottom drawer handle')
execute(movable, affordance_map=affordance_map, velocity_map=velocity_map, rotation_map=rotation_map, gripper_map=gripper_map)