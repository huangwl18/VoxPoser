import numpy as np
from perception_utils import detect

objects = ['green block', 'cardboard box']
# Query: gripper.
gripper = detect('gripper')
ret_val = gripper

objects = ['handle1', 'handle2', 'egg1', 'egg2', 'plate']
# Query: topmost handle.
handle1 = detect('handle1')
handle2 = detect('handle2')
if handle1.position[2] > handle2.position[2]:
    top_handle = handle1
else:
    top_handle = handle2
ret_val = top_handle

objects = ['vase', 'napkin box', 'mask']
# Query: table.
table = detect('table')
ret_val = table

objects = ['brown line', 'red block', 'monitor']
# Query: brown line.
brown_line = detect('brown line')
ret_val = brown_line

objects = ['green block', 'cup holder', 'black block']
# Query: any block.
block = detect('green block')
ret_val = block

objects = ['mouse', 'yellow bowl', 'brown bowl', 'sticker']
# Query: bowl closest to the sticker.
yellow_bowl = detect('yellow bowl')
brown_bowl = detect('brown bowl')
sticker = detect('sticker')
if np.linalg.norm(yellow_bowl.position - sticker.position) < np.linalg.norm(brown_bowl.position - sticker.position):
    closest_bowl = yellow_bowl
else:
    closest_bowl = brown_bowl
ret_val = closest_bowl

objects = ['grape', 'wood tray', 'strawberry', 'white tray', 'blue tray', 'bread']
# Query: tray that contains the bread.
wood_tray = detect('wood tray')
white_tray = detect('white tray')
bread = detect('bread')
if np.linalg.norm(wood_tray.position - bread.position) < np.linalg.norm(white_tray.position - bread.position):
    tray_with_bread = wood_tray
else:
    tray_with_bread = white_tray
ret_val = tray_with_bread

objects = ['glass', 'vase', 'plastic bottle', 'block', 'phone case']
# Query: anything fragile.
fragile_items = []
for obj in ['glass', 'vase']:
    item = detect(obj)
    fragile_items.append(item)
ret_val = fragile_items

objects = ['blue block', 'red block']
# Query: green block.
ret_val = None