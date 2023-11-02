import numpy as np

class PushingDynamicsModel:
    """
    Heuristics-based pushing dynamics model.
    Translates the object by gripper_moving_distance in gripper_direction.
    """
    def __init__(self):
        pass

    def forward(self, inputs, max_per_batch=2000):
        """split inputs into multiple batches if exceeds max_per_batch"""
        num_batch = int(np.ceil(inputs[0].shape[0] / max_per_batch))
        output = []
        for i in range(num_batch):
            start = i * max_per_batch
            end = (i + 1) * max_per_batch
            output.append(self._forward_batched([x[start:end] for x in inputs]))
        output = np.concatenate(output, axis=0)
        return output

    def _forward_batched(self, inputs):
        (pcs, contact_position, gripper_direction, gripper_moving_distance) = inputs
        # to float16
        pcs = pcs.astype(np.float16)
        contact_position = contact_position.astype(np.float16)
        gripper_direction = gripper_direction.astype(np.float16)
        gripper_moving_distance = gripper_moving_distance.astype(np.float16)
        # find invalid push (i.e., outward push)
        obj_center = np.mean(pcs, axis=1)  # B x 3
        is_outward = np.sum((obj_center - contact_position) * gripper_direction, axis=1) < 0  # B
        moving_dist = gripper_moving_distance.copy()
        moving_dist[is_outward] = 0
        # translate pc by gripper_moving_distance in gripper_direction
        output = pcs + moving_dist[:, np.newaxis, :] * gripper_direction[:, np.newaxis, :]  # B x N x 3
        return output
