import os
import numpy as np
import numpy as np
import plotly.graph_objects as go
import datetime
from transforms3d.quaternions import mat2quat

def set_lmp_objects(lmps, objects):
    if isinstance(lmps, dict):
        lmps = lmps.values()
    for lmp in lmps:
        lmp._context = f'objects = {objects}'

def get_clock_time(milliseconds=False):
    curr_time = datetime.datetime.now()
    if milliseconds:
        return f'{curr_time.hour}:{curr_time.minute}:{curr_time.second}.{curr_time.microsecond // 1000}'
    else:
        return f'{curr_time.hour}:{curr_time.minute}:{curr_time.second}'

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_prompt(prompt_fname):
    # get current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    # get full path to file
    if '/' in prompt_fname:
        prompt_fname = prompt_fname.split('/')
        full_path = os.path.join(curr_dir, 'prompts', *prompt_fname)
    else:
        full_path = os.path.join(curr_dir, 'prompts', prompt_fname)
    # read file
    with open(full_path, 'r') as f:
        contents = f.read().strip()
    return contents

def normalize_vector(x, eps=1e-6):
    """normalize a vector to unit length"""
    x = np.asarray(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return np.zeros_like(x) if norm < eps else (x / norm)
    elif x.ndim == 2:
        norm = np.linalg.norm(x, axis=1)  # (N,)
        normalized = np.zeros_like(x)
        normalized[norm > eps] = x[norm > eps] / norm[norm > eps][:, None]
        return normalized

def normalize_map(map):
    """normalization voxel maps to [0, 1] without producing nan"""
    denom = map.max() - map.min()
    if denom == 0:
        return map
    return (map - map.min()) / denom

def calc_curvature(path):
    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])
    dz = np.gradient(path[:, 2])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    ddz = np.gradient(dz)
    curvature = np.sqrt((ddy * dx - ddx * dy)**2 + (ddz * dx - ddx * dz)**2 + (ddz * dy - ddy * dz)**2) / np.power(dx**2 + dy**2 + dz**2, 3/2)
    # convert any nan to 0
    curvature[np.isnan(curvature)] = 0
    return curvature

class IterableDynamicObservation:
    """acts like a list of DynamicObservation objects, initialized with a function that evaluates to a list"""
    def __init__(self, func):
        assert callable(func), 'func must be callable'
        self.func = func
        self._validate_func_output()

    def _validate_func_output(self):
        evaluated = self.func()
        assert isinstance(evaluated, list), 'func must evaluate to a list'

    def __getitem__(self, index):
        def helper():
            evaluated = self.func()
            item = evaluated[index]
            # assert isinstance(item, Observation), f'got type {type(item)} instead of Observation'
            return item
        return helper

    def __len__(self):
        return len(self.func())

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

    def __call__(self):
        static_list = self.func()
        return static_list

class DynamicObservation:
    """acts like dict observation but initialized with a function such that it uses the latest info"""
    def __init__(self, func):
        try:
            assert callable(func) and not isinstance(func, dict), 'func must be callable or cannot be a dict'
        except AssertionError as e:
            print(e)
            import pdb; pdb.set_trace()
        self.func = func
    
    def __get__(self, key):
        evaluated = self.func()
        if isinstance(evaluated[key], np.ndarray):
            return evaluated[key].copy()
        return evaluated[key]
    
    def __getattr__(self, key):
        return self.__get__(key)
    
    def __getitem__(self, key):
        return self.__get__(key)

    def __call__(self):
        static_obs = self.func()
        if not isinstance(static_obs, Observation):
            static_obs = Observation(static_obs)
        return static_obs

class Observation(dict):
    def __init__(self, obs_dict):
        super().__init__(obs_dict)
        self.obs_dict = obs_dict
    
    def __getattr__(self, key):
        return self.obs_dict[key]
    
    def __getitem__(self, key):
        return self.obs_dict[key]

    def __getstate__(self):
        return self.obs_dict
    
    def __setstate__(self, state):
        self.obs_dict = state

def pointat2quat(pointat):
    """
    calculate quaternion from pointat vector
    """
    up = np.array(pointat, dtype=np.float32)
    up = normalize_vector(up)
    rand_vec = np.array([1, 0, 0], dtype=np.float32)
    rand_vec = normalize_vector(rand_vec)
    # make sure that the random vector is close to desired direction
    if np.abs(np.dot(rand_vec, up)) > 0.99:
        rand_vec = np.array([0, 1, 0], dtype=np.float32)
        rand_vec = normalize_vector(rand_vec)
    left = np.cross(up, rand_vec)
    left = normalize_vector(left)
    forward = np.cross(left, up)
    forward = normalize_vector(forward)
    rotmat = np.eye(3).astype(np.float32)
    rotmat[:3, 0] = forward
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up
    quat_wxyz = mat2quat(rotmat)
    return quat_wxyz

def visualize_points(point_cloud, point_colors=None, show=True):
    """visualize point clouds using plotly"""
    if point_colors is None:
        point_colors = point_cloud[:, 2]
    fig = go.Figure(data=[go.Scatter3d(x=point_cloud[:, 0], y=point_cloud[:, 1], z=point_cloud[:, 2],
                                    mode='markers', marker=dict(size=3, color=point_colors, opacity=1.0))])
    if show:
        fig.show()
    else:
        # save to html
        fig.write_html('temp_pc.html')
        print(f'Point cloud saved to temp_pc.html')

def _process_llm_index(indices, array_shape):
    """
    processing function for returned voxel maps (which are to be manipulated by LLMs)
    handles non-integer indexing
    handles negative indexing with manually designed special cases
    """
    if isinstance(indices, int) or isinstance(indices, np.int64) or isinstance(indices, np.int32) or isinstance(indices, np.int16) or isinstance(indices, np.int8):
        processed = indices if indices >= 0 or indices == -1 else 0
        assert len(array_shape) == 1, "1D array expected"
        processed = min(processed, array_shape[0] - 1)
    elif isinstance(indices, float) or isinstance(indices, np.float64) or isinstance(indices, np.float32) or isinstance(indices, np.float16):
        processed = np.round(indices).astype(int) if indices >= 0 or indices == -1 else 0
        assert len(array_shape) == 1, "1D array expected"
        processed = min(processed, array_shape[0] - 1)
    elif isinstance(indices, slice):
        start, stop, step = indices.start, indices.stop, indices.step
        if start is not None:
            start = np.round(start).astype(int)
        if stop is not None:
            stop = np.round(stop).astype(int)
        if step is not None:
            step = np.round(step).astype(int)
        # only convert the case where the start is negative and the stop is positive/negative
        if (start is not None and start < 0) and (stop is not None):
            if stop >= 0:
                processed = slice(0, stop, step)
            else:
                processed = slice(0, 0, step)
        else:
            processed = slice(start, stop, step)
    elif isinstance(indices, tuple) or isinstance(indices, list):
        processed = tuple(
            _process_llm_index(idx, (array_shape[i],)) for i, idx in enumerate(indices)
        )
    elif isinstance(indices, np.ndarray):
        print("[IndexingWrapper] Warning: numpy array indexing was converted to list")
        processed = _process_llm_index(indices.tolist(), array_shape)
    else:
        print(f"[IndexingWrapper] {indices} (type: {type(indices)}) not supported")
        raise TypeError("Indexing type not supported")
    # give warning if index was negative
    if processed != indices:
        print(f"[IndexingWrapper] Warning: index was changed from {indices} to {processed}")
    # print(f"[IndexingWrapper] {idx} -> {processed}")
    return processed

class VoxelIndexingWrapper:
    """
    LLM indexing wrapper that uses _process_llm_index to process indexing
    behaves like a numpy array
    """
    def __init__(self, array):
        self.array = array

    def __getitem__(self, idx):
        return self.array[_process_llm_index(idx, tuple(self.array.shape))]
    
    def __setitem__(self, idx, value):
        self.array[_process_llm_index(idx, tuple(self.array.shape))] = value
    
    def __repr__(self) -> str:
        return self.array.__repr__()
    
    def __str__(self) -> str:
        return self.array.__str__()
    
    def __eq__(self, other):
        return self.array == other
    
    def __ne__(self, other):
        return self.array != other
    
    def __lt__(self, other):
        return self.array < other
    
    def __le__(self, other):
        return self.array <= other
    
    def __gt__(self, other):
        return self.array > other
    
    def __ge__(self, other):
        return self.array >= other
    
    def __add__(self, other):
        return self.array + other
    
    def __sub__(self, other):
        return self.array - other
    
    def __mul__(self, other):
        return self.array * other
    
    def __truediv__(self, other):
        return self.array / other
    
    def __floordiv__(self, other):
        return self.array // other
    
    def __mod__(self, other):
        return self.array % other
    
    def __divmod__(self, other):
        return self.array.__divmod__(other)
    
    def __pow__(self, other):
        return self.array ** other
    
    def __lshift__(self, other):
        return self.array << other
    
    def __rshift__(self, other):
        return self.array >> other
    
    def __and__(self, other):
        return self.array & other
    
    def __xor__(self, other):
        return self.array ^ other
    
    def __or__(self, other):
        return self.array | other
    
    def __radd__(self, other):
        return other + self.array
    
    def __rsub__(self, other):
        return other - self.array
    
    def __rmul__(self, other):
        return other * self.array
    
    def __rtruediv__(self, other):
        return other / self.array
    
    def __rfloordiv__(self, other):
        return other // self.array
    
    def __rmod__(self, other):
        return other % self.array
    
    def __rdivmod__(self, other):
        return other.__divmod__(self.array)
    
    def __rpow__(self, other):
        return other ** self.array
    
    def __rlshift__(self, other):
        return other << self.array
    
    def __rrshift__(self, other):
        return other >> self.array
    
    def __rand__(self, other):
        return other & self.array
    
    def __rxor__(self, other):
        return other ^ self.array
    
    def __ror__(self, other):
        return other | self.array
    
    def __getattribute__(self, name):
        if name == "array":
            return super().__getattribute__(name)
        elif name == "__getitem__":
            return super().__getitem__
        elif name == "__setitem__":
            return super().__setitem__
        else:
            # print(name)
            return super().array.__getattribute__(name)
    
    def __getattr__(self, name):
        return self.array.__getattribute__(name)