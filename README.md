## VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models

#### [[Project Page]](https://voxposer.github.io/) [[Paper]](https://voxposer.github.io/voxposer.pdf) [[Video]](https://www.youtube.com/watch?v=Yvn4eR05A3M)

[Wenlong Huang](https://wenlong.page)<sup>1</sup>, [Chen Wang](https://www.chenwangjeremy.net/)<sup>1</sup>, [Ruohan Zhang](https://ai.stanford.edu/~zharu/)<sup>1</sup>, [Yunzhu Li](https://yunzhuli.github.io/)<sup>1,2</sup>, [Jiajun Wu](https://jiajunwu.com/)<sup>1</sup>, [Li Fei-Fei](https://profiles.stanford.edu/fei-fei-li)<sup>1</sup>

<sup>1</sup>Stanford University, <sup>2</sup>University of Illinois Urbana-Champaign

<img  src="media/teaser.gif" width="550">

This is the official demo code for [VoxPoser](https://voxposer.github.io/), a method that uses large language models and vision-language models to zero-shot synthesize trajectories for manipulation tasks.

In this repo, we provide the implementation of VoxPoser in [RLBench](https://sites.google.com/view/rlbench) as its task diversity best resembles our real-world setup. Note that VoxPoser is a zero-shot method that does not require any training data. Therefore, the main purpose of this repo is to provide a demo implementation rather than an evaluation benchmark.

**Note: This codebase currently does not contain the perception pipeline used in our real-world experiments, which produces a real-time mapping from object names to object masks. Instead, it uses the object masks provided as part of RLBench's `get_observation` function. If you are interested in deploying the code on a real robot, you may find more information in the section [Real World Deployment](#real-world-deployment).**

If you find this work useful in your research, please cite using the following BibTeX:

```bibtex
@article{huang2023voxposer,
      title={VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models},
      author={Huang, Wenlong and Wang, Chen and Zhang, Ruohan and Li, Yunzhu and Wu, Jiajun and Fei-Fei, Li},
      journal={arXiv preprint arXiv:2307.05973},
      year={2023}
    }
```

## Setup Instructions

Note that this codebase is best run with a display. For running in headless mode, refer to the [instructions in RLBench](https://github.com/stepjam/RLBench#running-headless).

- Create a conda environment:
```Shell
conda create -n voxposer-env python=3.9
conda activate voxposer-env
```

- See [Instructions](https://github.com/stepjam/RLBench#install) to install PyRep and RLBench (Note: install these inside the created conda environment).

- Install other dependencies:
```Shell
pip install -r requirements.txt
```

- Obtain an [OpenAI API](https://openai.com/blog/openai-api) key, and put it inside the first cell of the demo notebook.

## Running Demo

Demo code is at `src/playground.ipynb`. Instructions can be found in the notebook.

## Code Structure

Core to VoxPoser:

- **`playground.ipynb`**: Playground for VoxPoser.
- **`LMP.py`**: Implementation of Language Model Programs (LMPs) that recursively generates code to decompose instructions and compose value maps for each sub-task.
- **`interfaces.py`**: Interface that provides necessary APIs for language models (i.e., LMPs) to operate in voxel space and to invoke motion planner.
- **`planners.py`**: Implementation of a greedy planner that plans a trajectory (represented as a series of waypoints) for an entity/movable given a value map.
- **`controllers.py`**: Given a waypoint for an entity/movable, the controller applies (a series of) robot actions to achieve the waypoint.
- **`dynamics_models.py`**: Environment dynamics model for the case where entity/movable is an object or object part. This is used in `controllers.py` to perform MPC.
- **`prompts/rlbench`**: Prompts used by the different Language Model Programs (LMPs) in VoxPoser.

Environment and utilities:

- **`envs`**:
  - **`rlbench_env.py`**: Wrapper of RLBench env to expose useful functions for VoxPoser.
  - **`task_object_names.json`**: Mapping of object names exposed to VoxPoser and their corresponding scene object names for each individual task.
- **`configs/rlbench_config.yaml`**: Config file for all the involved modules in RLBench environment.
- **`arguments.py`**: Argument parser for the config file.
- **`LLM_cache.py`**: Caching of language model outputs that writes to disk to save cost and time.
- **`utils.py`**: Utility functions.
- **`visualizers.py`**: A Plotly-based visualizer for value maps and planned trajectories.

## Real-World Deployment
To adapt the code to deploy on a real robot, most changes should only happen in the environment file (e.g., you can consider making a copy of `rlbench_env.py` and implementing the same APIs based on your perception and controller modules).

Our perception pipeline consists of the following modules: [OWL-ViT](https://huggingface.co/docs/transformers/en/model_doc/owlvit) for open-vocabulary detection in the first frame, [SAM](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#segment-anything) for converting the produced bounding boxes to masks in the first frame, and [XMEM](https://github.com/hkchengrex/XMem) for tracking the masks over time for the subsequent frames. Now you may consider simplifying the pipeline using only an open-vocabulary detector and [SAM 2](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#latest-updates----sam-2-segment-anything-in-images-and-videos) for segmentation and tracking. Our controller is based on the OSC implementation from [Deoxys](https://github.com/UT-Austin-RPL/deoxys_control). More details can be found in the [paper](https://voxposer.github.io/voxposer.pdf).

To avoid compounded latency introduced by different modules (especially the perception pipeline), you may also consider running a concurrent process that only performs tracking.

## Acknowledgments
- Environment is based on [RLBench](https://sites.google.com/view/rlbench).
- Implementation of Language Model Programs (LMPs) is based on [Code as Policies](https://code-as-policies.github.io/).
- Some code snippets are from [Where2Act](https://cs.stanford.edu/~kaichun/where2act/).
