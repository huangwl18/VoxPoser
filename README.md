## VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models

#### [[Project Page]](https://voxposer.github.io/) [[Paper]](https://voxposer.github.io/voxposer.pdf) [[Video]](https://www.youtube.com/watch?v=Yvn4eR05A3M)

[Wenlong Huang](https://wenlong.page)<sup>1</sup>, [Chen Wang](https://www.chenwangjeremy.net/)<sup>1</sup>, [Ruohan Zhang](https://ai.stanford.edu/~zharu/)<sup>1</sup>, [Yunzhu Li](https://yunzhuli.github.io/)<sup>1,2</sup>, [Jiajun Wu](https://jiajunwu.com/)<sup>1</sup>, [Li Fei-Fei](https://profiles.stanford.edu/fei-fei-li)<sup>1</sup>

<sup>1</sup>Stanford University, <sup>2</sup>University of Illinois Urbana-Champaign

<img  src="media/teaser.gif" width="550">

This is the official demo code for [VoxPoser](https://voxposer.github.io/), a method that uses large language models and vision-language models to zero-shot synthesize trajectories for manipulation tasks.

In this repo, we provide the implementation of VoxPoser in the [RLBench](https://sites.google.com/view/rlbench) environment for the diversity of tasks it offers. Note that VoxPoser is a zero-shot method that does not require any training data. Therefore, the main purpose of this repo is to provide a demo implementation rather than an evaluation benchmark.

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