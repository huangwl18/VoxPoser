### VoxPoser Docker Image

### Docker Build

```
docker build . -f Dockerfile -t voxposer
```
### Docker Run
```
docker run -it --privileged --gpus all --net=host -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY voxposer
```

### Debug
```
docker run -it --privileged --gpus all --net=host --entrypoint="" -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY voxposer /bin/bash
```

CoppeliaSim
```
cd CoppeliaSim
./coppeliaSim
```

VoxPoser
```
cd VoxPoser/
jupyter notebook --allow-root
```
