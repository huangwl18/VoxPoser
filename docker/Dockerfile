FROM brunneis/python:3.9.0-ubuntu-20.04

RUN apt-get update -y
RUN apt-get install -y git wget gcc xz-utils libgomp1 libgl1-mesa-glx libffi-dev python3-pip
RUN python3.9 -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace

# CoppeliaSim
ENV COPPELIASIM_ROOT=/workspace/CoppeliaSim \
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/CoppeliaSim \
  QT_QPA_PLATFORM_PLUGIN_PATH=/workspace/CoppeliaSim
RUN wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz && \
  mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1 && \
  rm -rf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
RUN apt-get install -y libglib2.0-0 libxkbcommon-x11-0 libfontconfig libxrender1 libdbus-1-dev libavcodec-dev libavformat-dev libswscale-dev

# RLBench
RUN git clone --depth=1 https://github.com/stepjam/RLBench.git
RUN pip3 install -e RLBench

# VoxPoser
RUN git clone --depth=1 https://github.com/huangwl18/VoxPoser.git
RUN cd VoxPoser && pip3 install -r requirements.txt
## hack openai version to 0.28
RUN pip uninstall -y openai && pip install openai==0.28

ENTRYPOINT ["/bin/sh", "-c" , "cd VoxPoser && jupyter notebook --allow-root"]
