FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu16.04

LABEL maintainer "martin@martin-benson.com"

# Install dependencies
RUN apt-get update && \
    apt-get install --assume-yes git libprotobuf-dev libopenblas-dev liblapack-dev protobuf-compiler wget python3-pip sudo
    
RUN git clone https://github.com/nagadomi/distro.git ~/torch --recursive && \
    cd ~/torch && \
    bash install-deps && \
    ./install.sh    

WORKDIR /root/torch

SHELL ["/bin/bash", "-c"]

# Export environment variables manually
ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-beta1/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua'
ENV LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so'
ENV PATH=/root/torch/install/bin:$PATH
ENV LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH
ENV DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH
ENV LUA_CPATH='/root/torch/install/lib/?.so;'$LUA_CPATH

RUN source /root/.bashrc && \ 
    luarocks install csvigo && \
    luarocks install loadcaffe && \
    git clone https://github.com/xuelongmu/deep-photo-styletransfer.git  ~/deep_photo && \
    cd /root/deep_photo && \
    sh models/download_models.sh

WORKDIR /root/deep_photo
RUN make clean && make && pip3 install numpy scipy==1.1.0 Pillow
