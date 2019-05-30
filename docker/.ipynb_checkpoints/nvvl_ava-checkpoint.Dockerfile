FROM reg.qiniu.com/ava-public/ava-pytorch:py35-cuda90-cudnn7
LABEL maintainer="baoyixin@qiniu.com"

# nvvl dependencies
ARG FFMPEG_VERSION=3.4.2

# nvcuvid deps
RUN apt-get update --fix-missing && \
    apt-get install -y libx11-6 libxext6 yasm wget pkg-config && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple cmake cffi
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

# ffmpeg from source
RUN cd /tmp && wget -q http://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    tar xf ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    rm ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    cd ffmpeg-$FFMPEG_VERSION && \
    ./configure \
      --prefix=/usr/local \
      --enable-shared && \
    make -j8 && make install && \
    cd /tmp && rm -rf ffmpeg-$FFMPEG_VERSION

# install nvvl: git clone https://github.com/pikerbright/nvvl.git
RUN apt install -y libgl1-mesa-glx && \
    cp -r nvvl /tmp/nvvl && \
    cd /tmp && \
    cd nvvl && \
    cp examples/pytorch_superres/docker/libnvcuvid.so /usr/local/cuda/lib64/stubs && \
    cd pytorch && \
    python3 setup.py install && \

RUN wget -O /tmp/PRC-tz http://devtools.dl.atlab.ai/docker/PRC-tz && mv /tmp/PRC-tz /etc/localtime
