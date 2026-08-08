FROM pytorch:r39.2.tegra-aarch64-cu132-24.04-pytorch AS trt

FROM opencv:r39.2.tegra-aarch64-cu132-24.04

RUN apt-get update && apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# TensorRT shared libs (all nvinfer/onnxparser variants)
COPY --from=trt /usr/local/cuda-13.2/targets/aarch64-linux/lib/libnvinfer*.so* \
                /usr/local/cuda-13.2/targets/aarch64-linux/lib/
COPY --from=trt /usr/local/cuda-13.2/targets/aarch64-linux/lib/libnvonnxparser*.so* \
                /usr/local/cuda-13.2/targets/aarch64-linux/lib/

# trtexec + python wheel
COPY --from=trt /usr/bin/trtexec /usr/bin/trtexec
COPY --from=trt /usr/local/lib/python3.12/dist-packages/tensorrt-10.16.1.11-cp312-none-linux_aarch64.whl /tmp/

RUN pip install --no-cache-dir /tmp/tensorrt-10.16.1.11-cp312-none-linux_aarch64.whl

# rtmlib stack, no opencv override
RUN pip install --no-cache-dir --index-url https://pypi.org/simple \
    onnxruntime numpy tqdm pycuda
RUN pip install --no-cache-dir --index-url https://pypi.org/simple \
    --no-deps rtmlib
RUN pip install --no-cache-dir --index-url https://pypi.org/simple \
    onnxruntime numpy tqdm pycuda aiohttp
    
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.2/targets/aarch64-linux/lib:${LD_LIBRARY_PATH}

WORKDIR /workspace
