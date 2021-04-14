# tensorflow_to_onnx_to_tensorrt

## NOTICE -- 注意
不同版本，不同平台，不同系统的tensorrt生成的engine是不能通用的

## env -- 环境
host：用于训练、存储tensorflow的frozen graph，X86_64
device: 用于部署推理

`pip install tensorflow-gpu==1.15.5 nvidia-pyindex tf2onnx pycuda`

## 将frozen_graph转成onnx -- convert frozen_graph to onxx
使用tf2onnx进行转换，命令如下

`python/python3 -m tf2onnx.convert --graphdef FROZEN_GRAPH_PATH --output OUTPUT_ONNX_MODEL_FILE_PATH --inputs THE_NAME_OF_INPUT_TENSOR_NAME --outputs THE_NAME_OF_OUTPUT_TNEOSR_NAME`

`ps:THE_NAME_OF_INPUT_TENSOR_NAME是生成frozen_graph时使用的tensor name不是node name，如 input:0。
THE_NAME_OF_OUTPUT_TENSOR_NAME类似`


其他格式请参考https://github.com/onnx/tensorflow-onnx

## 生成engine并将其序列化保存 -- create cuda engine and save by serialization

需要自己修改脚本中的对应路径

`python3 onnx_to_trt.py`

## 推理 -- do inference

自已修改脚本对应路径

`python3 inference_trt.py`

其他参考：
https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/
https://developer.nvidia.com/zh-cn/blog/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/
