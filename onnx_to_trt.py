#!/usr/bin/python

import tensorrt as trt
import tensorflow as tf

TRT_LOGGER=trt.Logger(trt.Logger.WARNING)
with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network,TRT_LOGGER) as parser:
    builder.max_batch_size=2
    builder.max_workspace_size=3*1000*1000*1000
    builder.fp16_mode=True
    f=open('/workspace/mbv3_c4.onnx',mode='rb')

    parser.parse(f.read())
    network.get_input(0).shape=(1,256,512,3)
    print('\n\n\n\n\n\n')
    print(network)

    print('0000')
    f.close()
    engine=builder.build_cuda_engine(network)
    print('1111')
    with open('/workspace/mbv3_c4.engine',mode='wb') as plans:
        print('2222')
        plans.write(engine.serialize())
        print('3333')

