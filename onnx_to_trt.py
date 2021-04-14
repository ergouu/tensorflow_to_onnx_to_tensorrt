#!/usr/bin/python

import tensorrt as trt
import tensorflow as tf

TRT_LOGGER=trt.Logger(trt.Logger.WARNING)
with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network,TRT_LOGGER) as parser:
    builder.max_batch_size=2
    builder.max_workspace_size=3*1000*1000*1000
    
    # set model precision, fp16/fp32/int8
    builder.fp16_mode=True
    
    # load onnx model from file
    f=open('/workspace/mbv3_c4.onnx',mode='rb')

    parser.parse(f.read())
    
    #set the input shape
    network.get_input(0).shape=(1,256,512,3)
    f.close()
    print('\n\n\n\n\n\n')
    
    #build cuda engine
    engine=builder.build_cuda_engine(network)
    
    #serialize the engine plan and write to a file to save
    with open('/workspace/mbv3_c4.engine',mode='wb') as plans:
        
        plans.write(engine.serialize())
        

