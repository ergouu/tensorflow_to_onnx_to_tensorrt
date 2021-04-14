import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2


def allocate_buffers(engine,batch_size,data_type):
    """
    allocate memory in host and devices
    h in h_input or h_output means host
    d means device
    """
    
    #pagedlocked means page never be swapped to disk
    h_input=cuda.pagelocked_empty(batch_size*trt.volume(engine.get_binding_shape(0)),dtype=trt.nptype(data_type))
    h_outpu=cuda.pagelocked_empty(batch_size*trt.volume(engine.get_binding_shape(1)),dtype=trt.nptype(data_type))
    
    #allocate GPU mem
    d_input=cuda.mem_alloc(h_input.nbytes)
    d_output=cuda.mem_alloc(h_outpu.nbytes)
    stream=cuda.Stream()
    return h_input,h_outpu,d_input,d_output,stream

def load_image_to_buffer(pics,pagelocked_buffer):
    #load input data to locked memory page
    preprocessed=np.asarray(pics).ravel()
    np.copyto(pagelocked_buffer,preprocessed)

def do_inference(engine,pics_1,h_input_1,h_output_1,d_input_1,d_output_1,stream,batch_size,height,weight):
    
    # trainsfer input data to input memory zone first
    load_image_to_buffer(pics_1,h_input_1)
    
    # create context to execute engine plan
    with engine.create_execution_context() as context:
        # one engine could be executed in several different context
        
        # transfer data to device input
        cuda.memcpy_htod_async(d_input_1,h_input_1)

        # do inference
        context.profiler=trt.Profiler()
        context.execute(batch_size=1,bindings=[int(d_input_1),int(d_output_1)])
        
        #fetch output back to host
        cuda.memcpy_dtoh_async(h_output_1,d_output_1)

        stream.synchronize()
        
        out=h_output_1.reshape((batch_size,-1,height,weight))
        return out

if __name__=='__main__':
    TRT_LOGGER=trt.Logger(trt.Logger.WARNING)
    with trt.Runtime(TRT_LOGGER) as runtime:
        f=open('/workspace/mbv3_c4.engine',mode='rb')
        # deserialize from string file
        engine=runtime.deserialize_cuda_engine(f.read())
        f.close()
        h_input,h_output,d_input,d_output,stream=allocate_buffers(engine,1,trt.float32)
        
        # input data preprocess
        img=cv2.imread('/workspace/1.jpg',cv2.IMREAD_UNCHANGED)
        img=cv2.resize(img,(512,256))
        img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        img=img/127.5-1.0
        pic=np.transpose(img,(2,0,1))
        
        # do inference
        out=do_inference(engine,pic,h_input,h_output,d_input,d_output,stream,1,64,128)
        print(out.shape)
        print(type(out))



