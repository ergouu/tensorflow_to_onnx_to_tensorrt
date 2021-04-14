from tensorflow.python.tools import freeze_graph
import tensorflow as tf

from net import ultranet
from config import CFG

ckpt_path = "/home/ergouu/data/tested_models/mobilenetv3l+espcn+tf1+cellsize4/-11172"

def main():
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, shape=[None, 256, 512, 3], name='input')
    flow = ultranet(x, CFG.BACK_BONE, CFG.CELL_SIZE, is_train=False)
    flow = tf.identity(flow,name='final_out_put')

    with tf.Session() as sess:

        tf.train.write_graph(sess.graph_def, './pb_model', 'model.pb')
        freeze_graph.freeze_graph(
            input_graph='./pb_model/model.pb',
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,
            output_node_names='final_out_put',
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='./pb_model/frozen_model.pb',
            clear_devices=False,
            initializer_nodes=''
            )

    print("done")

if __name__ == '__main__':
    main()