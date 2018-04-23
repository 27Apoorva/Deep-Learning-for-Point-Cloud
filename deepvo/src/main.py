# Code Skeleton - Inspired by Tensoflow RNN tutorial: ptb_word_lm.py
import os
import sys
import numpy as np
import tensorflow as tf
import cv2
import math
import warnings
import argparse
sys.path.append('fn2/')

parser = argparse.ArgumentParser(description='Directory to save model')
parser.add_argument('--model_dir', action="store", dest="model_dir", default='./model_dir')
parser.add_argument('--with_gpu', action="store_true", dest="with_gpu", default=False)
parser.add_argument('--without_angles', action="store_true", dest="only_position", default=False)
parser.add_argument('--use_pretrained_cnn', action="store_true", dest="use_pretrained_cnn", default=False)
parser.add_argument('--start_testing', action="store_true", dest="test_flag", default=False)
FLAGS = parser.parse_args()
""" Hyper Parameters for learning"""
LEARNING_RATE = 0.0005
BATCH_SIZE = 1
LSTM_HIDDEN_SIZE = 550
LSTM_NUM_LAYERS = 2
# global training steps
NUM_TRAIN_STEPS = 2000
TIME_STEPS = 5
MODEL_DIR = FLAGS.model_dir
if FLAGS.with_gpu:
    # FlowNetS Parameters
    import src.flownet_s.flownet_s as fns
    from src.training_schedules import LONG_SCHEDULE
    Mode = fns.Mode

def isRotationMatrix(R):
    """ Checks if a matrix is a valid rotation matrix
        referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    """ calculates rotation matrix to euler angles
        referred from https://www.learnopencv.com/rotation-matrix-to-euler-angles
    """
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def build_rcnn_graph(config, input_, sess):
    """ CNN layers connected to RNN which connects to final output """

    # create 2 LSTMCells
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [config.hidden_size, config.hidden_size]]

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    rnn_inputs = []
    reuse = None
    for stacked_img in input_:
        if FLAGS.use_pretrained_cnn:
            rnn_inputs.append(get_optical_flow(stacked_img, reuse=reuse))
        else:
            rnn_inputs.append(cnn_layers(stacked_img, reuse=reuse))
        reuse = True
    # Flattening the final convolution layers to feed them into RNN
    rnn_inputs = [tf.reshape(rnn_inputs[i],[-1, 20*6*1024]) for i in range(len(rnn_inputs))]
    assert rnn_inputs[0].shape == (config.batch_size, 20*6*1024)

    #max_time = len(rnn_inputs)

    #rnn_inputs = tf.convert_to_tensor(rnn_inputs)
    #tf.summary.histogram('final_cnn_layer_activations', rnn_inputs)

    #config._initial_state = config.get_initial_state()
    # 'outputs' is a tensor of shape [batch_size, max_time, 1000]
    # 'state' is a N-tuple where N is the number of LSTMCells containing a
    # tf.contrib.rnn.LSTMStateTuple for each cell
    outputs, state = tf.nn.static_rnn(cell=multi_rnn_cell,
                                       inputs=rnn_inputs,
                                       dtype=tf.float32)
    # Tensor shaped: [batch_size, max_time, cell.output_size]
    #outputs = tf.unstack(outputs, max_time, axis=1)
    assert outputs[0].shape == (config.batch_size, config.hidden_size)
    return outputs, state

def get_ground_6d_poses(p):
    """ For 6dof pose representaion """
    # import pdb; pdb.set_trace()
    pos = np.array([p[3], p[7], p[11]])
    R = np.array([[p[0], p[1], p[2]], [p[4], p[5], p[6]], [p[8], p[9], p[10]]])
    angles = rotationMatrixToEulerAngles(R)
    return np.concatenate((pos, angles))

def cnn_layers(input_layer, reuse = None):
        """ input: input_layer of concatonated images (img, img_next) where \
                shape of each imgae is (1280, 384, 3)
            output: 6th convolutional layer
            The structure of the CNN is inspired by the network for optical flow estimation
                in A. Dosovitskiy, P. Fischery, E. Ilg, C. Hazirbas, V. Golkov, P. van der
                Smagt, D. Cremers, T. Brox et al Flownet: Learning optical flow
                with convolutional networks, in Proceedings of IEEE International
                Conference on Computer Vision (ICCV)
        """
        # input_layer size [1280, 384, 6]
        # Convolutional Layer #1
        # Computes 32 features using a 7x7 filter with ReLU activation.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 1280, 384, 1]
        # Output Tensor Shape: [batch_size, 1280, 384, 32]
        with tf.variable_scope("cnns", reuse=reuse):
            conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=64,
                kernel_size=[7, 7],
                padding="same",
                strides=2,
                reuse = reuse,
                activation=tf.nn.relu, name='cnv1')
            # Pooling Layer #1
            # First max pooling layer with a 2x2 filter and stride of 2
            # Input Tensor Shape: [batch_size, 1280, 384, 64]
            # Output Tensor Shape: [batch_size, 640, 192, 64]
            conv2 = tf.layers.conv2d(
                    inputs=conv1,
                    filters=128,
                    kernel_size=[5, 5],
                    padding ="same",
                    strides=2,
                    reuse = reuse,
                    activation=tf.nn.relu, name='cnv2')
            conv3 = tf.layers.conv2d(
                    inputs=conv2,
                    filters=256,
                    kernel_size=[5, 5],
                    padding ="same",
                    strides=2,
                    reuse = reuse,
                    activation=tf.nn.relu, name='cnv3')
            conv3_1 = tf.layers.conv2d(
                    inputs=conv3,
                    filters=256,
                    kernel_size=[3, 3],
                    padding ="same",
                    strides=1,
                    reuse = reuse,
                    activation=tf.nn.relu, name='cnv3_1')
            conv4 = tf.layers.conv2d(
                    inputs=conv3_1,
                    filters=512,
                    kernel_size=[3, 3],
                    padding ="same",
                    strides=2,
                    reuse = reuse,
                    activation=tf.nn.relu, name='cnv4')
            conv4_1 = tf.layers.conv2d(
                    inputs=conv4,
                    filters=512,
                    kernel_size=[3, 3],
                    padding ="same",
                    strides=1,
                    reuse = reuse,
                    activation=tf.nn.relu, name='cnv4_1')
            conv5 = tf.layers.conv2d(
                    inputs=conv4_1,
                    filters=512,
                    kernel_size=[3, 3],
                    padding ="same",
                    strides=2,
                    reuse = reuse,
                    activation=tf.nn.relu, name='cnv5')
            conv5_1 = tf.layers.conv2d(
                    inputs=conv5,
                    filters=512,
                    kernel_size=[3, 3],
                    padding ="same",
                    strides=1,
                    reuse = reuse,
                    activation=tf.nn.relu, name='cnv5_1')
            output = tf.layers.conv2d(
                    inputs=conv5_1,
                    filters=1024,
                    kernel_size=[3, 3],
                    padding ="same",
                    reuse = reuse,
                    strides=2, name='output')
            """ The output is connected to RNN
            """
        return output

def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    for i in not_initialized_vars: # only for testing
        print(i.name)

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

def get_optical_flow(input_layer, reuse = None, sess=None):
    flownet = fns.FlowNetS(mode=Mode.TRAIN, debug=True, reuse=reuse)

    inputs = {
            'input_a': input_layer[:, :, :, :3],
            'input_b': input_layer[:, :, :, 3:6]
            }
    training_schedule = LONG_SCHEDULE
    output  = flownet.model(inputs, training_schedule, trainable=False)
    return output

# Dataset Class
class Kitty(object):
    """ Class for manipulating Dataset"""
    def __init__(self, config, data_dir='/media/appu/EXTERNAL_HDD/data_odometry_color/dataset_2/', isTraining=True):
        self._config = config
        self._data_dir= data_dir
        self._img_height, self._img_width = 384, 1280
        self._current_initial_frame = 0
        self._current_trajectory_index = 0
        self._prev_trajectory_index = 0
        self._current_train_epoch = 0
        self._current_test_epoch = 0
        self._training_trajectories = [5,3]
        self._test_trajectories = [5,3]
        if isTraining:
            self._current_trajectories = self._training_trajectories
        else:
            self._current_trajectories = self._test_trajectories
        if not config.only_position:
            self._pose_size = 6
        else:
            self._pose_size = 3

    def get_image(self, trajectory, frame_index):
	img_path =  self._data_dir + 'sequences/'+ '%02d' % trajectory + '/image_2/' +  '%06d' % frame_index + '.png'
        img = cv2.imread( self._data_dir + 'sequences/'+ '%02d' % trajectory + '/image_2/' +  '%06d' % frame_index + '.png')
        if img is not None:
            # Normalizing and Subtracting mean intensity value of the corresponding image
            img = img/np.max(img)
            img = img - np.mean(img)
            img = cv2.resize(img, (self._img_width, self._img_height), fx=0, fy=0)
        return img, img_path

    def get_poses(self, trajectory):
        with open(self._data_dir + 'poses/' +  '%02d' % trajectory + '.txt') as f:
            poses = np.array([[float(x) for x in line.split()] for line in f])
        return poses

    def _set_next_trajectory(self, isTraining):
        print 'in _set_next_trajectory, current_trj_index is %d'%self._current_trajectory_index
        # import pdb; pdb.set_trace()
        if (self._current_trajectory_index < len(self._current_trajectories)-1):
        # if(True):
            self._prev_trajectory_index = self._current_trajectory_index
            self._current_trajectory_index += 1
            self._current_initial_frame = 0
        else:
            print 'New Epoch Started'
            if isTraining:
                self._current_train_epoch += 1
            else:
                import pdb; pdb.set_trace()
                self._current_test_epoch += 1
            self._prev_trajectory_index = self._current_trajectory_index
            self._current_trajectory_index = 0
            self._current_initial_frame = 0
            # self._current_trajectory_index += 1

    def get_next_batch(self, isTraining):
        """ Function that returns the batch for dataset
        """
        img_batch = []
        label_batch = []
        img_path_batch = []
        print('in get_next_batch function PDB NEXT:')
        # import pdb; pdb.set_trace()
        if isTraining:
            self._current_trajectories = self._training_trajectories
        else:
            self._current_trajectories = self._test_trajectories

        poses = self.get_poses(self._current_trajectories[self._current_trajectory_index])

        for j in range(self._config.batch_size):
            img_stacked_series = []
            labels_series = []
            img_path_series = []
            print('Current Trajectory is : %d'% self._current_trajectories[self._current_trajectory_index])

            read_img, read_path = self.get_image(self._current_trajectories[self._current_trajectory_index], self._current_initial_frame + self._config.time_steps)

            if (read_img is None):
            # if(True ):
                self._set_next_trajectory(isTraining)
            for i in range(self._current_initial_frame, self._current_initial_frame + self._config.time_steps):
                img1, img1_path = self.get_image(self._current_trajectories[self._current_trajectory_index], i)
                img2, img2_path = self.get_image(self._current_trajectories[self._current_trajectory_index], i+1)
                img_aug = np.concatenate([img1, img2], -1)
                img_stacked_series.append(img_aug)
                img_path_series.append(img1_path)
                cf = self._current_initial_frame
                if self._pose_size == 3:
                    pose = np.array([poses[i,3], poses[i,7], poses[i,11]]) - np.array([poses[cf,3], poses[cf,7], poses[cf,11]])
                else:
                    pose = get_ground_6d_poses(poses[i,:]) - get_ground_6d_poses(poses[cf,:])
                labels_series.append(pose)
                # self._set_next_trajectory(isTraining)
            img_batch.append(img_stacked_series)
            img_path_batch.append(img_path_series)
            label_batch.append(labels_series)
            self._current_initial_frame += self._config.time_steps
        label_batch = np.array(label_batch)
        img_batch = np.array(img_batch)
        # print label_batch.shape
        # print img_batch.shape
        # print("Label_batch")
        # print label_batch[0,:,:]
        # import pdb; pdb.set_trace()
        return img_batch, label_batch, img_path_batch

# Config class
class Config(object):
    """configuration of RNN """
    def __init__(self, lstm_hidden_size=1000, lstm_num_layers=2, batch_size=1, num_steps= 20, learning_rate=0.001, only_position=True, time_steps=4):
        self.hidden_size = lstm_hidden_size
        self.num_layers = lstm_num_layers
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.only_position = only_position
        self.time_steps = time_steps

def find_global_step():
    model_dir = MODEL_DIR
    if os.path.isdir(model_dir):
        metafiles = [f for f in os.listdir(model_dir) if
                (os.path.isfile(os.path.join(model_dir, f)) and f.endswith(".meta"))]
        if metafiles:
            metafiles = sorted(metafiles)
            print('MetaFiles : ',metafiles)
            if metafiles[-1][11:-5] == "":
                global_step = 5000
            else:
                global_step = int(metafiles[-1][11:-5])
            resume_Training = True
        else:
            global_step = 0
            resume_Training = False
    else:
        global_step = 0
        resume_Training = False
    return global_step, resume_Training

def inference():
        config = Config(lstm_hidden_size=LSTM_HIDDEN_SIZE, lstm_num_layers=LSTM_NUM_LAYERS,
            time_steps=TIME_STEPS, num_steps=NUM_TRAIN_STEPS, batch_size=BATCH_SIZE, only_position=FLAGS.only_position)
        # configuration
        config_proto = tf.ConfigProto(device_count = {'GPU': 0})
        sess = tf.Session(config=config_proto)
        kitty_data = Kitty(config)
        """ input_batch must be in shape of [?, TIME_STEPS, 384, 1280, 6] """
        #tf.reset_default_graph()
        print('Restoring Entire Session from checkpoint : %s'%MODEL_DIR+"model.meta")
        imported_meta = tf.train.import_meta_graph(MODEL_DIR + "model.meta")
        print('Sucess')
        imported_meta.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))
        input_data = tf.get_default_graph().get_tensor_by_name("input/Placeholder:0")
        # placeholder for labels
        labels_ = tf.get_default_graph().get_tensor_by_name("input/Placeholder_1:0")
        loss_op = tf.get_default_graph().get_tensor_by_name("loss_l2_norm/loss:0")
        poses = []
        poses.append(tf.get_default_graph().get_tensor_by_name("Wx_plus_b/xw_plus_b:0"))
        poses.append(tf.get_default_graph().get_tensor_by_name("Wx_plus_b/xw_plus_b_1:0"))
        poses.append(tf.get_default_graph().get_tensor_by_name("Wx_plus_b/xw_plus_b_2:0"))
        poses.append(tf.get_default_graph().get_tensor_by_name("Wx_plus_b/xw_plus_b_3:0"))
        poses.append(tf.get_default_graph().get_tensor_by_name("Wx_plus_b/xw_plus_b_4:0"))
        while kitty_data._current_test_epoch < 1:
            input_, ground_truth_batch, img_path_batch = kitty_data.get_next_batch(isTraining=False)
            print (kitty_data._current_test_epoch)
            output = sess.run(poses, feed_dict={input_data:input_})
	    print('len of output: %d'%len(output))
            for i in range(len(output)):
                fh = open("output_file","a")
                fh.write("%f %f %f %f %f %f\n"%(ground_truth_batch[:,i,0],ground_truth_batch[:,i,1],ground_truth_batch[:,i,2],ground_truth_batch[:,i,3],ground_truth_batch[:,i,4],ground_truth_batch[:,i,5])) #str(read_img) + str(pose) + '\n')
                fh.close()
                fh = open("estimated","a")
                fh.write("%f %f %f %f %f %f\n"%(output[i][0,0],output[i][0,1],output[i][0,2],output[i][0,3],output[i][0,4],output[i][0,5])) #str(read_img) + str(pose) + '\n')
                fh.close()
                fh = open("img_file_names\n","a")
                fh.write(str(img_path_batch[0][i])) #str(read_img) + str(pose) + '\n')
                fh.close()

def main():
    """ main function """
    config = Config(lstm_hidden_size=LSTM_HIDDEN_SIZE, lstm_num_layers=LSTM_NUM_LAYERS,
            time_steps=TIME_STEPS, num_steps=NUM_TRAIN_STEPS, batch_size=BATCH_SIZE, only_position=FLAGS.only_position)
    # configuration
    kitty_data = Kitty(config)
    if FLAGS.with_gpu:
        sess = tf.Session()
    else:
        config_proto = tf.ConfigProto(device_count = {'GPU': 0})
        sess = tf.Session(config=config_proto)
    if not config.only_position:
        pose_size = 6
    else:
        pose_size = 3
        warnings.warn("Warning! Orientation data is ignored!")

    # only for gray scale dataset, for colored channels will be 6
    height, width, channels = 384, 1280, 6

    global_step, resume_Training = find_global_step()
    print('Global Step : %d'%global_step)
    if resume_Training:
        tf.reset_default_graph()
        sess = tf.Session()
        print('Restoring Entire Session from checkpoint : %s'%MODEL_DIR+"model.meta")
        imported_meta = tf.train.import_meta_graph(MODEL_DIR + "model.meta")
        print('Sucess')
        input_data = tf.get_default_graph().get_tensor_by_name("input/Placeholder:0")
        # placeholder for labels
        labels_ = tf.get_default_graph().get_tensor_by_name("input/Placeholder_1:0")
        loss_op = tf.get_default_graph().get_tensor_by_name("loss_l2_norm/loss:0")
        tf.summary.scalar('loss_l2_norm', loss_op)
        train_op = tf.get_default_graph().get_operation_by_name("train/Adam")
        merged = tf.summary.merge_all()
        #saver = tf.train.Saver()
        #saver.restore(sess, '')
    else:
        with tf.name_scope('input'):
            # placeholder for input
            input_data = tf.placeholder(tf.float32, [config.batch_size, config.time_steps, height, width, channels])
            # placeholder for labels
            labels_ = tf.placeholder(tf.float32, [config.batch_size, config.time_steps, pose_size])

        with tf.name_scope('unstacked_input'):
            # Unstacking the input into list of time series
            input_ = tf.unstack(input_data, config.time_steps, 1)
            # Unstacking the labels into the time series
            pose_labels = tf.unstack(labels_, config.time_steps, 1)


        # Building the RCNN Network which
        # which returns the time series of output layers
        with tf.name_scope('RCNN'):
            (outputs, _)  = build_rcnn_graph(config, input_, sess=sess)
        if FLAGS.use_pretrained_cnn:
            # Restoring FlowNetS variables
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='FlowNetS')
            pretrained_saver = tf.train.Saver(var_list=var_list)
            pretrained_saver.restore(sess, './fn2/checkpoints/FlowNetS/flownet-S.ckpt-0')
        ## Output layer to compute the output
        with tf.name_scope('weights'):
            regression_w = tf.get_variable('regression_w', shape=[config.hidden_size, pose_size], dtype=tf.float32)
        with tf.name_scope('biases'):
            regression_b = tf.get_variable("regression_b", shape=[pose_size], dtype=tf.float32)

        # Pose estimate by multiplication with RCNN_output and Output layer
        with tf.name_scope('Wx_plus_b'):
            pose_estimated = [tf.nn.xw_plus_b(output_state, regression_w, regression_b) for output_state in outputs]
            max_time = len(pose_estimated)

        # Converting the list of tensor into a tensor
        # Probably this is the part that is unnecessary and causing problems (slowing down the computations)

        # Loss function for all the frames in a batch
        with tf.name_scope('loss_l2_norm'):
            position = [pose_es[:,:3] - pose_lab[:,:3] for pose_es, pose_lab in zip(pose_estimated, pose_labels)]
            angles = [pose_es[:,3:6] - pose_lab[:,3:6] for pose_es, pose_lab in zip(pose_estimated, pose_labels)]
            pose_error = (tf.square(position))
            angle_error = (tf.square(angles))
            loss_op = tf.reduce_sum(pose_error + 100*angle_error, name='loss')
            tf.summary.scalar('loss_l2_norm', loss_op)

        #optimizer
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate,
                    beta1=0.9,
                    beta2=0.999,
                    epsilon=1e-08,
                    use_locking=False,
                    name='Adam')
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=config.learning_rate)
            train_op = optimizer.minimize(loss_op)
        # Merge all the summeries and write them out to model_dir
        # by default ./model_dir
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()

    if (global_step != 0):
        imported_meta.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))
    else:
        initialize_uninitialized(sess)
    train_writer = tf.summary.FileWriter(MODEL_DIR + 'train', sess.graph)
    test_writer = tf.summary.FileWriter(MODEL_DIR + 'test')
    # Training and Testing Loop
    #for i in range(global_step, global_step + config.num_steps):
    # i = 0
    while kitty_data._current_train_epoch < 5:
        print('current_train_epoch: %d'%kitty_data._current_train_epoch)
        for i in range(global_step, global_step + config.num_steps):
        # for i in range(0, 5):
            print('step : %d'%i)
            if i % 10 == 0:  # Record summaries and test-set accuracy
                batch_x, batch_y, path = kitty_data.get_next_batch(isTraining=False)
                # print batch_y
                summary, acc = sess.run(
                        [merged, loss_op], feed_dict={input_data:batch_x, labels_:batch_y})
                test_writer.add_summary(summary, i)
                print('Accuracy at step %s: %s' % (i, acc))
            else:  # Record train set summaries, and train
                batch_x, batch_y,path = kitty_data.get_next_batch(isTraining=True)
                summary, _ = sess.run(
                    [merged, train_op], feed_dict={input_data:batch_x, labels_:batch_y})
                train_writer.add_summary(summary, i)
                train_loss = sess.run(loss_op,
                        feed_dict={input_data:batch_x, labels_:batch_y})
                print('Train_error at step %s: %s' % (i, train_loss))
        # i += 1
    save_path = saver.save(sess, MODEL_DIR + 'model')
    print("Model saved in file: %s" % save_path)
    print("epochs trained: " + str(kitty_data._current_train_epoch))
    train_writer.close()
    test_writer.close()


if __name__ == "__main__":
    if FLAGS.test_flag:
        inference()
    else:
        main()

    """
    print find_global_step()
    # Test Code for checking feeding mechanism
    config = Config(lstm_hidden_size=LSTM_HIDDEN_SIZE, lstm_num_layers=LSTM_NUM_LAYERS,
            time_steps=TIME_STEPS, num_steps=NUM_TRAIN_STEPS, batch_size=BATCH_SIZE)
    kitty_data = Kitty(config)
    for i in range(config.num_steps):
        batch_x, batch_y = kitty_data.get_next_batch(isTraining=False)
        height, width, channels = 376, 1241, 2
    print('epochs: %d'%kitty_data._current_train_epoch)
    with tf.name_scope('input'):
        input_data = tf.placeholder(tf.float32, [config.time_steps, None, height, width, channels])
        # placeholder for labels
        labels_ = tf.placeholder(tf.float32, [config.time_steps, None, 3])
    with tf.Session() as sess:
        sess.run([input_data, labels_], feed_dict={input_data:batch_x, labels_:batch_y})
    """
