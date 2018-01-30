import tensorflow as tf
from PIL import Image
if __name__=='__main__':
    tfrecords_filename = "/home/yinquan/wanyingd/DistillProject/DistillData/tf_record/flowers_train_00000-of-00005.tfrecord"
    filename_queue = tf.train.string_input_producer([tfrecords_filename],) 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'feature': tf.FixedLenFeature((128), tf.float32),
                                           'image' : tf.FixedLenFeature([], tf.string),
                                       })  
    image = tf.decode_raw(features['image'],tf.float32)
    image = tf.reshape(image, [224, 224, 3])
    #number of record in one tfrecord, you may manually modify this parameter
    num_of_record=10
   # image = tf.reshape(image, [7,30])
    feature = tf.cast(features['feature'], tf.float32)
    with tf.Session() as sess: 
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        for i in range(num_of_record):
            example, l = sess.run([image,feature])
        #img=Image.fromarray(example, 'RGB')
        #img.save('./'+str(i)+'_''Label_'+str(l)+'.jpg')
        #print(example, l)
            print(l)
            print(example)
        coord.request_stop()
        coord.join(threads)
