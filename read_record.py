import tensorflow as tf
from PIL import Image
import numpy as np

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/feature': tf.FixedLenFeature((128), tf.float32),
                                           'image/encoded' : tf.FixedLenFeature([], tf.string),
                                       })  
    image=features['image/encoded']
    image_decoded = tf.image.decode_jpeg(image)
	feature = tf.cast(features['image/feature'], tf.float32)
    return image_decoded,feature


if __name__=='__main__':
    tfrecords_filename = "test.tfrecord"
    #image = tf.reshape(image, [224, 224, 3])
    # image = tf.reshape(image, [7,30])
    #number of records in one tfrecord files, you may manually modify this parameter
    num_of_record=4
    with tf.Session() as sess: 
	filename_queue = tf.train.string_input_producer([tfrecords_filename]) 
        image,feature=read_and_decode(filename_queue)
        image.set_shape([160,160,3])

        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
       
        #img=Image.fromarray(example, 'RGB')
        #img.save('./'+str(i)+'_''Label_'+str(l)+'.jpg')
       # print(example, l)

	for i in range(num_of_record):
            example, l = sess.run([image,feature])
	    print('image')
	    print (example)
	    print('feature')
	    print(l)
    
        coord.request_stop()
        coord.join(threads)
