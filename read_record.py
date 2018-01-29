import tensorflow as tf
from PIL import Image
if __name__=='__main__':
    tfrecords_filename = "flowers_train_0000-of-0001.tfrecord"
    filename_queue = tf.train.string_input_producer([tfrecords_filename],) 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/feature': tf.FixedLenFeature((128), tf.float32),
                                           'image/encoded' : tf.FixedLenFeature([], tf.string),
                                       })  
    image = tf.image.decode_jpeg(features['image/encoded'])
    #image = tf.reshape(image, [224, 224, 3])
   # image = tf.reshape(image, [7,30])
    feature = tf.cast(features['image/feature'], tf.float32)
    with tf.Session() as sess: 
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        example, l = sess.run([image,feature])
        #img=Image.fromarray(example, 'RGB')
        #img.save('./'+str(i)+'_''Label_'+str(l)+'.jpg')
        #print(example, l)
        print(l.shape)
        print(example.shape)
        coord.request_stop()
        coord.join(threads)
