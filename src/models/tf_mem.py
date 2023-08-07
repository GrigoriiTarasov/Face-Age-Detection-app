import tensorflow as tf

def limit_gpu_gb(gb: float)->None:
    '''Sets for all gpus max mem used to gb. Prevents GPU mem overflowing. Used for libs thats consumes >7Gb on <8Gb systems'''
    
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu, 
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gb*1024)])
