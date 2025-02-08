import tensorflow as tf
from tensorflow.keras import layers, Model

def crop_and_concat(x,skip):
	def crop(inputs):
		x,skip = inputs
		x_shape = tf.shape(x)
		skip_shape = tf.shape(skip)
		crop_size = skip_shape[1] - x_shape[1]
		skip_cropped = tf.slice(skip,[0,0,0], [-1, x_shape[1], -1])
		return tf.concat([x,skip_cropped], axis=-1)
	return layers.Lambda(crop)([x, skip])

def build_waveunet(input_shape, num_classes):
   inputs = layers.Input(shape=input_shape)
  
   filters = [16,32,64,128,256]
   skips = []
   x = inputs
   for f in filters:
       x = layers.Conv1D(f, kernel_size=15,strides=2, padding='same', activation='relu')(x)
       skips.append(x)
      
   x = layers.Conv1D(512,kernel_size=15, strides=1, padding='same', activation='relu')(x)
  
   for f, skip in zip(reversed(filters), reversed(skips)):
       x = layers.Conv1DTranspose(f, kernel_size=15, strides=2, padding='same', activation='relu')(x)
       x = crop_and_concat(x, skip)
      
   x = layers.GlobalAveragePooling1D()(x)
   outputs = layers.Dense(num_classes, activation='sigmoid')(x)
  
   model = Model(inputs,outputs)
   return model


input_shape = (44100,1)
num_classes = 11
model = build_waveunet(input_shape,num_classes)
model.summary()
