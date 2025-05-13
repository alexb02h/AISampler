#Loading all our libraries
import tensorflow as tf
import random
import numpy as np
import librosa
import os
import keras
import tensorflow_model_optimization as tfmot
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model
from keras.layers import Lambda
from keras.saving import register_keras_serializable
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.utils import shuffle
import tensorflow.keras.backend as K

keras.config.enable_unsafe_deserialization()

#Function to resize tensor to make the shape of the reference tensor
@register_keras_serializable()
class crop_to_match(tf.keras.layers.Layer):
	def __init__(self,**kwargs): super(crop_to_match,self).__init__(**kwargs)

	def call(self,inputs):
		x,skip = inputs
		min_len = tf.minimum(tf.shape(x)[1],tf.shape(skip)[1])
		return x[:,:min_len,:], skip[:,:min_len,:]
	
	def get_config(self):
		config = super(crop_to_match,self).get_config()
		return config

#Loading .wav files from file_path
def load_wav(file_path):
	target_shape = 88200
	waveform, sr = librosa.load(file_path,sr=44100, mono=True)
	
	if len(waveform) < target_shape: 
		waveform = np.pad(waveform, (0,target_shape - len(waveform)))
	else: 
		waveform = waveform[:target_shape]
	
	return waveform.reshape((target_shape,1))

training_path = "musdb18hq/train"

def add_target_noise(target,stddev=0.01): return target + np.random.normal(0,stddev, target.shape).astype(np.float32)

def Augment_audio(x,y_v,y_d, sr=44100):
	gain = random.uniform(0.8,1.2)
	noise_level = 0.001 * random.random()

	min_len = min(len(x), len(y_v), len(y_d))
	x = x[:min_len]
	y_v = y_v[:min_len]
	y_d = y_d[:min_len]

	def augment_wave(wave):
		wave = np.squeeze(wave)
		wave *= gain
		noise = noise_level * np.random.randn(len(wave))
		wave += noise
		wave = np.pad(wave,(0,max(0,44100-len(wave))))[:44100]
		return np.expand_dims(wave,axis=-1)

	return augment_wave(x),augment_wave(y_v),augment_wave(y_d)

def augment_dataset(X,Y_vocals, Y_drums,sr=44100,num_augmented=2):
	X_aug = []
	Y_vocals_aug = []
	Y_drums_aug = []

	for x, y_v, y_d in zip(X, Y_vocals, Y_drums):
		for _ in range(num_augmented):
			x_aug, y_v_aug, y_d_aug = Augment_audio(x,y_v,y_d,sr)
			X_aug.append(x_aug)
			Y_vocals_aug.append(y_v_aug)
			Y_drums_aug.append(y_d_aug)
	print("Augmented shapes:", np.array(X_aug).shape, np.array(Y_vocals_aug).shape, np.array(Y_drums_aug).shape)
	return np.array(X_aug),np.array(Y_vocals_aug),np.array(Y_drums_aug)

def load_datasets():
	input = np.load("inputs.npy")
	vocals = np.load("vocals.npy")
	drums = np.load("drums.npy")
	return input, vocals, drums

X_train, Y_vocals, Y_drums = load_datasets() #Loading the training data

X_orig_train, X_orig_val, Y_vocals_train, Y_vocals_val, Y_drums_train, Y_drums_val = train_test_split(
    X_train, Y_vocals, Y_drums, test_size=0.1, random_state=42
)

target_shape = (X_train.shape[0],44100,1)


X_aug,Y_vocals_aug, Y_drums_aug = augment_dataset(X_orig_train,Y_vocals_train, Y_drums_train)

X_train_combined = np.concatenate([X_orig_train, X_aug])
Y_vocals_combined = np.concatenate([Y_vocals_train,Y_vocals_aug])
Y_drums_combined = np.concatenate([Y_drums_train, Y_drums_aug])

X_train_combined = X_train_combined.astype(np.float32)
Y_vocals_combined = Y_vocals_combined.astype(np.float32)
Y_drums_combined = Y_drums_combined.astype(np.float32)

X_train_combined, Y_vocals_combined, Y_drums_combined = shuffle(
	X_train_combined, Y_vocals_combined, Y_drums_combined, random_state=42
)

@register_keras_serializable()
class normalize(tf.keras.layers.Layer):
	def call(self, inputs): return inputs/(K.max(K.abs(inputs)) + K.epsilon())
	def compute_output_shape(self,input_shape): return input_shape

def build_waveunet(input_shape, num_classes):
	inputs = layers.Input(shape=input_shape)

	#x = layers.Lambda(lambda x:x / (tf.reduce_max(tf.abs(x)) + 1e-7))(inputs)

	skips = []
	filters = [16,32,64,128,256]

	x = inputs

	#Downsampling : extracting features and reduing the sequence length
	for f in filters: 
		x = layers.Conv1D(f,kernel_size=5,padding='same')(x)
		x = layers.BatchNormalization()(x)
		x = layers.LeakyReLU(0.2)(x)

		x = layers.Conv1D(f,kernel_size=5,padding='same')(x)
		x = layers.BatchNormalization()(x)
		x = layers.LeakyReLU(0.2)(x)

		skips.append(x)
		x = layers.MaxPooling1D(2,padding='same')(x)

	x = layers.Conv1D(512,kernel_size=5,padding='same')(x)
	x = layers.BatchNormalization()(x)
	x = layers.LeakyReLU(0.2)(x)

	#Upsampling : reverses the Downsampling and recovers the original sequence length
	for f in reversed(filters):
		x = layers.UpSampling1D(2)(x)
		skip = skips.pop()

		if x.shape[1] > skip.shape[1]:
			x = layers.Cropping1D((0,x.shape[1] - skip.shape[1]))(x)
		elif x.shape[1] < skip.shape[1]:
			skip = layers.Cropping1D((0,skip.shape[1] - x.shape[1]))(skip)		

		x = layers.Concatenate()([x,skip])

		x = layers.Conv1D(f,kernel_size=5,padding='same')(x)
		x = layers.BatchNormalization()(x)
		x = layers.LeakyReLU(0.2)(x)

		x = layers.Conv1D(f,kernel_size=5,padding='same')(x)
		x = layers.BatchNormalization()(x)
		x = layers.LeakyReLU(0.2)(x)

	vocals = layers.Conv1D(1,5,padding='same', name='vocals')(x)
	drums = layers.Conv1D(1,5,padding='same',name='drums')(x)

	return Model(inputs=inputs,outputs={"vocals":vocals,"drums":drums})

def si_sdr_loss(y_true,y_pred,eps=1e-8):
	y_true = tf.squeeze(y_true,axis=-1)
	y_pred = tf.squeeze(y_pred,axis=-1)

	dot = tf.reduce_sum(y_true * y_pred,axis=1,keepdims=True)
	s_target = dot * y_true / (tf.reduce_sum(y_true ** 2,axis=1,keepdims=True) + eps)
	e_noise = y_pred - s_target

	si_sdr = 10 * tf.math.log(
		(tf.reduce_sum(s_target ** 2, axis=1) + eps)/
		(tf.reduce_sum(e_noise ** 2,axis=1) + eps)
	) / tf.math.log(10.0)

	return -tf.reduce_mean(si_sdr)

def multi_resolution_stft_loss(y_true, y_pred):
	y_true = tf.squeeze(y_true,axis=-1)
	y_pred = tf.squeeze(y_pred,axis=-1)

	resolutions = [
		(512,128),
		(1024,256),
		(2048,512)
	]	
	
	losses = []
	for frame_length,frame_step  in resolutions:
		stft_true = tf.signal.stft(y_true,frame_length=frame_length,frame_step=frame_step,fft_length=frame_length)
		stft_pred = tf.signal.stft(y_pred,frame_length=frame_length,frame_step=frame_step,fft_length=frame_length)

		mag_true = tf.abs(stft_true)
		mag_pred = tf.abs(stft_pred)

		losses.append(tf.reduce_mean(tf.abs(mag_true - mag_pred)))

	return tf.reduce_mean(losses)

def combined_loss(y_true,y_pred,alpha=0.7):
	loss_sisdr = si_sdr_loss(y_true,y_pred)
	loss_stft = multi_resolution_stft_loss(y_true,y_pred)
	return alpha * loss_sisdr + (1 - alpha) * loss_stft

input_shape = (44100,1)
num_classes = 2

model = build_waveunet(input_shape=input_shape,num_classes=num_classes)
model.summary()

custom_objects = {
	'crop_to_match' : crop_to_match,
	'normalize' : normalize,
	'<lambda>': lambda x: x[:,:44100,:]
}

#model = load_model('Wave-U-Net.keras',custom_objects=custom_objects)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay( 
        initial_learning_rate=3e-5,
        decay_steps=2000,
        decay_rate=0.98
)

losses = {
	"vocals": lambda y_true,y_pred: combined_loss(y_true,y_pred,alpha=0.7),
	"drums": lambda y_true,y_pred: combined_loss(y_true,y_pred,alpha=0.7)
}

optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule, clipnorm = 1.0)
model.compile(
		optimizer=optimizer,
		loss=losses,
		metrics={
			"vocals":[si_sdr_loss, multi_resolution_stft_loss],
			 "drums":[si_sdr_loss, multi_resolution_stft_loss]
		}
)

callbacks = [
	tf.keras.callbacks.EarlyStopping(
		monitor='val_loss',
		patience=10,
		min_delta=1e-3,
		restore_best_weights=True),
	tf.keras.callbacks.ModelCheckpoint(
		'Wave-U-Net.weights.h5', 
		save_best_only=True,
		monitor='val_loss'),
	tf.keras.callbacks.TensorBoard(
		log_dir='./logs',
		histogram_freq=0),
	tf.keras.callbacks.CSVLogger("training_history.csv"),
]



model.fit(
	X_train_combined,
	{"vocals":Y_vocals_combined,"drums":Y_drums_combined},
	batch_size=8, 
	epochs=100, 
	validation_data=(X_orig_val, {"vocals":Y_vocals_val, "drums":Y_drums_val}), 
	callbacks=callbacks)

sample_output = model.predict(X_train[:1])
print(np.min(sample_output),np.max(sample_output))

model.save_weights("Wave-U-Net.weights.h5")
