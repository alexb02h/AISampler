import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf
import keras
import os
import h5py
import musdb
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
from skimage.transform import resize
from mir_eval.separation import bss_eval_sources
import tensorflow.keras.backend as K

class normalize(tf.keras.layers.Layer):
        def call(self, inputs): return inputs/(K.max(K.abs(inputs)) + K.epsilon())

class crop_to_match(tf.keras.layers.Layer):
	def __init__(self,**kwargs): super(crop_to_match,self).__init__(**kwargs)

	def call(self,inputs):
		x,skip = inputs
		min_len = tf.minimum(tf.shape(x)[1],tf.shape(skip)[1])
		return x[:,:min_len,:], skip[:,:min_len,:]

	def get_config(self):
                config = super(crop_to_match,self).get_config()
                return config

def build_waveunet(input_shape, num_classes):
	inputs = layers.Input(shape=input_shape)

	x = layers.Lambda(lambda x:x / (tf.reduce_max(tf.abs(x)) + 1e-7))(inputs)

	skips = []
	filters = [16,32,64,128,256]


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

	vocals = layers.Conv1D(1,5,padding='same',activation='tanh', name='vocals')(x)
	drums = layers.Conv1D(1,5,padding='same',activation='tanh',name='drums')(x)

	return Model(inputs=inputs,outputs={"vocals":vocals,"drums":drums})

def process_song(audio,model,chunk_size=44100,overlap=0.25):
	total_length = len(audio)
	overlap_samples = int(chunk_size * overlap)
	hop_size = chunk_size - overlap_samples
	num_chunks = (total_length - overlap_samples) // hop_size + 1
	padded_length = (num_chunks - 1) * hop_size + chunk_size

	if padded_length > total_length : audio = np.pad(audio,(0,padded_length - total_length))

	vocals = np.zeros(padded_length)
	drums = np.zeros(padded_length)

	window = np.hanning(overlap_samples * 2)
	fade_in = window[:overlap_samples]
	fade_out = window[overlap_samples:]

	for i in range(0,padded_length - chunk_size + 1, hop_size):
		chunk = audio[i:i + chunk_size]
		chunk = chunk / (np.max(np.abs(chunk)) + 1e-9)
		model_input = chunk.reshape(1,chunk_size,1)

		with tf.device('/CPU:0'): predictions = model.predict(model_input,verbose=0)

		vocals_chunk = predictions['vocals'].squeeze()
		drums_chunk = predictions['drums'].squeeze()

		if i > 0:
			vocals_chunk[:overlap_samples] *= fade_in
			drums_chunk[:overlap_samples] *= fade_in

			vocals[i:i+overlap_samples] *= fade_out
			drums[i:i+overlap_samples] *= fade_out

		vocals[i:i+chunk_size] += vocals_chunk
		drums[i:i+chunk_size] += drums_chunk

	return vocals[:total_length],drums[:total_length]

def evaluate_sdr(reference, estimate):
	ref = reference[np.newaxis,:]
	est = estimate[np.newaxis,:]
	sdr, _, _, _ = bss_eval_sources(ref,est)
	return sdr[0]

def main():
	print("Loading model...")
	
	model = build_waveunet(input_shape=(44100,1),num_classes=2)
	model.load_weights("Wave-U-Net.weights.h5")
	print("Model Loaded")

	print("Loading MUSDB18 test set...")
	mus = musdb.DB(root="/Users/alexander/Desktop/AISampler/musdb18hq", subsets="test", is_wav=True)
	print(f"Loaded {len(mus)} test tracks.")

	os.makedirs("outputs", exist_ok=True)

	results=[]	

	for track in mus:
		print(f"Processing track: {track.name}")
		mixture = np.mean(track.audio.T, axis=0)
		vocals_ref = track.targets['vocals'].audio.T[0]
		drums_ref = track.targets['drums'].audio.T[0]

		vocals_est,drums_est = process_song(mixture, model)

		sf.write(f"outputs/{track.name}_vocals.wav", vocals_est, track.rate)
		sf.write(f"outputs/{track.name}_drums.wav", drums_est, track.rate)

		vocals_sdr = evaluate_sdr(vocals_ref[:len(vocals_est)],vocals_est)
		drums_sdr = evaluate_sdr(drums_ref[:len(drums_est)],drums_est)

		print(f"SDR - Vocals: {vocals_sdr:.2f} dB | Drums: {drums_sdr:.2f} dB")
		results.append((track.name, vocals_sdr,drums_sdr))

	avg_vocals = np.mean([v for _, v ,_ in results])
	avg_drums = np.mean([d for _, _, d in results])
	print(f"\nAverage Vocals SDR: {avg_vocals:.2f} dB")
	print(f"Average Drums SDR: {avg_drums:.2f} dB")

if __name__ == "__main__": main()

