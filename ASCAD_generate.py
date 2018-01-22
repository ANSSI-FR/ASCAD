import os.path
import sys
import h5py
import numpy as np
import random

def check_file_exists(file_path):
	if os.path.exists(file_path) == False:
		print("Error: provided file path '%s' does not exist!" % file_path)
		sys.exit(-1)
	return

# The AES SBox that we will use to generate our labels
AES_Sbox = np.array([
            0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
            0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
            0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
            0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
            0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
            0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
            0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
            0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
            0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
            0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
            0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
            0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
            0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
            0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
            0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
            ])

# Our labelization function:
# It is as simple as the computation of the result of Sbox(p[3] + k[3]) (see the White Paper)
# Note: you can of course adapt the labelization here (say if you want to attack the first byte Sbox(p[0] + k[0])
# or if you want to attack another round of the algorithm).
def labelize(plaintexts, keys):
	return AES_Sbox[plaintexts[:, 2] ^ keys[:, 2]]


# TODO: sanity checks on the parameters
def extract_traces(traces_file, labeled_traces_file, profiling_index = [n for n in range(0, 50000)], attack_index = [n for n in range(50000, 60000)], target_points=[n for n in range(45400, 46100)], profiling_desync=0, attack_desync=0):
	check_file_exists(traces_file)
	check_file_exists(os.path.dirname(labeled_traces_file))
	# Open the raw traces HDF5 for reading
	try:
		in_file  = h5py.File(traces_file, "r")
	except:
		print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % traces_file)
		sys.exit(-1)
	raw_traces = in_file['traces']
	raw_plaintexts = in_file['metadata']['plaintext']
	raw_ciphertexts = in_file['metadata']['ciphertext']
	raw_keys = in_file['metadata']['key']
	raw_masks = in_file['metadata']['masks']
	# Extract a larger set of points to handle desynchronization
	min_target_point = min(target_points)
	max_target_point = max(target_points)
	desync_raw_traces = raw_traces[:, range(min_target_point, max_target_point + 1 + max(profiling_desync, attack_desync))]
	# Apply a desynchronization on the traces if asked to
	# See White Paper 
	# In order to desynchronize our traces, we only shift the
	# target_points by the given amount
	# Select the profiling and attack traces and zoom in on the points of interest
	raw_traces_profiling = np.zeros([len(profiling_index), len(target_points)], desync_raw_traces.dtype)
	profiling_desync_metadata = np.zeros(len(profiling_index), np.uint32)
	curr_trace = 0
	for trace in profiling_index:
		# Desynchronize the profiling traces with the asked amount
		r_desync = random.randint(0, profiling_desync)
		profiling_desync_metadata[curr_trace] = r_desync 
		curr_point = 0
		for point in map(int.__sub__, target_points, [min_target_point] * len(target_points)):
			raw_traces_profiling[curr_trace, curr_point] = desync_raw_traces[trace, point+r_desync]
			curr_point += 1
		curr_trace += 1
	raw_traces_attack = np.zeros([len(attack_index), len(target_points)], desync_raw_traces.dtype)
	attack_desync_metadata = np.zeros(len(attack_index))
	curr_trace = 0
	for trace in attack_index:
		# Desynchronize the profiling traces with the asked amount
		r_desync = random.randint(0, attack_desync)
		attack_desync_metadata[curr_trace] = r_desync 
		curr_point = 0
		for point in map(int.__sub__, target_points, [min_target_point] * len(target_points)):
			raw_traces_attack[curr_trace, curr_point] = desync_raw_traces[trace, point+r_desync]
			curr_point += 1
		curr_trace += 1
	# Compute our labels
	labels_profiling = labelize(raw_plaintexts[profiling_index], raw_keys[profiling_index])
	labels_attack  = labelize(raw_plaintexts[attack_index], raw_keys[attack_index])
	# Open the output labeled file for writing
	try:
		out_file = h5py.File(labeled_traces_file, "w")
	except:
		print("Error: can't open HDF5 file '%s' for writing ..." % labeled_traces_file)
		sys.exit(-1)
	# Create our HDF5 hierarchy in the output file:
	# 	- Profilinging traces with their labels
	#	- Attack traces with their labels
	profiling_traces_group = out_file.create_group("Profiling_traces")
	attack_traces_group = out_file.create_group("Attack_traces")
	# Datasets in the groups
	profiling_traces_group.create_dataset(name="traces", data=raw_traces_profiling, dtype=raw_traces_profiling.dtype)
	attack_traces_group.create_dataset(name="traces", data=raw_traces_attack, dtype=raw_traces_attack.dtype)
	# Labels in the groups
	profiling_traces_group.create_dataset(name="labels", data=labels_profiling, dtype=labels_profiling.dtype)
	attack_traces_group.create_dataset(name="labels", data=labels_attack, dtype=labels_attack.dtype)	
	# Put the metadata (plaintexts, keys, ...) so that one can check the key rank
	metadata_type = np.dtype([("plaintext", raw_plaintexts.dtype, (16,)),
				  ("ciphertext", raw_ciphertexts.dtype, (16,)),
				  ("key", raw_keys.dtype, (16,)),
				  ("masks", raw_masks.dtype, (16,)),
				  ("desync", np.uint32, (1,)),
				 ])
	profiling_metadata = np.array([(raw_plaintexts[n], raw_ciphertexts[n], raw_keys[n], raw_masks[n], profiling_desync_metadata[k]) for n, k  in zip(profiling_index, range(0, len(profiling_desync_metadata)))], dtype=metadata_type)
	profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type)
	attack_metadata = np.array([(raw_plaintexts[n], raw_ciphertexts[n], raw_keys[n], raw_masks[n], attack_desync_metadata[k]) for n, k in zip(attack_index, range(0, len(attack_desync_metadata)))], dtype=metadata_type)
	attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type)

	out_file.flush()
	out_file.close()

# Our folders
ascad_data_folder = "ASCAD_data/"
ascad_databases_folder = ascad_data_folder + "ASCAD_databases/"

original_raw_traces_file = ascad_databases_folder + "ATMega8515_raw_traces.h5"
extract_traces(original_raw_traces_file, ascad_databases_folder + "ASCAD.h5")
extract_traces(original_raw_traces_file, ascad_databases_folder + "ASCAD_desync50.h5", attack_desync = 50)
extract_traces(original_raw_traces_file, ascad_databases_folder + "ASCAD_desync100.h5", attack_desync = 100)


