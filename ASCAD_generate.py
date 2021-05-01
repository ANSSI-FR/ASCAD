import os
import sys
import h5py
import numpy as np
import random
from tqdm import tqdm

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

# G auxiliary function that is used to generate the permution of indices
G = np.array([0x0C, 0x05, 0x06, 0x0b, 0x09, 0x00, 0x0a, 0x0d, 0x03, 0x0e, 0x0f, 0x08, 0x04, 0x07, 0x01, 0x02])

# The permution function on the 16 indices i. The function is defined from the masks m0, m1, m2, and m3.
def permIndices(i,m0,m1,m2,m3):
	x0,x1,x2,x3 = m0&0x0f, m1&0x0f, m2&0x0f, m3&0x0f
	return G[G[G[G[(15-i)^x0]^x1]^x2]^x3]

# Two Tables to process a field multplication over GF(256): a*b = alog (log(a) + log(b) mod 255)
log_table=[ 0, 0, 25, 1, 50, 2, 26, 198, 75, 199, 27, 104, 51, 238, 223, 3,
	100, 4, 224, 14, 52, 141, 129, 239, 76, 113, 8, 200, 248, 105, 28, 193,
	125, 194, 29, 181, 249, 185, 39, 106, 77, 228, 166, 114, 154, 201, 9, 120,
	101, 47, 138, 5, 33, 15, 225, 36, 18, 240, 130, 69, 53, 147, 218, 142,
	150, 143, 219, 189, 54, 208, 206, 148, 19, 92, 210, 241, 64, 70, 131, 56,
	102, 221, 253, 48, 191, 6, 139, 98, 179, 37, 226, 152, 34, 136, 145, 16,
	126, 110, 72, 195, 163, 182, 30, 66, 58, 107, 40, 84, 250, 133, 61, 186,
	43, 121, 10, 21, 155, 159, 94, 202, 78, 212, 172, 229, 243, 115, 167, 87,
	175, 88, 168, 80, 244, 234, 214, 116, 79, 174, 233, 213, 231, 230, 173, 232,
	44, 215, 117, 122, 235, 22, 11, 245, 89, 203, 95, 176, 156, 169, 81, 160,
	127, 12, 246, 111, 23, 196, 73, 236, 216, 67, 31, 45, 164, 118, 123, 183,
	204, 187, 62, 90, 251, 96, 177, 134, 59, 82, 161, 108, 170, 85, 41, 157,
	151, 178, 135, 144, 97, 190, 220, 252, 188, 149, 207, 205, 55, 63, 91, 209,
	83, 57, 132, 60, 65, 162, 109, 71, 20, 42, 158, 93, 86, 242, 211, 171,
	68, 17, 146, 217, 35, 32, 46, 137, 180, 124, 184, 38, 119, 153, 227, 165,
	103, 74, 237, 222, 197, 49, 254, 24, 13, 99, 140, 128, 192, 247, 112, 7 ]

alog_table =[1, 3, 5, 15, 17, 51, 85, 255, 26, 46, 114, 150, 161, 248, 19, 53,
	95, 225, 56, 72, 216, 115, 149, 164, 247, 2, 6, 10, 30, 34, 102, 170,
	229, 52, 92, 228, 55, 89, 235, 38, 106, 190, 217, 112, 144, 171, 230, 49,
	83, 245, 4, 12, 20, 60, 68, 204, 79, 209, 104, 184, 211, 110, 178, 205,
	76, 212, 103, 169, 224, 59, 77, 215, 98, 166, 241, 8, 24, 40, 120, 136,
	131, 158, 185, 208, 107, 189, 220, 127, 129, 152, 179, 206, 73, 219, 118, 154,
	181, 196, 87, 249, 16, 48, 80, 240, 11, 29, 39, 105, 187, 214, 97, 163,
	254, 25, 43, 125, 135, 146, 173, 236, 47, 113, 147, 174, 233, 32, 96, 160,
	251, 22, 58, 78, 210, 109, 183, 194, 93, 231, 50, 86, 250, 21, 63, 65,
	195, 94, 226, 61, 71, 201, 64, 192, 91, 237, 44, 116, 156, 191, 218, 117,
	159, 186, 213, 100, 172, 239, 42, 126, 130, 157, 188, 223, 122, 142, 137, 128,
	155, 182, 193, 88, 232, 35, 101, 175, 234, 37, 111, 177, 200, 67, 197, 84,
	252, 31, 33, 99, 165, 244, 7, 9, 27, 45, 119, 153, 176, 203, 70, 202,
	69, 207, 74, 222, 121, 139, 134, 145, 168, 227, 62, 66, 198, 81, 243, 14,
	18, 54, 90, 238, 41, 123, 141, 140, 143, 138, 133, 148, 167, 242, 13, 23,
	57, 75, 221, 124, 132, 151, 162, 253, 28, 36, 108, 180, 199, 82, 246, 1 ]

# Multiplication function in GF(2^8)
def multGF256(a,b):
	if (a==0) or (b==0):
		return 0
	else:
		return alog_table[(log_table[a]+log_table[b]) %255]


# The single labelization function used for ASCADv1:
# It is as simple as the computation of the result of Sbox(p[2] + k[2]) (see the White Paper)
# Note: you can of course adapt the labelization here (say if you want to attack the first byte Sbox(p[0] + k[0])
# or if you want to attack another round of the algorithm).
def labelize(plaintexts, keys):
	return np.uint8(AES_Sbox[plaintexts[:, 2] ^ keys[:, 2]])

# The multi labelization function used for ASCADv2:
# It computes :
#	- the multiplative mask value alpha which correpond to r_m
#	- the additive mask value beta wich correspnd to rout
#	- the result of maskedSbox(p[i]^k[i]) for each i in [0..15]
#	- the result of maskedSbox(p[permInd[i]]^k[permInd[i]]) for each i in [0..15]
#	- the permuted indice value permInd[i] for each i in [0..15]
def multilabelize(metadata):
	def mult_sbox_mask_f(data, target_byte):
		ind = permIndices(target_byte,data["masks"][0],data["masks"][1],data["masks"][2],data["masks"][3])
		alpha = data["masks"][18]
		beta = data["masks"][17]
		S = AES_Sbox[data["plaintext"][ind]^data["key"][ind]]
		return multGF256(alpha,S)^beta

	def mult_sbox_mask_with_perm_f(data, target_byte):
		alpha = data["masks"][18]
		beta = data["masks"][17]
		S = AES_Sbox[data["plaintext"][target_byte]^data["key"][target_byte]]
		return multGF256(alpha,S)^beta

	def permind_f(data, target_byte):
		ind = permIndices(target_byte,data["masks"][0],data["masks"][1],data["masks"][2],data["masks"][3])
		return ind

	def alpha_mask_f(data):
		alpha = data["masks"][18]
		return alpha

	def beta_mask_f(data):
		beta = data["masks"][17]
		return beta

	y_alpha = np.array([alpha_mask_f(d) for d in metadata])
	y_beta = np.array([beta_mask_f(d) for d in metadata])
	y_sbox = []
	y_sbox_with_perm = []
	y_permind = []
	for i in range(16):
		y_sbox.append(np.array([mult_sbox_mask_f(d, i) for d in metadata]))
		y_sbox_with_perm.append(np.array([mult_sbox_mask_with_perm_f(d, i) for d in metadata]))
		y_permind.append(np.array([permind_f(d, i) for d in metadata]))
	y_sbox = np.transpose(y_sbox)
	y_sbox_with_perm = np.transpose(y_sbox_with_perm)
	y_permind = np.transpose(y_permind)
	multilabel_type = np.dtype([("alpha_mask", np.uint8, (1,)),
				  ("beta_mask", np.uint8, (1,)),
				  ("sbox_masked", np.uint8, (16,)),
				 ("sbox_masked_with_perm", np.uint8, (16,)),
				  ("perm_index", np.uint8, (16,))])
	multilabel = np.array([(y_alpha[n], y_beta[n], y_sbox[n], y_sbox_with_perm[n], y_permind[n]) for n in range(len(metadata))],  dtype=multilabel_type)
	return multilabel

# TODO: sanity checks on the parameters
# This function extract the POIs of the traces contained in a single file and labelized them.
def extract_traces(traces_file, labeled_traces_file, profiling_index, attack_index, target_points, profiling_desync=0, attack_desync=0, multilabel=0):
	print("Begin extraction")
	traces_file = os.path.normpath(traces_file)
	check_file_exists(traces_file)
	check_file_exists(os.path.dirname(labeled_traces_file))
	# Open the raw traces HDF5 for reading
	try:
		in_file	 = h5py.File(traces_file, "r")
	except:
		print("Error2: can't open HDF5 file '%s' for reading (it might be malformed) ..." % traces_file)
		sys.exit(-1)

	raw_traces = in_file['traces']
	raw_data = in_file['metadata']

	raw_plaintexts = raw_data['plaintext']
	raw_keys = raw_data['key']
	raw_masks = raw_data['masks']

	#TODO: deal with the case where "ciphertext" entry is there
	# Extract a larger set of points to handle desynchronization
	min_target_point = min(target_points)
	max_target_point = max(target_points)

	target_points = np.array(target_points)
	#we look for consecutive values in the target points, which would allow for a huge speed increase in the extraction
	diff = np.ediff1d(target_points)
	consecutive_indices = np.split(target_points,np.where(diff !=1)[0]+1)

	#######################################
	print("Processing profiling traces...")
	#######################################
	raw_traces_profiling = np.zeros([len(profiling_index), len(target_points)], raw_traces.dtype)
	profiling_desync_metadata = np.zeros(len(profiling_index), np.uint32)
	curr_trace = 0
	for trace in tqdm(profiling_index):
		if attack_desync !=0 or profiling_desync !=0:
			r_desync = random.randint(0, profiling_desync)
		else:
			r_desync = 0
		profiling_desync_metadata[curr_trace] = r_desync
		curr_point = 0
		for cons_chunk in consecutive_indices:
			raw_traces_profiling[curr_trace,curr_point:curr_point+len(cons_chunk)] = raw_traces[trace,cons_chunk[0]+r_desync:cons_chunk[len(cons_chunk)-1]+r_desync+1]
			curr_point += len(cons_chunk)
		curr_trace += 1

	####################################
	print("Processing attack traces...")
	####################################
	raw_traces_attack = np.zeros([len(attack_index), len(target_points)], raw_traces.dtype)
	attack_desync_metadata = np.zeros(len(attack_index))
	curr_trace = 0
	for trace in tqdm(attack_index):
		if attack_desync !=0 or profiling_desync !=0:
			r_desync = random.randint(0,attack_desync)
		else:
			r_desync = 0
		attack_desync_metadata[curr_trace] = r_desync
		curr_point = 0
		for cons_chunk in consecutive_indices:
			raw_traces_attack[curr_trace, curr_point:curr_point+len(cons_chunk)] = raw_traces[trace, cons_chunk[0]+r_desync:cons_chunk[len(cons_chunk)-1]+r_desync+1]
			curr_point += len(cons_chunk)
		curr_trace += 1

	############################
	print("Computing labels...")
	############################
	# Compute our labels
	if multilabel == 1:
		labels_profiling = multilabelize(raw_data[profiling_index])
		labels_attack  = multilabelize(raw_data[attack_index])
	else:
		labels_profiling = labelize(raw_plaintexts[profiling_index], raw_keys[profiling_index])
		labels_attack  = labelize(raw_plaintexts[attack_index], raw_keys[attack_index])

	print("Creating output_file...")
	# Open the output labeled file for writing
	try:
		out_file = h5py.File(labeled_traces_file, "w")
	except:
		print("Error3: can't open HDF5 file '%s' for writing ..." % labeled_traces_file)
		sys.exit(-1)
	# Create our HDF5 hierarchy in the output file:
	#	- Profilinging traces with their labels
	#	- Attack traces with their labels
	profiling_traces_group = out_file.create_group("Profiling_traces")
	attack_traces_group = out_file.create_group("Attack_traces")
	# Datasets in the groups
	profiling_traces_group.create_dataset(name="traces", data=raw_traces_profiling, dtype=raw_traces_profiling.dtype)
	attack_traces_group.create_dataset(name="traces", data=raw_traces_attack, dtype=raw_traces_attack.dtype)
	# Labels in the groups
	profiling_traces_group.create_dataset(name="labels", data=labels_profiling, dtype=labels_profiling.dtype)
	attack_traces_group.create_dataset(name="labels", data=labels_attack, dtype=labels_attack.dtype)
	#TODO: deal with the case where "ciphertext" entry is there
	# Put the metadata (plaintexts, keys, ...) so that one can check the key rank
	metadata_type = np.dtype([("plaintext", raw_plaintexts.dtype, (len(raw_plaintexts[0]),)),
				  ("key", raw_keys.dtype, (len(raw_keys[0]),)),
				  ("masks", raw_masks.dtype, (len(raw_masks[0]),)),
				  ("desync", np.uint32, (1,)),
				 ])
	profiling_metadata = np.array([(raw_plaintexts[n], raw_keys[n], raw_masks[n], profiling_desync_metadata[k]) for n, k  in zip(profiling_index, range(0, len(profiling_desync_metadata)))], dtype=metadata_type)
	profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type)
	attack_metadata = np.array([(raw_plaintexts[n], raw_keys[n], raw_masks[n], attack_desync_metadata[k]) for n, k in zip(attack_index, range(0, len(attack_desync_metadata)))], dtype=metadata_type)
	attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type)

	out_file.flush()
	out_file.close()
	in_file.close()

# This function concatenates two h5 groups f1_gp and f2_gp in a single group fout_gp. It assumes that the groups are formed of arrays of the same name.
# For each entry name k in f1_gp and f2_gp, an array that results in the concatenation of the arrays f1_gp[k] and f2_gp[k] is created in fout_gp.
# The arrays are concatenated along their first axis.
def h5_concatenate_group(f1_gp, f2_gp, fout_gp):
	keys = f1_gp.keys()
	dtypes = {}
	shapes = {}
	for k in keys:
		dtypes[k] = f1_gp[k].dtype
		shapes[k] = (f1_gp[k].shape[0] + f2_gp[k].shape[0], ) + f1_gp[k].shape[1:]
	for k in keys:
		fout_gp.create_dataset(k, shapes[k], dtype=dtypes[k])
		for i in tqdm(range(shapes[k][0])):
			if i<f1_gp[k].shape[0]:
				fout_gp[k][i] = f1_gp[k][i]
			else:
				fout_gp[k][i] = f2_gp[k][i-f1_gp[k].shape[0]]

# This function concatenates the arrays contained in the h5 files file1 and file2, resulting in a ew h5 file fileout.
# The arrays are concatenated along their first axis.
def h5_concatenate_file(file1, file2, fileout):
	f1 = h5py.File(file1, "r")
	f2 = h5py.File(file2, "r")
	fout = h5py.File(fileout, "w")

	print("Concatenation Profiling_traces")
	fout_profile_gp = fout.create_group("Profiling_traces")
	h5_concatenate_group(f1["Profiling_traces"], f2["Profiling_traces"], fout_profile_gp)

	print("Concatenation Attack_traces")
	fout_attack_gp = fout.create_group("Attack_traces")
	h5_concatenate_group(f1["Attack_traces"], f2["Attack_traces"], fout_attack_gp)

	fout.flush()
	fout.close()
	f1.close()
	f2.close()

# This function concatenates a list of files.
def h5_concatenate_file_list(file_list, fileout):
	if (len(file_list)<2):
		print("Error: traces_file_list shall contain at least two files")
		sys.exit(-1)
	fileout_tmp= fileout+ ".tmp"
	fileout_bis_tmp= fileout+ "_bis.tmp"
	print("Concatenation 1/{}".format(len(file_list)-1))
	h5_concatenate_file(file_list[0], file_list[1], fileout_tmp)
	for i,file_i in enumerate(file_list[2:]):
		print("Concatenation {}/{}".format(i+2, len(file_list)-1))
		h5_concatenate_file(fileout_tmp, file_i, fileout_bis_tmp)
		os.rename(fileout_bis_tmp, fileout_tmp)
	os.rename(fileout_tmp, fileout)

# This function extracts the POIs of the traces contained in different h5 files and labelized them.
# A temporary labelized file is created for each file of the input list.
# Then the temporary labelized files are concatenated to form a single file.
def extract_multiple_files(traces_files_list, labeled_traces_file, profiling_index, attack_index, target_points, profiling_desync=0, attack_desync=0, multilabel=0):
	last_window_offset = 0
	labeled_traces_files_list = [labeled_traces_file + "_part_{}.tmp".format(i) for i in range(len(traces_files_list))]
	for i, traces_file_i in enumerate(traces_files_list):
		try:
			in_file_i	 = h5py.File(traces_file_i, "r")
		except:
			print("Error1: can't open HDF5 file '%s' for reading (it might be malformed) ..." % traces_file_i)
			sys.exit(-1)
		l = len(in_file_i['traces'])
		in_file_i.close()
		window_i = np.arange(last_window_offset, last_window_offset+l)
		labeled_traces_file_i = labeled_traces_files_list[i]
		profiling_intersect_index_i = np.intersect1d(profiling_index,window_i)
		profiling_index_i = [x-last_window_offset for x in profiling_intersect_index_i]
		attack_intersect_index_i = np.intersect1d(attack_index,window_i)
		attack_index_i = [x-last_window_offset for x in attack_intersect_index_i]
		print("Extraction of file {}".format(traces_file_i))
		extract_traces(traces_file_i, labeled_traces_file_i, profiling_index_i, attack_index_i, target_points, profiling_desync, attack_desync, multilabel)
		last_window_offset += l
	h5_concatenate_file_list(labeled_traces_files_list, labeled_traces_file)
	for labeled_traces_file in labeled_traces_files_list:
		os.remove(labeled_traces_file)


def read_parameters_file(param_filename):
	#read parameters for the extract_traces function from given filename
	#TODO: sanity checks on parameters
	param_file = open(param_filename,"r")

	#FIXME: replace eval() by ast.linear_eval()
	my_parameters= eval(param_file.read())

	files_splitted = 0
	traces_file = ""
	traces_files_list = []
	if ("files_splitted" in my_parameters):
		files_splitted = my_parameters["files_splitted"]
		if ("traces_files_list" not in my_parameters):
			print("Error: traces_files_list parameter must be defined when files_splitted option is activated ...")
			sys.exit(-1)
		traces_files_list = my_parameters["traces_files_list"]
	if ("traces_file" in my_parameters):
		traces_file = my_parameters["traces_file"]
	multilabel = 0
	if ("multilabel" in my_parameters):
		multilabel = my_parameters["multilabel"]
	labeled_traces_file = my_parameters["labeled_traces_file"]
	profiling_index = my_parameters["profiling_index"]
	attack_index = my_parameters["attack_index"]
	target_points = my_parameters["target_points"]
	profiling_desync = my_parameters["profiling_desync"]
	attack_desync = my_parameters["attack_desync"]

	param_file.close()

	return files_splitted, traces_file, traces_files_list, labeled_traces_file, profiling_index, attack_index, target_points, profiling_desync, attack_desync, multilabel


if __name__ == "__main__":
	if len(sys.argv)!=2:
		#default parameters values
		ascad_data_folder = "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/"
		ascad_databases_folder = ascad_data_folder + "ASCAD_databases/"

		original_raw_traces_file = ascad_databases_folder + "ATMega8515_raw_traces.h5"

		profiling_index = [n for n in range(0, 50000)]
		attack_index = [n for n in range(50000, 60000)]
		target_points=[n for n in range(45400, 46100)]
		profiling_desync=0
		attack_desync=0
		extract_traces(original_raw_traces_file, ascad_databases_folder + "ASCAD.h5"		  , profiling_index, attack_index, target_points, profiling_desync=0, attack_desync=0)
		extract_traces(original_raw_traces_file, ascad_databases_folder + "ASCAD_desync50.h5" , profiling_index, attack_index, target_points, profiling_desync=0, attack_desync = 50)
		extract_traces(original_raw_traces_file, ascad_databases_folder + "ASCAD_desync100.h5", profiling_index, attack_index, target_points, profiling_desync=0, attack_desync = 100)

	else:
		#get parameters from user input
		files_splitted, traces_file, traces_files_list, labeled_traces_file, profiling_index, attack_index, target_points, profiling_desync, attack_desync, multilabel = read_parameters_file(sys.argv[1])

		#execute the extraction function
		if (files_splitted !=1 ):
			extract_traces(traces_file, labeled_traces_file, profiling_index, attack_index, target_points, profiling_desync, attack_desync, multilabel)
		else:
			extract_multiple_files(traces_files_list, labeled_traces_file, profiling_index, attack_index, target_points, profiling_desync, attack_desync, multilabel)

