import librosa
import os
import math
import os.path
import numpy as np
from numpy.linalg import norm
#import numba
#from numba import jit, cuda 
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import numba
    from numba import jit, cuda

class loadSongs():
        
    data_dict = {"target_data.npy": ["target", list()], 
                 "source_data.npy": ["source", list()]}
    
    def load_songs(self):
        
        for file in self.data_dict:
            
            print("Loading {} data".format(self.data_dict[file][0]))
            
            if os.path.exists(file):
                self.data_dict[file][1] = np.load(file, allow_pickle = True)
            
            else:
                for song in tqdm(os.listdir(self.data_dict[file][0])):
                    if song.endswith(".mp3"):
                        self.data_dict[file][1].append([song, 
                                    librosa.load(os.path.join(self.data_dict[file][0], song))])
                np.save(file, self.data_dict[file][1], allow_pickle = True)
        
        print("Done.")
        return self.data_dict["target_data.npy"][1], self.data_dict["source_data.npy"][1]
    
target, source = loadSongs().load_songs()

# X threads in a block; all blocks combined should make up the length of the search song
# so one thread in a block must compare upto X entries

# @cuda.jit
# def compute_norm(A, norm):

#     size = cuda.grid(1)
#     if size < len(A):
#         norm[size] = A[size]*A[size]

threadsperblock = 128

@cuda.jit
def dot_prod_kernel(A, B, dot_prod, norm_A, norm_B):
    
    """
    A/B = vectors
    dot_prod = vector to store the dot product of A and B
    norm_A = norm of A .. bruh
    norm_B = bleh
    """

    size = cuda.grid(1)
    if size < len(A):
        dot_prod[size] = A[size] * B[size]
        norm_A[size] = A[size] * A[size]
        norm_B[size] = B[size] * B[size]

# @cuda.jit
# def parallel_reduce(dot_prod, norm_A, norm_B):
#     """
#     dot_prod = vector of element-wise product of A and B
#     norm_A = vector of element-wise squares of A 
#     norm_B = vector of element-wise squares of B

#     The function will write the sum of each vector to their 0th index 
#     using parallel reduction.
#     """

#     sdot_prod = cuda.shared.array(shape = (threadsperblock,), dtype = numba.float64)
#     snorm_A = cuda.shared.array(shape = (threadsperblock,), dtype = numba.float64)
#     snorm_B = cuda.shared.array(shape = (threadsperblock,), dtype = numba.float64)

#     tid = cuda.grid(1)

#     sdot_prod[tid] = dot_prod[tid]
#     snorm_A[tid] = norm_A[tid]
#     snorm_B[tid] = norm_B[tid]

#     cuda.syncthreads()

#     s = cuda.blockDim.x/2
#     while s > 0:
#         if tid < s:
#             sdot_prod[tid] += sdot_prod[tid + s]
#             snorm_A[tid] += sdot_prod[tid + s]
#             snorm_B[tid] += snorm_B[tid + s]
#         cuda.syncthreads()

#         s >>= 1
    
#     if tid == 0:
#         dot_prod[cuda.blockIdx.x] = sdot_prod[0]
#         snorm_A[cuda.blockIdx.x] = snorm_A[0]
#         snorm_B[cuda.blockIdx.x] = snorm_B[0]


@cuda.reduce
def sum_reduce(a, b):
    return a + b

def find_songs(source, target):
    
    match_file = open("Matches.txt", "w")

    for song in target:
        
        target_freq = song[1][0]
        target_rate = song[1][1]

        max_comparison = len(target_freq)
        blockspergrid = math.ceil(max_comparison / threadsperblock)
        
        for another_song in source:
            
            source_freq = another_song[1][0]
            compare = len(source_freq) if len(source_freq) < len(target_freq) else len(target_freq)
            
            dot_prod = cuda.device_array((compare,))
            norm_A = cuda.device_array((compare,))
            norm_B = cuda.device_array((compare,))

            print("Matching {} with {}".format(song[0], another_song[0]))
            
            for i in tqdm(range(0, len(target_freq), math.ceil(target_rate))):
                target_glob = cuda.to_device(target_freq[i : i + compare])
                source_glob = cuda.to_device(source_freq[:compare])

                dot_prod_kernel[blockspergrid, threadsperblock](target_glob, source_glob,
                                                                dot_prod, norm_A, norm_B)
                dot = sum_reduce(dot_prod)
                normA = sum_reduce(norm_A)
                normB = sum_reduce(norm_B)

                similarity = dot/math.sqrt(normA * normB)
                
                if similarity > 0.01:
                    offset = i/target_rate 
                    match_file.write("{0} , {1} , {2:5.2f}, {3:5.2f} \n".format(song[0], another_song[0], offset, compare/target_rate))
                    print("{} MATCHED WITH {}".format(song[0], another_song[0]))
                    break
    
    match_file.close()


find_songs(source, target)
 
# dummy = np.array([1, 2, 3, 4, 5])
# dummy1 = np.array([4, 5, 6, 7, 8])
# dummy2 = np.array([1, 1, 1, 1, 1])
# dummy_glob = cuda.to_device(dummy)
# dummy1_glob = cuda.to_device(dummy1)
# dummy2_glob = cuda.to_device(dummy2)

# dot = sum_reduce(dummy_glob)
# norm = sum_reduce(dummy1_glob)
# norm1 = sum_reduce(dummy2_glob)

# print("DOT: {}".format(dot))
# print("NORM_A: {}".format(norm))
# print("NORM_B: {}".format(norm1))
# dot = cuda.device_array((5,))
# norm_A = cuda.device_array((5,))
# norm_B = cuda.device_array((5,))

#dot_prod_kernel[1, 16](dummy_glob, dummy1_glob, dot, norm_A, norm_B)
# dot = dot.copy_to_host()
# norm_A = norm_A.copy_to_host()
# norm_B = norm_B.copy_to_host()
# dot_p = cuda.to_device(dot)
# norm_Ap = cuda.to_device(norm_A)
# norm_Bp = cuda.to_device(norm_B)
#parallel_reduce[1, 16](dummy_glob, dummy1_glob, dummy2_glob)
#print(sum_reduce(dummy2_glob))
#compute_norm[1, 16](dummy, norm_A)
#compute_norm[1, 16](dummy2, norm_B)
# print(dot)
# print("DOT: {}".format(dot.copy_to_host()))
# print("NORM_A: {}".format(norm_A.copy_to_host()))
# print("NORM_B: {}".format(norm_B.copy_to_host()))
# print(norm([1,2,3,4,5]))
# print(norm([4,5,6,7,8]))

# print(target[0][1][0][1653750:1653800])
# print(source[1][1][0][:50])