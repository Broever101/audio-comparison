import librosa
import os
import math
import os.path
import numpy as np
from numpy.linalg import norm
import numba
from numba import jit, cuda 
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

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

@cuda.jit
def dot_prod_kernel(A, B, dot_prod, norm_A, norm_B):
    
    """
    A/B = vectors
    dot_prod = vector to store the dot product of search and this
    norm_A = norm of A .. bruh
    norm_B = bleh
    """
    
    size = cuda.grid(1)
    if size < len(A):
        dot_prod[size] = A[size] * B[size]
        norm_A[size] = A[size] * A[size]
        norm_B[size] = B[size] * B[size]
        
        
def find_songs(source, target):
    
    for song in target:
        
        target_freq = song[1][0]
        target_rate = song[1][1]

        max_comparison = len(target_freq)
        threadsperblock = 128
        blockspergrid = math.ceil(max_comparison / threadsperblock)
        
        for another_song in source:
            
            source_freq = another_song[1][0]
            compare = len(source_freq) if len(source_freq) < len(target_freq) else len(target_freq)
            
            dot_prod = cuda.device_array((compare, 1))
            norm_A = cuda.device_array((compare, 1))
            norm_B = cuda.device_array((compare, 1))

            print("Matching {} with {}".format(song[0], another_song[0]))

            for i in tqdm(range(0, len(target_freq), 100)):
                target_glob = cuda.to_device(target_freq[i : i + compare])
                source_glob = cuda.to_device(source_freq[:compare])

                dot_prod_kernel[blockspergrid, threadsperblock](target_glob, source_glob,
                                                                dot_prod, norm_A, norm_B)
                
                norm_1 = norm_A.copy_to_host()
                norm_2 = norm_B.copy_to_host()
                dot_prod1 = dot_prod.copy_to_host()

                similarity = np.sum(dot_prod1)/math.sqrt(np.sum(norm_1) * np.sum(norm_2))
                
                if similarity > 0.5:
                    print("{} MATCHED WITH {}".format(song[0], another_song[0]))
                    break


find_songs(source, target)
 
# dummy = cuda.to_device(np.array([4, 5, 6, 7, 8]))
# dummy2 = cuda.to_device(np.array([1, 2, 3, 4, 5]))
# dot = cuda.device_array((5, 1))
# norm_A = cuda.device_array((5, 1))
# norm_B = cuda.device_array((5, 1))

# dot_prod_kernel[1, 16](dummy, dummy2, dot, norm_A, norm_B)
# #compute_norm[1, 16](dummy, norm_A)
# #compute_norm[1, 16](dummy2, norm_B)

# print("DOT: {}".format(sum(dot.copy_to_host())))
# print("NORM_A: {}".format(sum(norm_A.copy_to_host())))
# print("NORM_B: {}".format(sum(norm_B.copy_to_host())))
# print(norm([1,2,3,4,5]))
# print(norm([4,5,6,7,8]))

