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
                    match_file.write("{0} , {1} , {2:5.2f}, {3:5.2f} \n".format(song[0], 
                                            another_song[0], offset, compare/target_rate))

                    print("{} MATCHED WITH {}".format(song[0], another_song[0]))
                    break
    
    match_file.close()


find_songs(source, target)
 
