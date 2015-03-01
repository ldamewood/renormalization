from sklearn.utils.murmurhash cimport murmurhash3_bytes_s32

def murmur3(x):
    return murmurhash3_bytes_s32(x, 0)