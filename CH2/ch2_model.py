import numpy as np

# مسیر فایل npy
path = input('endter dir x.npy : ')

# لود کردن
arr = np.load(path)

print("type:", type(arr))
print("shape:", arr.shape)
print("dtype:", arr.dtype)


