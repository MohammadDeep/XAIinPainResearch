import numpy as np

# مسیر فایل npy
path = './datasets/painmonit/np-dataset/'
x = 'X.npy'
subjects = 'subjects.npy'
y_covas = 'y_covas.npy'
y_heater = 'y_heater.npy'
def read_File(path):
    # لود کردن
    arr = np.load(path)

    print("type:", type(arr))
    print("shape:", arr.shape)
    print("dtype:", arr.dtype)

for i in [x, subjects, y_covas, y_heater]:
    print('==================================')
    print( i )
    read_File(path + i)


