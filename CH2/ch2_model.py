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
    return arr

for i in [x]:
    print('==================================')
    print( i )
    X_raw = read_File(path + i)

    print('----------------------------')
    print('creating data for ch2:')
    # 2) حذف بعد اضافه‌ی آخر
    X_no_last = np.squeeze(X_raw, axis=-1)
    X = np.transpose(X_no_last, (0, 2, 1)) 
    print("X shape for HC2:", X.shape)


