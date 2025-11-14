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

for i in [y_covas,y_heater ]:
    print('==================================')
    print( i )
    y = read_File(path + i)

    print('----------------------------')
    print('creating data for ch2:')
    # برای covas (5 کلاس)
    y_covas_labels = np.argmax(y, axis=1).astype(int)   # شکل: (2495,)

    # برای heater (6 کلاس)
    y_heater_labels = np.argmax(y_heater, axis=1).astype(int) # شکل: (2495,)

    print("y_covas_labels:", y_covas_labels.shape, np.unique(y_covas_labels))
    

