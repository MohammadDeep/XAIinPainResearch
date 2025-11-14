import numpy as np
from aeon.classification.hybrid import HIVECOTEV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
# مسیر فایل npy
path = './datasets/painmonit/np-dataset/'
x_file = 'X.npy'
subjects_file = 'subjects.npy'
y_covas_file = 'y_covas.npy'
y_heater_file = 'y_heater.npy'

def read_File(path):
    arr = np.load(path)
    print("type:", type(arr))
    print("shape:", arr.shape)
    print("dtype:", arr.dtype)
    return arr

# ===================== X =====================
print('==================================')
print(x_file)
X_raw = read_File(path + x_file)

print('----------------------------')
print('creating data for hc2:')
# حذف بعد اضافه‌ی آخر: (N, T, C, 1) -> (N, T, C)
X_no_last = np.squeeze(X_raw, axis=-1)
# transpose به فرم (N, C, T) که hc2 می‌خواهد
X_ch2 = np.transpose(X_no_last, (0, 2, 1))
print("X_ch2 shape for HC2:", X_ch2.shape)

# (اختیاری) تبدیل به float32 برای مصرف رم کمتر
X_ch2 = X_ch2.astype(np.float32)

# ===================== y_covas =====================
print('==================================')
print(y_covas_file)
y = read_File(path + y_covas_file)

print('----------------------------')
print('creating y_covas for hc2:')
# (N, 5) one-hot -> (N,) لیبل عددی 0..4
y_covas_ch2 = np.argmax(y, axis=1).astype(int)
print("y_covas_ch2:", y_covas_ch2.shape, np.unique(y_covas_ch2))

# ===================== y_heater =====================
print('==================================')
print(y_heater_file)
y = read_File(path + y_heater_file)

print('----------------------------')
print('creating y_heater for hc2:')
# (N, 6) one-hot -> (N,) لیبل عددی 0..5
y_heater_ch2 = np.argmax(y, axis=1).astype(int)
print("y_heater_ch2:", y_heater_ch2.shape, np.unique(y_heater_ch2))

# ===================== subjects =====================
print('==================================')
print(subjects_file)
subjects_ch2 = read_File(path + subjects_file)

# ===================== انتخاب مسئله: covas یا heater =====================
# اینجا انتخاب می‌کنی hc2 را روی کدام لیبل اجرا کنی:
#   y_target = y_covas_ch2   # ۵ کلاسه (مثلا covas)
#   y_target = y_heater_ch2  # ۶ کلاسه (heater)
y_target = y_covas_ch2

# چک سازگاری تعداد نمونه‌ها
assert X_ch2.shape[0] == y_target.shape[0] == subjects_ch2.shape[0], "N mismatch!"

print("\nFinal shapes for hc2:")
print("X_ch2:", X_ch2.shape)
print("y_target:", y_target.shape)
print("subjects_ch2:", subjects_ch2.shape)

# ===================== ۲. train/test split ساده =====================
print("\n================ RANDOM SPLIT EXPERIMENT ================")

X_train, X_test, y_train, y_test = train_test_split(
    X_ch2,
    y_target,
    test_size=0.2,
    stratify=y_target,
    random_state=42
)

print("Train size:", X_train.shape[0], " Test size:", X_test.shape[0])

hc2 = HIVECOTEV2(
    time_limit_in_minutes=10,   # hc2 سنگین است؛ برای تست می‌تونی کم/زیادش کنی
    n_jobs=-1,                  # استفاده از همه هسته‌های CPU
    random_state=0,
    verbose=1
)

print("Fitting HC2 on random split...")
t0 = time.time()
hc2.fit(X_train, y_train)
t1 = time.time()
print(f"Fit time: {(t1 - t0):.1f} seconds")


print("Predicting...")
y_pred = hc2.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Random split accuracy:", acc)

# ===================== ۳. LOSO بر اساس subjects =====================
print("\n================ LOSO EXPERIMENT (by subject) ================")

unique_subjects = np.unique(subjects_ch2)
print("Unique subjects:", unique_subjects)

all_preds = []
all_trues = []

for s in unique_subjects:
    print(f"\n=== LOSO fold: subject = {s} ===")
    test_mask = (subjects_ch2 == s)
    train_mask = ~test_mask

    X_train, X_test = X_ch2[train_mask], X_ch2[test_mask]
    y_train, y_test = y_target[train_mask], y_target[test_mask]

    print("  Train size:", X_train.shape[0], " Test size:", X_test.shape[0])

    # اگر تو train فقط یک کلاس باشد، بعضی مدل‌ها مشکل پیدا می‌کنند
    if len(np.unique(y_train)) < 2:
        print("  [WARN] Train fold has only one class; skipping this subject.")
        continue

    hc2 = HIVECOTEV2(
        time_limit_in_minutes=10,
        n_jobs=-1,
        random_state=0,
        verbose=1
    )

    print("  Fitting HC2...")
    
    t0 = time.time()
    hc2.fit(X_train, y_train)
    t1 = time.time()
    print(f"Fit time: {(t1 - t0):.1f} seconds")
    

    print("  Predicting for this subject...")
    y_pred = hc2.predict(X_test)

    all_preds.append(y_pred)
    all_trues.append(y_test)

# اگر همه foldها اسکپ نشده باشند
if len(all_preds) > 0:
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)

    loso_acc = accuracy_score(all_trues, all_preds)
    print("\n=============================")
    print("Overall LOSO Accuracy:", loso_acc)
    print("=============================")
else:
    print("No valid LOSO folds (check label distribution per subject).")
