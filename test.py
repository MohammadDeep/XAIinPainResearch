import numpy as np
import os
import pandas as pd

# ===================== پارامترهای کلی =====================
# parametr for model ch2 
time_limit_in_minutes = 0
n_jobs = -1

'''
for test code
'''
debug_mode = False
max_subjects_debug = 3          # حداکثر چند سوژه برای تست
max_train_samples_debug = 20    # حداکثر چند نمونه train برای هر سوژه
max_test_samples_debug = 5      # حداکثر چند نمونه test برای هر سوژه

rng = np.random.default_rng(0)   # برای انتخاب تصادفی تکرارپذیر
if debug_mode:
    time_limit_in_minutes = 1

# مسیر فایل npy
save_modeles_path = "./CH2/modeles"

results_dir = "./CH2/result"

# اگر پوشه وجود نداشت، ساخته می‌شود. اگر وجود داشته باشد، کاری نمی‌کند.
os.makedirs(results_dir, exist_ok=True)
results_csv = os.path.join(results_dir, "ch2_result.csv")

path = './datasets/painmonit/np-dataset/'
x_file = 'X.npy'
subjects_file = 'subjects.npy'
y_covas_file = 'y_covas.npy'
y_heater_file = 'y_heater.npy'


def read_File(path_):
    """خواندن و چاپ اطلاعات اولیه یک فایل npy"""
    arr = np.load(path_)
    print("type:", type(arr))
    print("shape:", arr.shape)
    print("dtype:", arr.dtype)
    return arr


def print_class_stats(label_name, y_labels, subjects):
    """
    چاپ آمار تعداد نمونه‌ها برای هر کلاس:
      - به صورت کلی (global)
      - برای هر subject به تفکیک کلاس‌ها

    label_name: فقط برای چاپ (مثلا "COVAS" یا "HEATER")
    y_labels:  آرایه لیبل‌های عددی (N,) با مقادیر 0..K-1
    subjects:  آرایه شناسه سوژه‌ها با شکل (N,)
    """
    # چک سازگاری طول‌ها
    assert y_labels.shape[0] == subjects.shape[0], \
        "Length mismatch between labels and subjects!"

    # تعیین تعداد کلاس‌ها از روی بیشترین لیبل
    n_classes = int(y_labels.max()) + 1

    print(f"\n========== STATS FOR {label_name} ==========")

    # -------- آمار کلی (global) --------
    print("Global class counts:")
    global_counts = np.bincount(y_labels, minlength=n_classes)
    for cls in range(n_classes):
        print(f"  Class {cls}: {global_counts[cls]} samples")

    # -------- آمار به تفکیک subject --------
    print("\nPer-subject class counts:")
    unique_subjects = np.unique(subjects)

    # اگر حالت دیباگ روشن است، تعداد سوژه‌ها را محدود کن
    if debug_mode:
        unique_subjects = unique_subjects[:max_subjects_debug]
        print(f"DEBUG: using only subjects: {unique_subjects}")

    for subj in unique_subjects:
        mask = (subjects == subj)
        y_sub = y_labels[mask]

        if y_sub.size == 0:
            print(f"Subject {subj}: no samples")
            continue

        counts_sub = np.bincount(y_sub, minlength=n_classes)

        print(f"\nSubject {subj}: total {mask.sum()} samples")
        for cls in range(n_classes):
            print(f"  Class {cls}: {counts_sub[cls]} samples")
    print("=" * 60)


# ===================== X =====================
print('==================================')
print(x_file)
X_raw = read_File(os.path.join(path, x_file))

print('----------------------------')
print('creating data for hc2:')

# حذف بعد اضافه‌ی آخر: (N, T, C, 1) -> (N, T, C)
X_no_last = np.squeeze(X_raw, axis=-1)

# transpose به فرم (N, C, T) که hc2 می‌خواهد
X_ch2 = np.transpose(X_no_last, (0, 2, 1))
print("X_ch2 shape for HC2:", X_ch2.shape)

# در حالت دیباگ، طول زمانی سیگنال را نصف کن (هر 2 نمونه یکی)
if debug_mode:
    X_ch2 = X_ch2[:, :, ::2]
    print("DEBUG: X_ch2 shape after temporal downsample:", X_ch2.shape)

# (اختیاری) تبدیل به float32 برای مصرف رم کمتر
X_ch2 = X_ch2.astype(np.float32)

# ===================== y_covas =====================
print('==================================')
print(y_covas_file)
y = read_File(os.path.join(path, y_covas_file))

print('----------------------------')
print('creating y_covas for hc2:')
# (N, 5) one-hot -> (N,) لیبل عددی 0..4
y_covas_ch2 = np.argmax(y, axis=1).astype(int)
print("y_covas_ch2:", y_covas_ch2.shape, np.unique(y_covas_ch2))

# ===================== y_heater =====================
print('==================================')
print(y_heater_file)
y = read_File(os.path.join(path, y_heater_file))

print('----------------------------')
print('creating y_heater for hc2:')
# (N, 6) one-hot -> (N,) لیبل عددی 0..5
y_heater_ch2 = np.argmax(y, axis=1).astype(int)
print("y_heater_ch2:", y_heater_ch2.shape, np.unique(y_heater_ch2))

# ===================== subjects =====================
print('==================================')
print(subjects_file)
subjects_ch2 = read_File(os.path.join(path, subjects_file))

# ===================== آمار کلاس‌ها برای هر دو نوع لیبل =====================
print_class_stats("COVAS",  y_covas_ch2,  subjects_ch2)
print_class_stats("HEATER", y_heater_ch2, subjects_ch2)

# ===================== انتخاب مسئله: covas یا heater برای HC2 =====================
# اینجا انتخاب می‌کنی hc2 را روی کدام لیبل اجرا کنی:
#   y_target = y_covas_ch2   # ۵ کلاسه (covas)
#   y_target = y_heater_ch2  # ۶ کلاسه (heater)
y_target = y_covas_ch2

# چک سازگاری تعداد نمونه‌ها
assert X_ch2.shape[0] == y_target.shape[0] == subjects_ch2.shape[0], "N mismatch!"

print("\nFinal shapes for hc2:")
print("X_ch2:", X_ch2.shape)
print("y_target:", y_target.shape)
print("subjects_ch2:", subjects_ch2.shape)

# ===================== ۳. LOSO بر اساس subjects (برای ادامه‌ی کار HC2) =====================
print("\n================ LOSO EXPERIMENT (by subject) ================")

unique_subjects = np.unique(subjects_ch2)
print("Unique subjects:", unique_subjects)

# در حالت دیباگ، فقط چند سوژه اول را استفاده کن
if debug_mode:
    unique_subjects = unique_subjects[:max_subjects_debug]
    print("DEBUG: using only subjects:", unique_subjects)

# از اینجا به بعد می‌توانی حلقه‌ی LOSO و آموزش HC2 را بنویسی
# for test_subj in unique_subjects:
#     ...
