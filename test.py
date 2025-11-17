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


def make_class_counts_df(y_labels, subjects, label_name=None):
    """
    ساخت یک DataFrame که:
      - سطرها: subject ها + یک سطر TOTAL
      - ستون‌ها: class_0, class_1, ... (کلاس‌ها)
    """
    assert y_labels.shape[0] == subjects.shape[0], \
        "Length mismatch between labels and subjects!"

    # تعداد کلاس‌ها
    n_classes = int(y_labels.max()) + 1

    # سوژه‌های یکتا
    unique_subjects = np.unique(subjects)

    # در حالت دیباگ، فقط چند سوژه اول
    if debug_mode:
        unique_subjects = unique_subjects[:max_subjects_debug]
        print(f"DEBUG ({label_name}): using only subjects: {unique_subjects}")

    rows = []

    # آمار برای هر سوژه
    for subj in unique_subjects:
        mask = (subjects == subj)
        y_sub = y_labels[mask]
        counts = np.bincount(y_sub, minlength=n_classes)

        row = {"subject": subj}
        for cls in range(n_classes):
            row[f"class_{cls}"] = counts[cls]
        rows.append(row)

    # سطر جمع (TOTAL) – فقط روی همین سوژه‌های انتخاب‌شده
    mask_all = np.isin(subjects, unique_subjects)
    y_used = y_labels[mask_all]
    total_counts = np.bincount(y_used, minlength=n_classes)

    total_row = {"subject": "TOTAL"}
    for cls in range(n_classes):
        total_row[f"class_{cls}"] = total_counts[cls]
    rows.append(total_row)

    df = pd.DataFrame(rows)
    # ترتیب ستون‌ها: subject اول، بعد کلاس‌ها
    cols = ["subject"] + [f"class_{cls}" for cls in range(n_classes)]
    df = df[cols]
    return df


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

# ===================== ساخت و ذخیره‌ی CSV برای COVAS =====================
df_covas = make_class_counts_df(y_covas_ch2, subjects_ch2, label_name="COVAS")
covas_csv_path = os.path.join(results_dir, "covas_class_counts_per_subject.csv")
df_covas.to_csv(covas_csv_path, index=False, encoding="utf-8-sig")
print(f"\nSaved COVAS class counts to: {covas_csv_path}")

# ===================== ساخت و ذخیره‌ی CSV برای HEATER =====================
df_heater = make_class_counts_df(y_heater_ch2, subjects_ch2, label_name="HEATER")
heater_csv_path = os.path.join(results_dir, "heater_class_counts_per_subject.csv")
df_heater.to_csv(heater_csv_path, index=False, encoding="utf-8-sig")
print(f"Saved HEATER class counts to: {heater_csv_path}")

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
