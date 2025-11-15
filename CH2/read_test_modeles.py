import os
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from aeon.classification.hybrid import HIVECOTEV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import joblib
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

# ===================== ۴. ارزیابی مدل‌های ذخیره‌شده (LOSO) =====================
print("\n================ EVALUATE SAVED HC2 MODELS (LOSO) ================")

# لیست سوژه‌ها (همان که قبلاً استفاده کرده‌ای)
unique_subjects = np.unique(subjects_ch2)
print("Unique subjects:", unique_subjects)

all_preds = []   # برای نگه‌داشتن همه پیش‌بینی‌ها (برای محاسبه دقت کلی)
all_trues = []   # برای نگه‌داشتن همه لیبل‌های واقعی
per_subject_results = []  # برای ذخیره دقت هر سوژه به صورت جداگانه

for s in range(20):
    

    print(f"\n=== Evaluating saved HC2 model for subject = {s} ===")

    # مسیر فایل مدلی که قبلاً برای این سوژه ذخیره کرده‌ای
    model_path = f"./CH2/hc2_{s}_covas.joblib"

    # چک می‌کنیم که فایل مدل واقعاً وجود داشته باشد
    if not os.path.exists(model_path):
        print(f"  [WARN] Model file not found: {model_path}  --> skipping this subject")
        continue

    # ماسک تست برای این سوژه (همان مثل قبل)
    test_mask = (subjects_ch2 == s)

    # جدا کردن داده‌های تست این سوژه
    X_test = X_ch2[test_mask]
    y_test = y_target[test_mask]

    print("  Test size:", X_test.shape[0])

    # خواندن مدل ذخیره‌شده از روی دیسک
    hc2 = joblib.load(model_path)  # نوع خروجی: همان شیء HIVECOTEV2

    # پیش‌بینی روی داده‌های تست این سوژه
    y_pred = hc2.predict(X_test)

    # محاسبه دقت (accuracy) برای همین سوژه
    acc = accuracy_score(y_test, y_pred)
    print(f"  Subject {s} accuracy: {acc:.4f}")

    # نگه‌داشتن نتایج برای استفاده بعدی
    per_subject_results.append((s, acc, len(y_test)))
    all_preds.append(y_pred)
    all_trues.append(y_test)

# بعد از حلقه، اگر حداقل یک سوژه ارزیابی شده باشد
if len(all_preds) > 0:
    # چسباندن همه پیش‌بینی‌ها و لیبل‌های واقعی برای محاسبه دقت کلی
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)

    overall_acc = accuracy_score(all_trues, all_preds)
    print("\n=============================")
    print("Overall LOSO accuracy (using loaded models):", overall_acc)
    print("=============================")

    # چاپ دقت هر سوژه
    print("\nPer-subject accuracies:")
    for s, acc, n in per_subject_results:
        print(f"  Subject {s}  | n_test = {n:3d} | acc = {acc:.4f}")

    # گزارش کامل‌تر (precision, recall, f1) برای همه سوژه‌ها با هم
    print("\nDetailed classification report (all subjects together):")
    print(classification_report(all_trues, all_preds))
else:
    print("No subjects evaluated (no models found or empty test sets).")
