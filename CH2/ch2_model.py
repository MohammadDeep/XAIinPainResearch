import numpy as np
from aeon.classification.hybrid import HIVECOTEV2
from sklearn.metrics import accuracy_score, f1_score
import time
import joblib
import os
import pandas as pd

# parametr for model ch2 
time_limit_in_minutes = 0
n_jobs = -1

'''
for test code
'''
debug_mode = False
max_subjects_debug = 3          # حداکثر چند سوژه برای تست
max_train_samples_debug = 20   # حداکثر چند نمونه train برای هر سوژه
max_test_samples_debug = 5     # حداکثر چند نمونه test برای هر سوژه

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
# در حالت دیباگ، طول زمانی سیگنال را نصف کن (هر 2 نمونه یکی)
if debug_mode:
    X_ch2 = X_ch2[:, :, ::2]
    print("DEBUG: X_ch2 shape after temporal downsample:", X_ch2.shape)

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


# ===================== ۳. LOSO بر اساس subjects =====================
print("\n================ LOSO EXPERIMENT (by subject) ================")

unique_subjects = np.unique(subjects_ch2)
print("Unique subjects:", unique_subjects)
# در حالت دیباگ، فقط چند سوژه اول را استفاده کن
if debug_mode:
    unique_subjects = unique_subjects[:max_subjects_debug]
    print("DEBUG: using only subjects:", unique_subjects)


# اگر پوشه وجود نداشت، ساخته می‌شود. اگر وجود داشته باشد، کاری نمی‌کند.
os.makedirs(save_modeles_path, exist_ok=True)

for s in unique_subjects:
    print(f"\n=== LOSO fold: subject = {s} ===")
    model_path = os.path.join(save_modeles_path, f"hc2_subjects_{s}_covas_time_limit_in_minutes_{time_limit_in_minutes}.joblib")
    if os.path.isfile(model_path): 
        print('This model already exists and does not need to be trained. ') 
                # فقط اگر فایل CSV وجود دارد، بخوان
        if os.path.isfile(results_csv):
            df = pd.read_csv(results_csv)

            # پیدا کردن سطری که model_path برابر این مسیر است
            row = df[df["model_path"] == model_path]

            # اگر سطر پیدا شد
            if not row.empty:
                print("Row found:")
                print(row)

                # اگر فقط اولین سطر را می‌خواهی به صورت دیکشنری
                info = row.iloc[0].to_dict()
                print("\nAs dict:")
                for k, v in info.items():
                    print(f"{k}: {v}")
            else:
                print("No row found for this model_path in CSV.")
        else:
            print("Results CSV does not exist yet.")
        # در این حالت، تست دوباره انجام نمی‌دهی و سطر جدید هم ثبت نمی‌کنی
    else:  
        print('State training')
        test_mask = (subjects_ch2 == s)
        train_mask = ~test_mask

        X_train, X_test = X_ch2[train_mask], X_ch2[test_mask]
        y_train, y_test = y_target[train_mask], y_target[test_mask]

        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
            # در حالت دیباگ، تعداد نمونه‌های train و test را محدود کن
        if debug_mode:
            if n_train > max_train_samples_debug:
                idx_tr = rng.choice(n_train, size=max_train_samples_debug, replace=False)
                X_train = X_train[idx_tr]
                y_train = y_train[idx_tr]
                n_train = X_train.shape[0]

            if n_test > max_test_samples_debug:
                idx_te = rng.choice(n_test, size=max_test_samples_debug, replace=False)
                X_test = X_test[idx_te]
                y_test = y_test[idx_te]
                n_test = X_test.shape[0]

            print(f"  DEBUG: reduced Train size: {n_train}, Test size: {n_test}")

        print("  Train size:", n_train, " Test size:", n_test)


        # اگر تو train فقط یک کلاس باشد، بعضی مدل‌ها مشکل پیدا می‌کنند
        if len(np.unique(y_train)) < 2:
            print("  [WARN] Train fold has only one class; skipping this subject.")
            continue

        hc2 = HIVECOTEV2(
            time_limit_in_minutes= time_limit_in_minutes,
            # اجازه بده STC، DrCIF، Arsenal و TDE از تنظیمات کامل پیش‌فرض خودشان استفاده کنند
            stc_params=None,
            drcif_params=None,
            arsenal_params=None,
            tde_params=None,

            n_jobs= n_jobs,
            random_state=0,
            verbose=1
        )

        print("  Fitting HC2...")
        
        t0 = time.time()
        hc2.fit(X_train, y_train)
        t1 = time.time()
        fit_time_sec = t1 - t0
        print(f"Fit time: {(fit_time_sec):.1f} seconds")
    


        # یا اگر مسیر خاصی می‌خواهی:
        # joblib.dump(hc2, "./models/hc2_covas.joblib")


        print("  Predicting for this subject...")
        t2 = time.time()
        y_proba = hc2.predict_proba(X_test)
        y_pred = np.argmax(y_proba, axis=1)
        t3 = time.time()
        eval_time_total_sec = t3 - t2
        eval_time_per_sample_sec = eval_time_total_sec / n_test


        # دقت برای این سوژه
        acc_s = accuracy_score(y_test, y_pred)
        # F1 macro: میانگین F1 روی همه کلاس‌ها با وزن برابر
        f1_macro_s = f1_score(y_test, y_pred, average="macro")

        print(f"  Subject {s} accuracy: {acc_s:.4f}")
        print(f"  Subject {s} F1 macro      : {f1_macro_s:.4f}")
        print(f"  Eval time total (sec): {eval_time_total_sec:.4f}")
        print(f"  Eval time per sample (sec): {eval_time_per_sample_sec:.6f}")

        # ذخیره مدل این سوژه
        joblib.dump(hc2, model_path)

        # اضافه کردن یک سطر به نتایج
        row_dict = {
            "subject_id": int(s),
            "train_size": n_train,
            "test_size": n_test,
            "accuracy": acc_s,
            "f1_macro": f1_macro_s,
            "fit_time_sec": fit_time_sec,
            "eval_time_total_sec": eval_time_total_sec,
            "eval_time_per_sample_sec": eval_time_per_sample_sec,
            "model_path": model_path,
            "time_limit_in_minutes" : time_limit_in_minutes,
            "n_jobs": n_jobs,
            "y_test|y_proba" :  list(zip(y_test, y_proba))
        }
                # تبدیل به DataFrame تک‌سطره
        df_row = pd.DataFrame([row_dict])

        # اگر فایل وجود نداشت → با header؛ اگر وجود داشت → فقط append بدون header
        file_exists = os.path.exists(results_csv)
        df_row.to_csv(results_csv, mode="a", header=not file_exists, index=False)
        print(f"  Appended results row to: {results_csv}")
