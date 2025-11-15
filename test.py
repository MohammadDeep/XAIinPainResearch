import numpy as np
from aeon.classification.hybrid import HIVECOTEV2
from sklearn.metrics import accuracy_score
import time
import joblib
import os
import pandas as pd  # برای ساخت DataFrame و ذخیره CSV

# مسیر فایل npy و محل ذخیره مدل‌ها
save_modeles_path = "./CH2/modeles_time_limit_in_minutes_0"
path = './datasets/painmonit/np-dataset/'
x_file = 'X.npy'
subjects_file = 'subjects.npy'
y_covas_file = 'y_covas.npy'
y_heater_file = 'y_heater.npy'


def read_File(path):
    arr = np.load(path)
    print("File:", path)
    print("  type:", type(arr))
    print("  shape:", arr.shape)
    print("  dtype:", arr.dtype)
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

# ===================== ۳. LOSO بر اساس subjects =====================
print("\n================ LOSO EXPERIMENT (by subject) ================")

unique_subjects = np.unique(subjects_ch2)
print("Unique subjects:", unique_subjects)

all_preds = []   # برای محاسبه‌ی دقت کلی LOSO
all_trues = []

# جایی که مدل‌ها ذخیره می‌شن
os.makedirs(save_modeles_path, exist_ok=True)

# لیست سطرهای نتایج برای ساخت DataFrame
results_rows = []

for s in unique_subjects:
    if s != 0:   # اگر سوژه 0 رو نمی‌خوای، مثل قبل ردش می‌کنیم
        print(f"\n=== LOSO fold: subject = {s} ===")
        test_mask = (subjects_ch2 == s)
        train_mask = ~test_mask

        X_train, X_test = X_ch2[train_mask], X_ch2[test_mask]
        y_train, y_test = y_target[train_mask], y_target[test_mask]

        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        print("  Train size:", n_train, " Test size:", n_test)

        # اگر تو train فقط یک کلاس باشد، بعضی مدل‌ها مشکل پیدا می‌کنند
        if len(np.unique(y_train)) < 2:
            print("  [WARN] Train fold has only one class; skipping this subject.")
            continue

        hc2 = HIVECOTEV2(
            time_limit_in_minutes=0,   # بدون محدودیت زمانی برای بهترین عملکرد
            stc_params=None,
            drcif_params=None,
            arsenal_params=None,
            tde_params=None,
            n_jobs=-1,
            random_state=0,
            verbose=1
        )

        print("  Fitting HC2...")
        t0 = time.time()
        hc2.fit(X_train, y_train)
        t1 = time.time()
        fit_time_sec = t1 - t0
        print(f"Fit time: {fit_time_sec:.1f} seconds")

        print("  Predicting for this subject...")
        t2 = time.time()
        y_pred = hc2.predict(X_test)
        t3 = time.time()
        eval_time_total_sec = t3 - t2
        eval_time_per_sample_sec = eval_time_total_sec / n_test

        # دقت برای این سوژه
        acc_s = accuracy_score(y_test, y_pred)
        print(f"  Subject {s} accuracy: {acc_s:.4f}")
        print(f"  Eval time total (sec): {eval_time_total_sec:.4f}")
        print(f"  Eval time per sample (sec): {eval_time_per_sample_sec:.6f}")

        # اضافه‌کردن پیش‌بینی‌ها برای محاسبه‌ی LOSO کلی
        all_preds.append(y_pred)
        all_trues.append(y_test)

        # ذخیره مدل این سوژه
        model_path = os.path.join(save_modeles_path, f"hc2_{s}_covas.joblib")
        joblib.dump(hc2, model_path)

        # اضافه کردن یک سطر به نتایج
        results_rows.append({
            "subject_id": int(s),
            "train_size": n_train,
            "test_size": n_test,
            "accuracy": acc_s,
            "fit_time_sec": fit_time_sec,
            "eval_time_total_sec": eval_time_total_sec,
            "eval_time_per_sample_sec": eval_time_per_sample_sec,
            "model_path": model_path
        })

# اگر همه foldها اسکپ نشده باشند
if len(all_preds) > 0:
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)

    loso_acc = accuracy_score(all_trues, all_preds)
    print("\n=============================")
    print("Overall LOSO Accuracy:", loso_acc)
    print("=============================")

    # ساخت DataFrame از نتایج هر سوژه
    df_results = pd.DataFrame(results_rows)

    # (اختیاری) اضافه کردن ردیف کلی OVERALL
    df_overall = pd.DataFrame([{
        "subject_id": "OVERALL",
        "train_size": df_results["train_size"].sum(),
        "test_size": df_results["test_size"].sum(),
        "accuracy": loso_acc,
        "fit_time_sec": df_results["fit_time_sec"].sum(),
        "eval_time_total_sec": df_results["eval_time_total_sec"].sum(),
        "eval_time_per_sample_sec": (
            df_results["eval_time_total_sec"].sum() / df_results["test_size"].sum()
        ),
        "model_path": ""
    }])

    df_final = pd.concat([df_results, df_overall], ignore_index=True)

    # ذخیره در فایل CSV کنار مدل‌ها
    csv_path = os.path.join(save_modeles_path, "hc2_covas_loso_results_time_limit_0.csv")
    df_final.to_csv(csv_path, index=False)

    print(f"\nSaved per-subject results to: {csv_path}")
else:
    print("No valid LOSO folds (check label distribution per subject).")
