import os
import time
import numpy as np
import pandas as pd
import joblib

from aeon.classification.hybrid import HIVECOTEV2  # برای unpickle شدن مدل‌ها لازم است
from sklearn.metrics import accuracy_score, classification_report
import os

# ===================== تنظیم مسیرها و نام فایل‌ها =====================


folder_path = "./CH2/result"

# اگر پوشه وجود نداشت، ساخته می‌شود. اگر وجود داشته باشد، کاری نمی‌کند.
os.makedirs(folder_path, exist_ok=True)

DATA_PATH = "./datasets/painmonit/np-dataset/"
X_FILE = "X.npy"
SUBJECTS_FILE = "subjects.npy"
Y_COVAS_FILE = "y_covas.npy"
Y_HEATER_FILE = "y_heater.npy"

MODELS_DIR =  "./CH2/modeles_time_limit_in_minutes_0" # جایی که مدل‌ها را با نام hc2_{s}_covas.joblib ذخیره کرده‌ای
CSV_OUTPUT_PATH = folder_path +"/hc2_covas_loso_results_time_limit_in_minutes_0.csv"


def read_file(file_path: str) -> np.ndarray:
    """
    خواندن فایل npy و چاپ اطلاعات پایه‌ای آن.
    """
    arr = np.load(file_path)
    print("File:", file_path)
    print("  type:", type(arr))
    print("  shape:", arr.shape)
    print("  dtype:", arr.dtype)
    return arr


# ===================== ۱. خواندن و آماده‌سازی X =====================
print("==================================")
print("Loading X ...")
X_raw = read_file(os.path.join(DATA_PATH, X_FILE))

print("----------------------------")
print("Creating data for HC2:")

# حذف بعد اضافه‌ی آخر: (N, T, C, 1) -> (N, T, C)
X_no_last = np.squeeze(X_raw, axis=-1)

# تغییر ترتیب ابعاد به فرم (N, C, T) که HC2 انتظار دارد
X_ch2 = np.transpose(X_no_last, (0, 2, 1))
print("X_ch2 shape for HC2:", X_ch2.shape)

# تبدیل به float32 برای مصرف RAM کمتر
X_ch2 = X_ch2.astype(np.float32)


# ===================== ۲. خواندن و آماده‌سازی y_covas =====================
print("==================================")
print("Loading y_covas ...")
y_covas_raw = read_file(os.path.join(DATA_PATH, Y_COVAS_FILE))

print("----------------------------")
print("Creating y_covas for HC2:")
# (N, 5) one-hot -> (N,) لیبل عددی 0..4
y_covas_ch2 = np.argmax(y_covas_raw, axis=1).astype(int)
print("y_covas_ch2:", y_covas_ch2.shape, np.unique(y_covas_ch2))


# ===================== ۳. خواندن و آماده‌سازی y_heater (در صورت نیاز) =====================
print("==================================")
print("Loading y_heater ...")
y_heater_raw = read_file(os.path.join(DATA_PATH, Y_HEATER_FILE))

print("----------------------------")
print("Creating y_heater for HC2:")
# (N, 6) one-hot -> (N,) لیبل عددی 0..5
y_heater_ch2 = np.argmax(y_heater_raw, axis=1).astype(int)
print("y_heater_ch2:", y_heater_ch2.shape, np.unique(y_heater_ch2))


# ===================== ۴. خواندن سوژه‌ها =====================
print("==================================")
print("Loading subjects ...")
subjects_ch2 = read_file(os.path.join(DATA_PATH, SUBJECTS_FILE))


# ===================== ۵. انتخاب نوع مسئله (covas یا heater) =====================
# اینجا تعیین می‌کنی که روی کدام لیبل ارزیابی انجام شود:
#   y_target = y_covas_ch2   # ۵ کلاسه (COVAS)
#   y_target = y_heater_ch2  # ۶ کلاسه (HEATER)
y_target = y_covas_ch2  # در صورت نیاز، به y_heater_ch2 تغییر بده

# چک سازگاری تعداد نمونه‌ها
assert (
    X_ch2.shape[0] == y_target.shape[0] == subjects_ch2.shape[0]
), "N mismatch between X, y_target, subjects!"

N = X_ch2.shape[0]
print("\nFinal shapes:")
print("  X_ch2:", X_ch2.shape)
print("  y_target:", y_target.shape)
print("  subjects_ch2:", subjects_ch2.shape)
print(f"  Total samples: {N}")


# ===================== ۶. ارزیابی مدل‌های ذخیره‌شده (LOSO) =====================
print("\n================ EVALUATE SAVED HC2 MODELS (LOSO) ================")

all_preds = []           # پیش‌بینی‌های همه سوژه‌ها
all_trues = []           # لیبل‌های واقعی همه سوژه‌ها
per_subject_results = [] # (subject_id, accuracy, n_test, eval_time_total_sec, eval_time_per_sample_sec)

total_eval_time_all = 0.0   # مجموع زمان ارزیابی همه سوژه‌ها
total_n_all = 0             # مجموع تعداد نمونه‌های تست همه سوژه‌ها

# می‌توانی از unique_subjects استفاده کنی که داینامیک باشد
unique_subjects = np.unique(subjects_ch2)
print("Unique subjects in data:", unique_subjects)

n_modeles = sum(
    1 for name in os.listdir(MODELS_DIR)
    if os.path.isfile(os.path.join(MODELS_DIR, name))
)

print("Number of files:", n_modeles)

 
for s in range(n_modeles):
    print(f"\n=== Evaluating saved HC2 model for subject = {s} ===")

    # مسیر فایل مدل ذخیره‌شده برای این سوژه
    model_path = os.path.join(MODELS_DIR, f"hc2_{s}_covas.joblib")

    # اگر فایل مدل وجود نداشت، این سوژه را اسکپ کن
    if not os.path.exists(model_path):
        print(f"  [WARN] Model file not found: {model_path}  --> skipping this subject")
        continue

    # ماسک تست برای این سوژه
    test_mask = (subjects_ch2 == s)

    # اگر این سوژه هیچ نمونه‌ی تست نداشت، اسکپ کن
    if not np.any(test_mask):
        print("  [WARN] No test samples for this subject --> skipping")
        continue

    # جدا کردن داده‌های تست این سوژه
    X_test = X_ch2[test_mask]
    y_test = y_target[test_mask]
    n_test = X_test.shape[0]    # تعداد نمونه‌های تست این سوژه (n_test)

    print("  Test size (n_test):", n_test)

    # خواندن مدل ذخیره‌شده از روی دیسک
    hc2 = joblib.load(model_path)

    # ---------- اندازه‌گیری زمان ارزیابی (predict) ----------
    t0 = time.time()
    y_pred = hc2.predict(X_test)
    t1 = time.time()

    eval_time_total_sec = t1 - t0  # زمان کل پیش‌بینی برای این سوژه
    eval_time_per_sample_sec = eval_time_total_sec / n_test  # زمان متوسط برای هر نمونه

    print(f"  Eval time total (sec): {eval_time_total_sec:.4f}")
    print(f"  Eval time per sample (sec): {eval_time_per_sample_sec:.6f}")

    # محاسبه دقت (accuracy) برای همین سوژه
    acc = accuracy_score(y_test, y_pred)
    print(f"  Subject {s} accuracy: {acc:.4f}")

    # ذخیره نتایج این سوژه
    per_subject_results.append(
        (int(s), acc, n_test, eval_time_total_sec, eval_time_per_sample_sec)
    )
    all_preds.append(y_pred)
    all_trues.append(y_test)

    # به‌روزرسانی مجموع‌ها برای ردیف OVERALL
    total_eval_time_all += eval_time_total_sec
    total_n_all += n_test

# بعد از حلقه، اگر حداقل یک سوژه ارزیابی شده باشد:
if len(all_preds) > 0:
    # چسباندن همه پیش‌بینی‌ها و لیبل‌های واقعی
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)

    overall_acc = accuracy_score(all_trues, all_preds)
    print("\n=============================")
    print("Overall LOSO accuracy (using loaded models):", overall_acc)
    print("=============================")

    # چاپ دقت و زمان هر سوژه
    print("\nPer-subject accuracies and timing:")
    for s, acc, n_test, t_tot, t_per in per_subject_results:
        print(
            f"  Subject {s:2d} | n_test = {n_test:3d} | "
            f"acc = {acc:.4f} | "
            f"time_total = {t_tot:.4f}s | "
            f"time_per_sample = {t_per:.6f}s"
        )

    # گزارش کامل‌تر (precision, recall, f1) برای همه سوژه‌ها
    print("\nDetailed classification report (all subjects together):")
    print(classification_report(all_trues, all_preds))

    # ===================== ۷. ذخیره نتایج در CSV =====================
    # ساخت DataFrame از نتایج هر سوژه
    df_results = pd.DataFrame(
        per_subject_results,
        columns=[
            "subject_id",
            "accuracy",
            "n_test",                   # تعداد نمونه تست آن سوژه
            "eval_time_total_sec",      # زمان کل predict برای آن سوژه
            "eval_time_per_sample_sec", # زمان متوسط predict برای هر نمونه
        ],
    )

    # ردیف کلی OVERALL
    overall_time_per_sample = total_eval_time_all / total_n_all

    df_overall = pd.DataFrame(
        [
            {
                "subject_id": "OVERALL",
                "accuracy": overall_acc,
                "n_test": total_n_all,
                "eval_time_total_sec": total_eval_time_all,
                "eval_time_per_sample_sec": overall_time_per_sample,
            }
        ]
    )

    df_final = pd.concat([df_results, df_overall], ignore_index=True)

    # ذخیره در فایل CSV
    df_final.to_csv(CSV_OUTPUT_PATH, index=False)

    print(f"\nSaved per-subject accuracies and timing info to: {CSV_OUTPUT_PATH}")

else:
    print("No subjects evaluated (no models found or empty test sets).")
