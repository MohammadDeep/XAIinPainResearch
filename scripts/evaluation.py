import time
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import keras.backend as K
from datetime import datetime

# کدهای اضافه شده
import joblib
from tensorflow.keras.models import Model
# پایان کدهای اضافه شده

from scripts.classifier import rf

#-------------------------------------------------------------------------------------------------------
# HiddenPrints
#-------------------------------------------------------------------------------------------------------
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def from_categorical(x):
	"""Returns the class vector for a given one-hot vector.

	Parameters
	----------
	x: np. One-hot vector.

	Returns
	-------
	np: np.argmax(x, axis= 1)
	"""
	return np.argmax(x, axis= 1)

def accuracy(actual, predicted):
	"""Function to calculate accuracy for a list of actual and a list of predicted elements.

	Parameters
	----------
	actual: list. List with the actual labels.
	predicted: list. List with the predicted labels.

	Returns
	-------
	float: Accuracy
	"""

	assert len(actual) == len(predicted)

	correct = 0
	for a,b in zip (actual, predicted):
		if a == b:
			correct += 1

	return correct/len(actual)

def macro_f1_score(actual, predicted):
	"""Function to calculate macro f1_score for a multi class problem.

	Parameters
	----------
	actual: list. List with the actual labels.
	predicted: list. List with the predicted labels.

	Returns
	-------
	float: macro f1_score
	"""

	result = []
	# Calculate score for all classes
	for cl in np.unique(actual):

		x = two_class_f1_score(actual==cl, predicted==cl)
		result.append(x)

	return np.mean(result)

def two_class_f1_score(actual, predicted):
	"""Function to calculate f1_score for a two class problem. Lists must given with True/False inside.

	Parameters
	----------
	actual: list. List with the actual labels. Must contain either True or False.
	predicted: list. List with the predicted labels. Must contain either True or False.

	Returns
	-------
	float: f1_score
	"""
	tp = sum(actual & predicted)
	fp = sum(predicted) - tp
	fn = sum(actual) - tp

	if tp>0:
		precision=float(tp)/(tp+fp)
		recall=float(tp)/(tp+fn)

		return 2*((precision*recall)/(precision+recall))
	else:
		return 0

def leave_one_subject_out(data, subjects, subject_id):
	"""Function to split dataset into test and training set. Elements of subject (subject_id) are part of the test set.
	Rest is part of training set. The column including subject IDs is removed finally.

	Parameters
	----------
	data: list. List of nps. All nps will be splitted in training/testing set.
	subject_id: int. ID of a subject.

	Returns
	-------
	np: x_train, x_test, y_train, y_test
	"""

	testing = subjects==subject_id

	# Extract data/label with testing_indices
	test = [i[testing] if i is not None else None for i in data]

	# Delete thos values from the initial data/label
	train = [i[~testing] if i is not None else None for i in data]

	return train + test

def rfe_loso(X, aug, hcf, y, subjects, clf, rfes= None, step= 1, output_csv = Path("results", "rfe_loso.csv"), save_model_summary= False):
	"""Function to validate a keras model using leave one subject out validation in combination with a LOSO and RFE.

	Args:
		X (Np): X data from dataset.
		hcf (Np): hcf data from dataset.
		y (Np): y data from dataset.
		subjects (Np): subjects data from dataset.
		rfe (List): List of different number of features to evaluate. If 'None' checks for a list with [1..num_features]. Defaults to None.
		step (Int): Number of features to remove per iteration. Defaults to 1.
		clf (classifer): Classifier chosen form 'classifier.py'.
		output_csv (path, optional): Path to the output CSV. Defaults to str(Path("results", "rfe_loso.csv")).
		save_model_summary (bool, optional): Whether to save the models summary in a txt file. Defaults to True.
	"""

	to_drop = []
	clf.param["step"]= step

	# --- calculate the number of features for a dummy network and subject 0
	# create dummy classifier
	dummy = type(clf)(clf.param)
	# split data into folds
	x_train, aug_train, hcf_train, y_train, sub_train, x_test, aug_test, hcf_test, y_test, sub_test= leave_one_subject_out(
		[X, aug, hcf, y, subjects], subjects, 1)
	# set the dataset for the classifier
	dummy.set_dataset(train_data= (x_train, y_train), test_data= (x_test, y_test), aug_data= (aug_train, aug_test),
									hcf_data= (hcf_train, hcf_test), sub_data= (sub_train, sub_test))
	# caculate feature shape
	dummy.data_processing()
	num_features = dummy.get_features(0)[0].shape[1]
	# delete variables
	del dummy, x_train, aug_train, hcf_train, y_train, sub_train, x_test, aug_test, hcf_test, y_test, sub_test

	# if RFEs is empty, create a list starting with 'num_features' and decreasing to 1
	if rfes is None:
		rfes = list(np.arange(num_features, 0, -1))

	# make sure that we start with all features
	if rfes[0] != num_features:
		rfes.insert(0, num_features)

	assert type(step) == int and step > 0 and step <= num_features
	assert all(earlier >= later for earlier, later in zip(rfes, rfes[1:]))

	# for all rfe elements to test
	for i, rfe in enumerate(over_bar := tqdm(rfes)):

		if rfe > num_features:
			continue

		clf.param["rfe"]= rfe

		# loso
		start_date= datetime.now()
		start_time= time.time()
		all_accs = []
		all_fscores = []
		df_importance = pd.DataFrame([])
		for subject in (pbar := tqdm(np.unique(subjects))):
			with HiddenPrints():

				# split data into folds
				x_train, aug_train, hcf_train, y_train, sub_train, x_test, aug_test, hcf_test, y_test, sub_test= leave_one_subject_out(
					[X, aug, hcf, y, subjects], subjects, subject)

				# set the dataset for the classifier
				clf.set_dataset(train_data= (x_train, y_train), test_data= (x_test, y_test), aug_data= (aug_train, aug_test),
												hcf_data= (hcf_train, hcf_test), sub_data= (sub_train, sub_test))

				# retrieve the features from the classifier
				clf.data_processing()
				hcf_train, hcf_test = clf.get_features(subject)

				# remove the features detected by RFE
				hcf_train = hcf_train.drop(columns= to_drop)
				hcf_test = hcf_test.drop(columns= to_drop)

				# build an empty rf
				current_rf = rf(clf.param)

				# set the dataset
				current_rf.set_dataset(train_data= (x_train, y_train), test_data= (x_test, y_test), aug_data= (aug_train, aug_test),
								hcf_data= (hcf_train, hcf_test), sub_data= (sub_train, sub_test))

				# train the rf just on the selected features
				current_rf.create_model()
				current_rf.train()

				# save the importance for later use
				importance = pd.DataFrame(current_rf.model.feature_importances_.reshape(1,-1), columns=list(current_rf.hcf_train.columns))
				df_importance = pd.concat([df_importance, importance], ignore_index=True)

				# save acc
				pred = list(from_categorical(current_rf.predict_test()))
				actuals = list(from_categorical(current_rf.y_test))
				acc = accuracy(pred, actuals)
				all_accs.append(acc)
				all_fscores.append(macro_f1_score(actuals, pred))

				# show acc
				acc = round(np.nanmean(all_accs, axis= 0) *100, 2)
				pbar.set_description(f"Accuracy '{acc}'")

		# Calculate mean importance
		mean_importance = pd.DataFrame({"Mean": df_importance.mean(axis= 0)}).T
		df_importance = pd.concat([df_importance, mean_importance])

		# save feature names in "rfe_features" param
		clf.param["rfe_features"] = [i for i in list(df_importance.columns) if i not in to_drop]

		# save scores and best features
		save_data(clf, output_csv, start_date, start_time, all_fscores, all_accs, df_importance, save_model_summary)

		# if we reached the end of the loop, nothing to do
		if (i == (len(rfes) - 1)):
			break

		# --- find the number of features to remove
		# set the default of numbers of parameters to remove to step size
		num_to_remove = step
		# check the difference of features in current RFE and next RFE round
		diff = rfes[i] - rfes[i + 1]
		# if the difference is smaller than step size
		if (diff < step):
			# we update the 'num_to_remove' value
			num_to_remove = diff

		# if the next RFE value is different to what we want to have
		if (diff != num_to_remove):
			# we add it to the list of RFE
			rfes.insert(i + 1, rfes[i] - num_to_remove)
			# and update the progress bar
			over_bar.reset(total= len(rfes)) 

		# find the least important features - sort mean importance of all subjects and remove the 'num_to_remove' worst labels 
		least_important = list(df_importance.loc["Mean"].sort_values().index)[:num_to_remove]
		# save columns to delete
		to_drop.extend(least_important)

'''
==============================================================================
Add code (my):start 
==============================================================================
'''
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors

def _is_onehot(y: np.ndarray) -> bool:
    return (y.ndim == 2) and (y.dtype != object) and (y.shape[1] > 1)

def _to_labels(y_onehot: np.ndarray) -> np.ndarray:
    return np.argmax(y_onehot, axis=1).astype(int)

def _to_onehot(y_labels: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y_labels), n_classes), dtype=np.float32)
    out[np.arange(len(y_labels)), y_labels] = 1.0
    return out

def augment_hcf_with_smote(
    hcf_train,              # DataFrame: (N, F)
    y_train,                # np.ndarray: (N, C) one-hot  یا (N,) لیبل
    sub_train=None,         # Optional np.ndarray: (N,)
    scaler="robust",        # "robust" | "standard" | شیء اسکالر سازگار
    smote_kind="smote",     # "smote" | "smotetomek" | "smoteenn" | "borderline"
    k_neighbors=5,
    sampling_strategy="auto",
    random_state=42
):
    """
    SMOTE فقط روی hcf_train (ویژگی‌های دستی). hcf_test را اصلاً وارد این تابع نکن.
    خروجی: hcf_train_aug, y_train_aug [, sub_train_aug]
    """
    # --- 0) ورودی‌ها به فرم استاندارد
    if isinstance(hcf_train, pd.DataFrame):
        X = hcf_train.copy()
    else:
        # اگر numpy بود، به DataFrame با نام ستون‌های ساده تبدیل کن
        X = pd.DataFrame(hcf_train)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    y = np.asarray(y_train)
    y_is_onehot = _is_onehot(y)
    if y_is_onehot:
        n_classes = y.shape[1]
        y_labels = _to_labels(y)
    else:
        y_labels = y.astype(int).ravel()
        n_classes = int(np.max(y_labels)) + 1

    # --- 1) اسکیل فقط روی TRAIN
    if scaler == "robust":
        scaler_obj = RobustScaler()
    elif scaler == "standard":
        scaler_obj = StandardScaler()
    else:
        scaler_obj = scaler  # شیء اسکالر پاس‌داده‌شده
    Xs = scaler_obj.fit_transform(X.values)  # فقط train

    # --- 2) انتخاب نوع SMOTE
    sm = None
    kind = smote_kind.lower()
    if kind == "smotetomek":
        sm = SMOTETomek(
            smote=SMOTE(k_neighbors=k_neighbors, sampling_strategy=sampling_strategy, random_state=random_state),
            random_state=random_state
        )
    elif kind == "smoteenn":
        sm = SMOTEENN(
            smote=SMOTE(k_neighbors=k_neighbors, sampling_strategy=sampling_strategy, random_state=random_state)
        )
    elif kind == "borderline":
        sm = BorderlineSMOTE(k_neighbors=k_neighbors, sampling_strategy=sampling_strategy, random_state=random_state)
    else:
        sm = SMOTE(k_neighbors=k_neighbors, sampling_strategy=sampling_strategy, random_state=random_state)

    # --- 3) اجرای SMOTE روی TRAIN
    Xs_res, y_res = sm.fit_resample(Xs, y_labels)

    # --- 4) برگشت به مقیاس اصلی و DataFrame با همان ستون‌ها
    X_res = scaler_obj.inverse_transform(Xs_res)
    cols = list(X.columns)
    hcf_train_aug = pd.DataFrame(X_res, columns=cols)

    # --- 5) بازسازی y با همان قالب ورودی
    if y_is_onehot:
        y_train_aug = _to_onehot(y_res, n_classes)
    else:
        y_train_aug = y_res

    # --- 6) اگر sub_train داریم، برای نمونه‌های مصنوعی subject تعیین کنیم
    if sub_train is not None:
        sub_train = np.asarray(sub_train)
        N = len(X)
        N_aug = len(X_res)
        if N_aug == N:
            return hcf_train_aug, y_train_aug, sub_train  # چیزی اضافه نشده

        # نزدیک‌ترین همسایهٔ هر نمونهٔ جدید بین نمونه‌های اصلی (در فضای اسکیل‌شده)
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(Xs)  # فقط روی اصلی‌ها
        Xs_new = Xs_res[N:]
        idx_src = nn.kneighbors(Xs_new, return_distance=False).ravel()
        sub_new = sub_train[idx_src]
        sub_train_aug = np.concatenate([sub_train, sub_new], axis=0)
        return hcf_train_aug, y_train_aug, sub_train_aug

    return hcf_train_aug, y_train_aug



'''
==============================================================================
Add code (my):end
==============================================================================
'''



def loso_cross_validation(X, aug, hcf, y, subjects, clf, output_csv = Path("results", "loso.csv"), save_model_summary= False, runs = 0):
	"""Function to validate a keras model using leave one subject out validation.

	Args:
		X (Np): X data from dataset.
		hcf (Np): hcf data from dataset.
		y (Np): y data from dataset.
		subjects (Np): subjects data from dataset.
		clf (classifer): Classifier chosen form 'classifier.py'.
		output_csv (path, optional): Path to the output CSV. Defaults to str(Path("results", "5_loso.csv")).
		save_model_summary (bool, optional): Whether to save the models summary in a txt file. Defaults to True.
	"""

	start_date= datetime.now()
	start_time= time.time()

	all_accs = []
	all_fscores = []
	all_actuals = []
	all_predictions= []

	rfe_features = []

	df_importance = pd.DataFrame([])
	pmote = 'y'
	print('do you like use pmote for data(y/n)? == "n"')
	for subject in (pbar := tqdm(np.unique(subjects))):

		x_train, aug_train, hcf_train, y_train, sub_train, x_test, aug_test, hcf_test, y_test, sub_test= leave_one_subject_out(
				[X, aug, hcf, y, subjects], subjects, subject)

		# completly delete old classifier and instantiate new one
		clf = type(clf)(clf.param)

		
		
		if  pmote == 'n':
			# افزایش داده های 
			print('in file evaluation.py :' \
			'param["smote"] = True' \
			'use smote for add data ')
			# ← فقط روی hcf_train افزایش داده انجام بده
			out = augment_hcf_with_smote(
				hcf_train=hcf_train,
				y_train=y_train,          # one-hot یا لیبل؛ هرچی هست همان را بده
				sub_train=sub_train,          # اگر داری
				scaler="robust",              # یا "standard"
				smote_kind="smotetomek",      # "smote" / "borderline" / "smoteenn"
				k_neighbors=5,
				sampling_strategy="auto",
				random_state=42
			)

			# خروجی‌ها را جایگزین «Train» کن؛ Test دست‌نخورده بماند
			if len(out) == 3:
				hcf_train_aug, y_train_aug, sub_train_aug = out
			else:
				hcf_train_aug, y_train_aug = out
				sub_train_aug = sub_train



			clf.set_dataset(train_data= (x_train, y_train_aug), test_data= (x_test, y_test), aug_data= (aug_train, aug_test),
						hcf_data= (hcf_train_aug, hcf_test), sub_data= (sub_train_aug, sub_test))
		else : 
			clf.set_dataset(train_data= (x_train, y_train), test_data= (x_test, y_test), aug_data= (aug_train, aug_test),
						hcf_data= (hcf_train, hcf_test), sub_data= (sub_train, sub_test))

		clf.data_processing()
		clf.create_model()
		clf.train()

		# ====================================================================
		# شروع کد اضافه شده برای ذخیره مدل
		# ====================================================================
		print('in file evaluation.py you can change save model or note')
		save_modeles = True
		if save_modeles:
			# نام مدل (مانند 'cnn' یا 'rf') و شناسه سوژه را می‌گیریم
			# calssefier
			model_name = clf.name
			classes_list = clf.param.get("classes", [])
			

			classes_str = "class_" + "_".join(
				",".join(str(x) for x in group)      # داخل هر زیرلیست با کاما
				for group in classes_list            # بین زیرلیست‌ها با _
			)

			# sensores
			sensors_list = clf.param.get("selected_sensors", [])
			sensors_str = "sensors_".join(sensors_list)
			# n Tree
			n_estimators = clf.param.get("n_estimators") # دریافت عدد

			# اگر پارامتر وجود داشت، رشته را بساز، در غیر این صورت یک مقدار پیش‌فرض بگذار
			if n_estimators is not None:
				n_tree_str = f"n_tree_{n_estimators}" # مثلا: "n_tree_100"
			else:
				n_tree_str = "DL_model" # یا هر نام دیگری برای مدل‌های یادگیری عمیق که این پارامتر را ندارند

			# ایجاد مسیر برای ذخیره فایل
			# ابتدا مطمئن می‌شویم پوشه saved_models وجود دارد
			save_dir = Path("saved_models", classes_str,sensors_str,n_tree_str )
			os.makedirs(save_dir, exist_ok=True)
			# ... (کدهای قبلی برای ساختن save_dir اینجا قرار دارند) ...

		
			# ----- بلوک کد زیر را جایگزین کنید -----

			# تشخیص نوع مدل و ذخیره‌سازی بر اساس آن
			if model_name == "rf":
				# این شرط به طور خاص مدل جنگل تصادفی را مدیریت می‌کند
				save_path = save_dir / f"{model_name}_{runs}_subject_{subject}.joblib"
				print(f"--- SAVING RF MODEL to: {save_path} ---") # پیغام برای اطمینان از اجرا
				joblib.dump(clf.model, save_path)

			elif isinstance(clf.model, Model):
				# این شرط مدل‌های TensorFlow/Keras را مدیریت می‌کند
				save_path = save_dir / f"{model_name}_{runs}_subject_{subject}.h5"
				print(f"--- SAVING KERAS MODEL to: {save_path} ---") # پیغام برای اطمینان از اجرا
				clf.model.save(save_path)

			else:
				print(f"--- WARNING: Model type '{model_name}' not recognized for saving! ---")

			# -----------------------------------------
			# تشخیص نوع مدل و ذخیره‌سازی بر اساس آن
			if isinstance(clf.model, Model):
				# اگر مدل از نوع TensorFlow/Keras باشد
				print('save model 1')
				save_path = save_dir / f"{model_name}_{runs}_subject_{subject}.h5"
				clf.model.save(save_path)
				# print(f"Keras model for subject {subject} saved to: {save_path}") # (اختیاری)
			elif hasattr(clf.model, 'predict_proba'): # یک راه برای تشخیص مدل‌های Scikit-learn
				# اگر مدل از نوع Scikit-learn باشد (مانند RandomForest)
				print('save model 2')
				save_path = save_dir / f"{model_name}_subject_{subject}.joblib"
				joblib.dump(clf.model, save_path)
				# print(f"Scikit-learn model for subject {subject} saved to: {save_path}") # (اختیاری)
		else:
			print('modeles dont save.')		
		# ====================================================================
		# پایان کد اضافه شده
		# ====================================================================

		# Save prediction and actual values
		fold_predictions = list(from_categorical(clf.predict_test()))
		all_predictions.extend([fold_predictions])
		fold_actuals = list(from_categorical(clf.y_test))
		all_actuals.extend([fold_actuals])

		fold_acc = accuracy(fold_actuals, fold_predictions)
		all_accs.append(fold_acc)
		all_fscores.append(macro_f1_score(fold_actuals, fold_predictions))

		# save importance
		if hasattr(clf, "model") and hasattr(clf.model, "feature_importances_"):
			importance = pd.DataFrame(clf.model.feature_importances_.reshape(1,-1), columns=list(clf.hcf_train.columns))
			df_importance = pd.concat([df_importance, importance], ignore_index=True)

		if "rfe_features" in clf.param:
			rfe_features.extend(clf.param["rfe_features"])

		acc = round(np.nanmean(all_accs, axis= 0) *100, 2)
		pbar.set_description(f"Accuracy '{acc}'")

		K.clear_session()

	if len(rfe_features) != 0:
		clf.param["rfe_features"] = list(set(clf.param["rfe_features"]))

	save_data(clf, output_csv, start_date, start_time, all_fscores, all_accs, df_importance, save_model_summary)

	return 	all_fscores, all_accs, all_predictions, all_actuals

def save_data(clf, output_csv, start_date, start_time, all_fscores, all_accs, df_importance, save_model_summary):
	# create output directory if does not exist
	output_dir = output_csv.parent
	if not output_dir.exists():
		os.makedirs(output_dir)

	# --- Make entry in results datasheet
	df =  pd.DataFrame()
	now_date = datetime.now()
	df.loc[0, "Start time"]= start_date
	df.loc[0, "End time"]= now_date
	df.loc[0, "Duration"]= str(now_date-start_date).split('.')[0]
	df.loc[0, "Net"]= clf.name
	df.loc[0, "F1 mean"] = round(np.nanmean(all_fscores) * 100, 2)
	df.loc[0, "F1 std"] = round(np.std(all_fscores) * 100, 2)
	df.loc[0, "Accuracy mean"] = round(np.nanmean(all_accs) * 100, 2)
	df.loc[0, "Accuracy std"] = round(np.std(all_accs) * 100, 2)
	df.loc[0, "F1"] = str(all_fscores)
	df.loc[0, "Accs"] = str(all_accs)
	df.loc[0, "Param"] = str(sorted(clf.param.items()))

	df.to_csv(output_csv, sep= ";", mode='a', decimal=',', index= False, header= not output_csv.exists())

	# --- save feature importance
	if hasattr(clf, "model") and hasattr(clf.model, "feature_importances_"):
		output_dir_importance = Path(output_dir, "importance", clf.param["dataset"])
		if not output_dir_importance.exists():
			os.makedirs(output_dir_importance)
		df_importance.index.name = "Subject"
		mean_importance = pd.DataFrame({"Mean": df_importance.mean(axis= 0)}).T
		df_importance = pd.concat([df_importance, mean_importance])
		df_importance = df_importance.astype("float")
		df_importance.to_csv(Path(output_dir_importance, "{}_importance.csv".format(round(start_time))), sep= ";", decimal=',', index= True)

		best_features = df_importance.loc["Mean"].sort_values()
		best_features.index.name = "Feature"
		best_features.name = "Importance"
		best_features.to_csv(Path(output_dir_importance, "{}_bestfeatures.csv".format(round(start_time))), sep= ";", decimal=',', index= True)
	
	# --- Save model summary
	if save_model_summary:
		if hasattr(clf, "model") and hasattr(clf.model, 'summary'):
			file_name = str(now_date).replace(" ", "_").replace(":", "-").partition(".")[0] + ".txt"
			model_summary_path = Path(output_csv.parent, "keras_summaries", file_name)
			if not Path(model_summary_path).parent.exists():
				os.makedirs(Path(model_summary_path).parent)

			# Create summary txt
			with open(model_summary_path, 'w') as f:
				clf.model.summary(print_fn=lambda x: f.write(x + '\n'))

def five_loso(X, aug, hcf, y, subjects, clf, runs= 1, output_csv = Path("results", "5_loso.csv")):
	"""Function to validate a keras model using a leave one subject out validation 5 times and computing the mean.

	Args:
		X (Np): X data from dataset.
		hcf (Np): hcf data from dataset.
		y (Np): y data from dataset.
		subjects (Np): subjects data from dataset.
		clf (classifer): Classifier chosen form 'classifier.py'.
		runs (int, optional): Number of runs to evaluate. Defaults to 5.
		output_csv (path, optional): Path to the output CSV. Defaults to str(Path("results", "5_loso.csv")).
	"""

	start_date= datetime.now()

	acc_mean = []
	acc_std = []
	f1_mean = []
	f1_std = []
	all_predictions = []
	all_actuals = []

	for i in tqdm(np.arange(runs)):
		fscores, accs, predictions, actuals = loso_cross_validation(X, aug, hcf, y, subjects, clf, runs = i)

		acc_mean.append(np.nanmean(accs))
		acc_std.append(np.std(accs))
		f1_mean.append(np.nanmean(fscores))
		f1_std.append(np.std(fscores))

		all_predictions.extend(predictions)
		all_actuals.extend(actuals)

	# --- Make entry in results datasheet
	df =  pd.DataFrame()
	now_date = datetime.now()
	df.loc[0, "Start time"]= start_date
	df.loc[0, "End time"]= now_date
	df.loc[0, "Duration"]= str(now_date-start_date).split('.')[0]
	df.loc[0, "Net"]= clf.name
	df.loc[0, "Avg. acc mean"] = round(np.nanmean(acc_mean) * 100, 2)
	df.loc[0, "Std. acc mean"] = round(np.std(acc_mean) * 100, 2)
	df.loc[0, "Avg. acc std"] = round(np.nanmean(acc_std) * 100, 2)
	df.loc[0, "Std. acc std"] = round(np.std(acc_std) * 100, 2)
	df.loc[0, "Avg. F1 mean"] = round(np.nanmean(f1_mean) * 100, 2)
	df.loc[0, "Std. F1 mean"] = round(np.std(f1_mean) * 100, 2)
	df.loc[0, "Avg. F1 std"] = round(np.nanmean(f1_std) * 100, 2)
	df.loc[0, "Std. F1 std"] = round(np.std(f1_std) * 100, 2)
	df.loc[0, "All F1 mean"] = str(f1_mean)
	df.loc[0, "All F1 std"] = str(f1_std)
	df.loc[0, "All Acc mean"] = str(acc_mean)
	df.loc[0, "All Acc std"] = str(acc_std)
	df.loc[0, "Param"] = str(sorted(clf.param.items()))

	if not output_csv.parent.exists():
		os.makedirs(output_csv.parent)

	df.to_csv(output_csv, sep= ";", mode='a', decimal=',', index= False, header= not output_csv.exists())

	return 	f1_mean, acc_mean, all_predictions, all_actuals