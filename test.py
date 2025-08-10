import os
def find_folber(main_folder_path):
    
    all_subfolders = []

    if os.path.exists(main_folder_path) and os.path.isdir(main_folder_path):
        # os.walk در تمام پوشه‌ها و زیرپوشه‌ها قدم می‌زند
        for root, dirs, files in os.walk(main_folder_path):
            # در هر مرحله، 'dirs' لیستی از نام پوشه‌های موجود در 'root' است
            for directory in dirs:
                # آدرس کامل هر پوشه را به لیست اضافه می‌کنیم
                full_path = os.path.join(root, directory)
                all_subfolders.append(directory)
        
        print(f"all folder : '{main_folder_path}':")
        for folder in all_subfolders:
            print(folder)
    else:
        print(f"dont find folder'{main_folder_path}'")

from pathlib import Path
def fin_file(main_folder_path):
    # 1. آدرس پوشه‌ای که می‌خواهید در آن جستجو کنید
    

    # 2. فرمت فایلی که به دنبال آن هستید (با ستاره شروع می‌شود)
    file_format = '*.csv'  # مثال: برای پیدا کردن تمام فایل‌های متنی
    # file_format = '*.jpg'  # مثال: برای پیدا کردن تمام عکس‌های JPG
    # file_format = '*.py'   # مثال: برای پیدا کردن تمام فایل‌های پایتون

    # 3. جستجوی بازگشتی برای پیدا کردن تمام فایل‌ها با فرمت مورد نظر
    found_files = list(main_folder_path.rglob(file_format))

    # 4. نمایش نتایج
    if found_files:
        print(f"تعداد {len(found_files)} فایل با فرمت '{file_format}' پیدا شد:")
        for file_path in found_files:
            print(file_path)
    else:
        print(f"هیچ فایلی با فرمت '{file_format}' در پوشه '{main_folder_path}' پیدا نشد.")




from pathlib import Path
def find_dir_model(
    main_folder_path,
    specific_text,
    specific_text1,
    file_format = '*.jodlib'
    ):
  
   
    matching_files = []

    for file_path in main_folder_path.rglob(file_format):
    
        if specific_text in str(file_path) and specific_text1 in str(file_path):
            matching_files.append(file_path)

    return matching_files


find_folber('/home/mohammad/code_PMC/XAIinPainResearch/datasets/results')

from pathlib import Path

# آدرس دایرکتوری مورد نظر خود را اینجا وارد کنید
target_directory = Path('/home/mohammad/code_PMC/XAIinPainResearch/datasets/results')

# ابتدا بررسی می‌کنیم که آدرس معتبر و یک پوشه باشد
if target_directory.is_dir():
    # با یک لیست کامپرشن، تمام آیتم‌ها را پیمایش کرده و فقط پوشه‌ها را انتخاب می‌کنیم
    subfolders = [entry for entry in target_directory.iterdir() if entry.is_dir()]

    print(f"پوشه‌های موجود در '{target_directory}':")
    # نام هر پوشه را چاپ می‌کنیم
    for folder in subfolders:
        print(folder.name)
