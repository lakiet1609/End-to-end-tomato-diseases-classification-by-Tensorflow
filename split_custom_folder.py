import os
import random
import shutil
from pathlib import Path

def custom_folder(target_dir: str = None):
    ori_path = Path(target_dir)
    paths_train = ori_path / 'train' 
    paths_test = ori_path / 'test' 
        
    if paths_train.is_dir():
        shutil.rmtree(paths_train)
    paths_train.mkdir(parents = True, exist_ok = True)    
        
    if paths_test.is_dir():
        shutil.rmtree(paths_test)
    paths_test.mkdir(parents = True, exist_ok = True)  

    for subfolders in os.listdir(ori_path):
        if subfolders == 'test' or subfolders =='train':
            continue
        file_paths = os.path.join(ori_path, subfolders)
        class_name = str(subfolders).split('\\')[-1]
        sub_paths_train = Path(os.path.join(file_paths,class_name))
        
        if sub_paths_train.is_dir():
            shutil.rmtree(sub_paths_train)
        sub_paths_train.mkdir(parents = True, exist_ok = True)    
        
        filenames = os.listdir(file_paths)
        random.shuffle(filenames)
        split_up_ratio = 0.75
        train_split_idx = int(len(filenames) * split_up_ratio)
        train_filenames = filenames[:train_split_idx]
        
        for filename in train_filenames:
            if filename != class_name:
                filename_path = os.path.join(file_paths,filename)
                shutil.move(filename_path, sub_paths_train)
        
        shutil.move(sub_paths_train, paths_train)
        shutil.move(file_paths,paths_test)







        


        



 
