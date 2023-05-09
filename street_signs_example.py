import os
import glob
from sklearn.model_selection import train_test_split
import shutil

def split_data(path_to_data, path_to_train, path_to_val, split_size=0.1):

    folders = os.listdir(path_to_data)
    
    for folder in folders:
        
        full_path = os.path.join(path_to_data, folder)
        images_path = glob.glob(os.path.join(full_path, '*.png'))
        
        x_train, x_val = train_test_split(images_path, train_size=1-split_size,test_size=split_size)
        
        for x in x_train:
            
            path_to_folder = os.path.join(path_to_train, folder)
            
            if not os.path.exists(path_to_folder):
                os.makedirs(path_to_folder)
            
            shutil.copy(x, path_to_train)
        
        for x in x_val:
            
            path_to_folder = os.path.join(path_to_val, folder)
            
            if not os.path.exists(path_to_folder):
                os.makedirs(path_to_folder)
            
            shutil.copy(x, path_to_val)
        
if __name__=="__main__":
    
    path_to_data = '/Users/zharec/groking-computer-vision/archive/Train'
    path_to_train = '/Users/zharec/groking-computer-vision/training_data/train'
    path_to_val = '/Users/zharec/groking-computer-vision/training_data/val'
    
    split_data(path_to_data, path_to_train, path_to_val, 0.1)