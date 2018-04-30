import numpy as np
import shutil
import os


def ensuredir(path):
    if not os.path.exists(path):
        os.makedirs(path)


path = os.path.join('data', 'PH2Dataset', 'ph2 dataset images')
out_path = os.path.join('data', 'PH2Dataset', 'images')

for i, patient in enumerate(sorted(list(os.walk(path))[0][1])):
    print(patient)
    img_path = os.path.join(path, patient,
                            '%s_Dermoscopic_Image' % patient.upper(),
                            patient.upper() + '.bmp')
    mask_path = os.path.join(path, patient, '%s_lesion' % patient.upper(),
                             patient.upper() + '_lesion.bmp')
    
    img_out_path = os.path.join(out_path, 'imgs', 'seg')
    ensuredir(img_out_path)
    
    mask_out_path = os.path.join(out_path, 'masks', 'seg')
    ensuredir(mask_out_path)
    
    shutil.copy(img_path, os.path.join(img_out_path, '%04d.bmp' % i))
    shutil.copy(mask_path, os.path.join(mask_out_path, '%04d.bmp' % i))
