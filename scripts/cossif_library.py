import os
import cv2
import copy
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from scipy import spatial
from tqdm import tqdm
from prettytable import PrettyTable
import math


class CosSIF:
    def __init__(self):
        pass

    
    def _calculate_cosine_distance(self, imgA, imgB):                                
        # convert the images to (R,G,B) arrays
        imgA_array = np.array(imgA)
        imgB_array = np.array(imgB)

        # flatten the arrays so they are 1 dimensional vectors
        imgA_array = imgA_array.flatten()
        imgB_array = imgB_array.flatten()

        # divide the arrays by 255, the maximum RGB value to make sure every value is on a 0-1 scale
        imgA_array = imgA_array/255
        imgB_array = imgB_array/255

        similarity = -1 * (spatial.distance.cosine(imgA_array, imgB_array) - 1)

        return similarity


    def _get_folder_name(self, path):
        # get folder name from a path
        path = path.rstrip(os.path.sep)
        folder_name = os.path.basename(path)

        return folder_name


    def _create_temp_directories(self, t_path, s_path, record_save_path, image_size):
        temp_dir = os.path.join(record_save_path, 'temp_dir')
        temp_t_dir = os.path.join(temp_dir, 'temp_t_dir')
        temp_s_dir = os.path.join(temp_dir, 'temp_s_dir')

        if os.path.isdir(temp_dir)==True:
            shutil.rmtree(temp_dir)

        os.makedirs(temp_dir)
        os.makedirs(temp_t_dir)
        os.makedirs(temp_s_dir)

        # resize images
        def _resize_images(source, destination, colour):
            for name in tqdm(os.listdir(source), colour=colour):
                img = cv2.imread(os.path.join(source, name))
                img = cv2.resize(img, (image_size, image_size))
                cv2.imwrite(destination + '/' + name, img)     

        # target class
        print('Resizing images of the target class...')
        _resize_images(t_path, temp_t_dir, colour='MAGENTA')

        # secondary class
        print('Resizing images of the secondary class/classes...')
        for path in s_path:
            class_name = self._get_folder_name(path)
            class_path = temp_s_dir + '/' + class_name
            os.makedirs(class_path)
            _resize_images(path, class_path, colour='CYAN')

        return temp_dir, temp_t_dir, temp_s_dir


    def _calculate_similarities(self, t_path, s_path, record_save_path, file_name, record_range=0, image_size=64):
        # check if the defined save path is correct 
        if os.path.exists(record_save_path)!=True:
            raise FileNotFoundError(f"Error! Path '{record_save_path}' does not exist!")

        # if there's a single secondary class, put it into a list 
        if type(s_path) is not list:
            s_path = [s_path]

        # print target class 
        t_table = PrettyTable()
        t_table.field_names = ['TARGET CLASS', 'SAMPLES']
        t_table.add_row([self._get_folder_name(t_path), len(os.listdir(t_path))])
        print(t_table)

        # print secondary class/classes 
        s_table = PrettyTable()
        s_table.field_names = ['SECONDARY CLASS/CLASSES', 'SAMPLES']
        s_total = 0
        for path in s_path:
            s_total = s_total + len(os.listdir(path))
            s_table.add_row([self._get_folder_name(path), len(os.listdir(path))])
        print(s_table)

        # error handling of record range 
        if record_range==0:
            if len(s_path)==1:
                record_range = (len(os.listdir(s_path[0])) - 1)
            else:
                record_range = s_total
        else:
            # check if the record range is valid 
            t_total = len(os.listdir(t_path))
            if record_range > t_total or record_range < 1:
                raise ValueError(f"1 <= record_range <= {t_total-1}")

        # create temporary directories
        temp_dir, t_dir, s_dir = self._create_temp_directories(t_path, s_path, record_save_path, image_size)


        def _create_range_cell(record_range):
            cell = {
                'TARGET_CLASS_NAME': '', 
                'TARGET_CLASS_IMAGE': ''
            }
            for i in range(1, record_range+1):
                name = 'SECONDARY_CLASS_NAME_{}'.format(i) 
                img = 'SECONDARY_CLASS_IMAGE_{}'.format(i) 
                score = 'COSINE_DISTANCE_SCORE_{}'.format(i) 
                cell[name] = ''
                cell[img] = ''
                cell[score] = 0.0
            return cell


        def _update_range_cell(range_cell, temp_record):
            for idx, i in enumerate(temp_record):
                range_cell['SECONDARY_CLASS_NAME_%s' % str(idx + 1)] = i['NAME']
                range_cell['SECONDARY_CLASS_IMAGE_%s' % str(idx + 1)] = i['IMAGE']
                range_cell['COSINE_DISTANCE_SCORE_%s' % str(idx + 1)] = i['SCORE']


        def _create_temp_cell():
            temp = {'NAME': '', 'IMAGE': '', 'SCORE': 0.0}
            return temp


        # set R of records     
        set_R = []

        print('Calculating Similarities...')
        for t_img in tqdm(os.listdir(t_dir), colour='BLUE'):

            range_cell = _create_range_cell(record_range)

            # read image from target class
            imgA = cv2.imread(t_dir + '/' + t_img)
            imgA_from_array = Image.fromarray(imgA, 'RGB')

            temp_record = []

            for class_name in os.listdir(s_dir):

                class_path = s_dir + '/' + class_name

                for s_img in os.listdir(class_path):

                    # read image from secondary class/classes
                    imgB = cv2.imread(class_path + '/' + s_img)
                    imgB_from_array = Image.fromarray(imgB, 'RGB')

                    # calculate similarities between two images
                    result = self._calculate_cosine_distance(imgA_from_array, imgB_from_array)

                    if result==1:
                        continue
                    else:
                        temp = _create_temp_cell()
                        temp['NAME'] = class_name
                        temp['IMAGE'] = s_img
                        temp['SCORE'] = result
                        temp_record.append(temp)

            # sort by similarity score
            temp_sort = sorted(temp_record, key=lambda d: d['SCORE'])

            # filter the samples my the user given range
            temp_sort = temp_sort[-record_range:]

            # reverse sort 
            temp_sort = sorted(temp_sort, key=lambda d: d['SCORE'], reverse=True)

            # update record
            t_name = self._get_folder_name(t_path)
            range_cell['TARGET_CLASS_NAME'] = t_name
            range_cell['TARGET_CLASS_IMAGE'] = t_img
            _update_range_cell(range_cell, temp_sort)

            set_R.append(range_cell)

        # sort set_R in descending order 
        set_R = sorted(set_R, key=lambda d: d['COSINE_DISTANCE_SCORE_1'])

        # saving set_R file as numpy array
        np.save(os.path.join(record_save_path, file_name), np.array(set_R))

        # saving set_R file as csv document
        df = pd.DataFrame(set_R)
        df.to_csv(os.path.join(record_save_path, file_name + '.' + 'csv'), index=False)

        # delete temporary directories 
        shutil.rmtree(temp_dir)


    def _generate_threshold_score(self, set_R, alpha, remove='similar'):

        if 0 < alpha < 1:
            """
            len(set_R) ------> Stores the number of samples available in the set_R. [Example: len(set_R) = 304]
            alpha ------> Stores the percentage that will be filtered. [Eample: alpha = 0.85]
            x ------> Stores the number of samples after the filtering process. [Eample: x = math.ceil(304 * 0.85) = 259]                   
            """
            total_images = len(set_R)
            filtered_images = math.ceil(total_images * alpha)
            removed_images = total_images - filtered_images
            threshold_score = 0.0

            cell_count = 1 

            if remove=='similar':
                for cell in set_R:
                    if cell_count == filtered_images:
                        threshold_score = cell['COSINE_DISTANCE_SCORE_1']

                        # threshold table
                        threshold_table = PrettyTable()
                        threshold_table.field_names = ["Total Images", "Filtered Images", "Removed Images", "Threshold Score", "Order"]
                        threshold_table.add_row([total_images, filtered_images, removed_images, threshold_score, "Similar"])

                        print(threshold_table)
                    cell_count = cell_count + 1 
            elif remove=='dissimilar':
                for cell in reversed(set_R):
                    if cell_count == filtered_images:
                        threshold_score = cell['COSINE_DISTANCE_SCORE_1']
                        threshold_table = PrettyTable()
                        threshold_table.field_names = ["Total Images", "Filtered Images", "Removed Images", "Threshold Score", "Order"]
                        threshold_table.add_row([total_images, filtered_images, removed_images, threshold_score, "Dissimilar"])

                        print(threshold_table)
                    cell_count = cell_count + 1 
            else:
                print("Error: Invalid argument!")

            return threshold_score

        else:
            error_table = PrettyTable()
            error_table.field_names = ["Error"]
            error_table.add_row(["Filter range must be between 0 to 1"])
            print(error_table)


    def _export_filtered_images(self, source_dir, filtered_dir, removed_dir, set_R, alpha, remove='similar'):

        source_images = os.listdir(source_dir)
        filtered_images = []

        threshold = self._generate_threshold_score(set_R, alpha, remove)

        if remove=='similar':
            for i in set_R:
                if i["COSINE_DISTANCE_SCORE_1"] <= threshold:
                    filtered_images.append(i["TARGET_CLASS_IMAGE"])
        elif remove=='dissimilar':
            temp_dir = filtered_dir
            filtered_dir = removed_dir
            removed_dir = temp_dir
            for i in reversed(set_R):
                if i["COSINE_DISTANCE_SCORE_1"] < threshold:
                    filtered_images.append(i["TARGET_CLASS_IMAGE"])
        else:
            print("Error: Invalid argument!")

        for i in tqdm(source_images, colour="blue"):

            try:
                found = False

                for j in filtered_images:
                    name1 = os.path.splitext(i)[0]
                    name2 = os.path.splitext(j)[0]
                    if name1==name2:
                        image = cv2.imread(source_dir + "/" + i)
                        cv2.imwrite(filtered_dir + "/" + i, image)
                        found = True

                if found==False:
                    image = cv2.imread(source_dir + "/" + i)
                    cv2.imwrite(removed_dir + "/" + i, image)

            except AttributeError:
                print("Error!")  
     
    
    def calculate_and_filter(self, target_path, secondary_path, record_save_path, file_name, filter_type, filter_range, filtered_path,
                             removed_path, image_size=64, record_range=1, record_keep=True):
        # calculate similarities
        self._calculate_similarities(
            t_path=target_path, 
            s_path=secondary_path, 
            record_save_path=record_save_path, 
            file_name=f'{file_name}', 
            image_size=image_size,
            record_range=record_range
        )
        
        # load saved records
        set_R_of_records = np.load(os.path.join(record_save_path, f'{file_name}.npy'), allow_pickle=True)
        
        # export filtered images
        self._export_filtered_images(
            source_dir=target_path, 
            filtered_dir=filtered_path, 
            removed_dir=removed_path, 
            set_R=set_R_of_records,
            remove=filter_type,
            alpha=filter_range 
        )         
        
        # delete record files 
        if record_keep==False:
            os.remove(os.path.join(record_save_path, f'{file_name}.npy'))
            os.remove(os.path.join(record_save_path, f'{file_name}.csv'))
          
        
    def filter_from_records(self, target_path, record_save_path, file_name, filter_type, filter_range, filtered_path, removed_path):
        # load saved records
        set_R_of_records = np.load(os.path.join(record_save_path, f'{file_name}'), allow_pickle=True)

        # export filtered images
        self._export_filtered_images(
            source_dir=target_path, 
            filtered_dir=filtered_path, 
            removed_dir=removed_path, 
            set_R=set_R_of_records,
            remove=filter_type,
            alpha=filter_range 
        )         