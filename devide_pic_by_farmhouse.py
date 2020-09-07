import copy
import json
import os
import sys
from shutil import copyfile

path = r"C:/Users/83942/Desktop/github/preprocess_aicam_photos/data/img"  # 存所有图片的文件夹

data_path='C:/Users/83942/Desktop/github/preprocess_aicam_photos/data/'

original_json_path = r"C:/Users/83942/Desktop/github/preprocess_aicam_photos/data/img/result/device_serial.json"  # 保存图片与其对应设备号的json路径

farmhouse_json_data_list = []

device_name_list=[]

device_pic_dict={}
device_picindex_dict={}
def generate_single_farmhouse_json(device_name,pic_list):
    #路径处理
    single_farmhouse_path = data_path+device_name
    if not os.path.exists(single_farmhouse_path):
        os.mkdir(single_farmhouse_path)

    if not os.path.exists(single_farmhouse_path+'/result'):
        os.mkdir(single_farmhouse_path+'/result')

    json_path= single_farmhouse_path +"/result/msg.json"

    for pic_name in pic_list:
        single_farmhouse_path=data_path+device_name
        # 复制图片
        copyfile(path+'/'+pic_name,single_farmhouse_path+'/'+pic_name)

    imgs_index_list=device_picindex_dict[device_name]
    save_json_from_pic_indexs( json_path, imgs_index_list)
def save_json_from_pic_indexs(json_path,imgs_index_list):
    list_imgs = []

    with open(path + '/Result/msg.json') as fd:
        doc = json.loads(fd.read())



    frames = doc['RECORDS']
    for frame_index in range(0, len(frames)):
        if frame_index in imgs_index_list:
            list_imgs.append(frames[frame_index])

    data_dict={}
    data_dict["RECORDS"]=list_imgs
    # 把dict中的数据写入新的json，并将json文件保存
    json_data = json.dumps(data_dict, indent=1)
    with open(json_path, 'w') as f:
        f.write(json_data)



with open(original_json_path) as fd:
    doc = json.loads(fd.read())
    index = 0
    for pic in doc['RECORDS']:
        pic_name= pic['msg_id']+'.jpg'
        device_name = pic['device_serial']
        if device_name not in device_pic_dict.keys():
            device_pic_dict[device_name] = [pic_name]
        else:
            device_pic_dict[device_name].append(pic_name)
        if device_name not in device_picindex_dict.keys():
            device_picindex_dict[device_name] = [pic_name]
        else:
            device_picindex_dict[device_name].append(index)
        index+=1
    for device_name in device_picindex_dict.keys():
        generate_single_farmhouse_json(device_name,device_pic_dict[device_name])
