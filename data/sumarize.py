import copy
import json
import os
import sys
from shutil import copyfile

path_to_save = "summary"
img_num = 0
last_pic_name = ""
frames_to_save = []  # 保存每张图的标注信息
# 读取json文件

# 创建要保存的文件的文件夹
if not os.path.exists(path_to_save):
    os.mkdir(path_to_save)
    os.mkdir(path_to_save + '/result')

# 先打开模板文件
with open("../test.json", "r") as data_file:
    data_dict = json.load(data_file)

    data_dict['calibInfo']['VideoChannels'][0]['MediaInfo']['FilePath'] = path_to_save  # FilePath

    data_dict['calibInfo']['VideoChannels'][0]['MediaInfo'][
        'FrameNum'] = img_num  # FrameNumprint(os.listdir(self.data_path)[-1])

for path in os.listdir("imgs_to_summary"):
    print(path)

    for json_file in os.listdir("imgs_to_summary/" + path + "/result"):
        print(json_file)

        with open("imgs_to_summary/" + path + "/result/" + json_file) as single_json:
            single_json = json.load(single_json)
            frames = single_json['calibInfo']['VideoChannels'][0]['VideoInfo']['mapFrameInfos']
            print(len(frames))
            for single_frame in frames:
                # 复制每个图片标注的信息
                frames_to_save.append(copy.deepcopy(single_frame))
                # 获取文件名
                img_name = single_frame['key']['FrameNum']
                # 复制图片
                # print(img_name)
                if os.path.exists("imgs_to_summary/" + path + "/" + img_name) and not os.path.exists(
                        path_to_save + '/' + img_name):
                    copyfile("imgs_to_summary/" + path + "/" + img_name, path_to_save + '/' + img_name)
                last_pic_name = img_name
                img_num += 1
                print(img_num)
    data_dict['calibInfo']['VideoChannels'][0]['VideoInfo']['mapFrameInfos'] = frames_to_save
data_dict['calibInfo']['VideoChannels'][0]['MediaInfo']['FrameNum'] = img_num
data_dict['calibInfo']['VideoChannels'][0]['MediaInfo']['breakFrameNum'] = last_pic_name
# 把dict中的数据写入新的json，并将json文件保存
json_data = json.dumps(data_dict, indent=0)
with open(path_to_save + '/result/' + path_to_save + '.json', 'w') as f:
    f.write(json_data)
