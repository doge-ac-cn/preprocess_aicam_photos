import copy
import json
import os
import sys
import cv2
import numpy as np
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QLabel, QPushButton


class win(QDialog):
    def __init__(self):
        # 初始化一个img的ndarry，用于存储图像
        self.img = np.ndarray(())
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(1920, 1080)
        self.label = QLabel()

        # 布局设定
        layout = QGridLayout(self)
        layout.addWidget(self.label, 0, 1, 3, 4)
        # 选择数据文件夹
        self.data_path = QFileDialog.getExistingDirectory(self, "选取文件夹", './')

        self.index = 0

        # 读取json文件
        with open(self.data_path + '/Result/msg.json') as fd:
            with open("test.json", "r") as data_file:
                self.data_dict = json.load(data_file)

                self.data_dict['calibInfo']['VideoChannels'][0]['MediaInfo']['FilePath'] = self.data_path  # FilePath

                self.data_dict['calibInfo']['VideoChannels'][0]['MediaInfo']['FrameNum'] = len(
                    os.listdir(self.data_path)) - 1  # FrameNum
                print(os.listdir(self.data_path)[-1])
                self.data_dict['calibInfo']['VideoChannels'][0]['MediaInfo']['breakFrameNum'] = \
                os.listdir(self.data_path)[
                    -1]  # breakFrameNum
                self.doc = json.loads(fd.read())
        # 创建存储要保存的图片名的dict,key为图片index,value为要保存的目标
        self.imgs_to_save = {}
        # 创建要保存的文件的文件夹
        if not os.path.exists(self.data_path + '_convert'):
            os.mkdir(self.data_path + '_convert')
            os.mkdir(self.data_path + '_convert/Result')

        #加载第一张图
        self.loadImage()
    def loadImage(self):
        # 调用存储文件

        targets = json.loads(self.doc['RECORDS'][self.index]['algorithm_result'])
        region_points = \
        json.loads(self.doc['RECORDS'][self.index]['device_rule_result'])['alertInfo']['ruleInfo']['region']['polygon']
        alert_region_points = []
        for point in region_points:
            alert_region_points.append([int(float(point['x']) * 1920), int(float(point['y']) * 1080)])
        alert_region_points = np.array([alert_region_points], np.int32)
        original_img = cv2.imread(self.data_path + '/' + self.doc['RECORDS'][self.index]['msg_id'] + '.jpg')
        self.original_img =copy.deepcopy(original_img)

        img = copy.deepcopy(original_img)
        img =cv2.polylines(img, alert_region_points , 1, 255)

        #先绘制遮挡区域的mask
        mask = np.zeros(original_img.shape, np.uint8)
        mask = cv2.fillPoly(mask, alert_region_points, (255, 255, 255))
        #先画一个遮罩方便后面判断target在不在框内
        img_cache = cv2.bitwise_and(original_img,mask)
        self.targets_all = []
        self.targets_in_region=[]
        target_index=0
        for target in targets['targets']:
            rect = target['obj']['rect']

            dict_ = {"x1": int(float(img.shape[1]) * float(rect['x']))
                , "x2": int(float(img.shape[1]) * (float(rect['x']) + float(rect['w'])))
                , "y1": int(float(img.shape[0]) * float(rect['y']))
                , "y2": int(float(img.shape[0]) * (float(rect['y']) + float(rect['h'])))}

            # 判断target是否在区域内
            x_center = int(float(img.shape[1]) * (float(rect['x']) + float(rect['w']) / 2))
            y_center = int(float(img.shape[0]) * (float(rect['y']) + float(rect['h']) / 2))
            # 如果target在区域内，则添加这几个点到target对象内，并且给原图遮罩修正
            if (np.sum(img_cache[y_center]
                       [x_center]) != 0):

                # 添加这几个点到target对象内

                # 遮罩修正，去掉在区域内物体上覆盖的遮罩
                erase_mask_region = [
                    [dict_['x1'], dict_['y1']],
                    [dict_['x2'], dict_['y1']],
                    [dict_['x2'], dict_['y2']],
                    [dict_['x1'], dict_['y2']]

                ]
                # self.targets.append(erase_mask_region)
                erase_mask_region = np.array(erase_mask_region, np.int32)
                mask = cv2.fillPoly(mask, [erase_mask_region], (255,255, 255))
                cv2.rectangle(img, (dict_['x1'], dict_['y1']), (dict_['x2'], dict_['y2']), (0, 255, 0), 2)
                #区域内目标添加该目标索引
                self.targets_in_region.append(target_index)
            else:
                cv2.rectangle(img, (dict_['x1'], dict_['y1']), (dict_['x2'], dict_['y2']), (0, 0, 255), 2)

            self.targets_all .append(target_index)
            target_index += 1

        original_img = cv2.bitwise_and(original_img, mask)
        self.img_with_mask = original_img
        self.img = img
        self.displayImg()

    def displayImg(self):
        shrink = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.QtImg = QtGui.QImage(shrink.data,
                                  shrink.shape[1],
                                  shrink.shape[0],
                                  shrink.shape[1] * 3,
                                  QtGui.QImage.Format_RGB888)

        self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        self.label.show()

    def keyPressEvent(self, keyevent):
        print(keyevent.text())
        if keyevent.text() == 'a' or keyevent.text() == 'A':
            print("上一张")
            self.index -= 1
            self.loadImage()
        if keyevent.text() == 'd' or keyevent.text() == 'D':
            print("下一张")
            self.index += 1
            self.loadImage()
        if keyevent.text() == 'w'  or keyevent.text() == 'W':
            print("保存加遮罩的图片")
            # 在要保存的list内则显示已添加，否则直接添加进入list
            if (self.index not in self.imgs_to_save.keys()):
                self.imgs_to_save[self.index]=self.targets_in_region
                cv2.imwrite(self.data_path + '_convert/' + self.doc['RECORDS'][self.index]['msg_id'] + '.jpg',
                            self.img_with_mask)

            else:
                QtWidgets.QMessageBox.information(self, "图片整理工具", "已经添加过该图片了")
        if keyevent.text() == ' ' :
            print("保存原图")
            # 在要保存的list内则显示已添加，否则直接添加进入list
            if (self.index not in self.imgs_to_save.keys()):
                self.imgs_to_save[self.index]=self.targets_all
                cv2.imwrite(self.data_path + '_convert/' + self.doc['RECORDS'][self.index]['msg_id'] + '.jpg',
                            self.original_img)

            else:
                QtWidgets.QMessageBox.information(self, "图片整理工具", "已经添加过该图片了")


    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(self,
                                               '图片整理工具',
                                               "是否要退出程序？",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.saveImg()
            QtWidgets.QMessageBox.information(self, "图片整理工具", "保存数据完成，点击ok退出")
            event.accept()
        else:
            event.ignore()

    def saveImg(self):
        print(self.imgs_to_save)
        list_frames = []
        frames = self.doc['RECORDS']

        # 遍历每一张图
        for frame_index in range(0, len(frames)):
            if frame_index in self.imgs_to_save.keys():

                # 写入新的json文件
                frame = frames[frame_index]
                # print(frame)
                single_frame_model = copy.deepcopy(
                    self.data_dict['calibInfo']['VideoChannels'][0]['VideoInfo']['mapFrameInfos'][0])
                single_target_model = copy.deepcopy(single_frame_model['value']['mapTargets'][0])
                # single_frame_model['value']['mapTargets']
                targets_xy = []
                single_frame_model['key']['FrameNum'] = frame['msg_id'] + '.jpg'
                single_frame_model['value']['FrameNum'] = frame['msg_id'] + '.jpg'

                targets = json.loads(frame['algorithm_result'])['targets']

                target_index=0
                # 遍历每个目标
                for item in targets:
                    # print(item)
                    if(target_index in self.imgs_to_save[frame_index]):
                        rect = item['obj']['rect']
                        single_target_model['key'] = item['obj']['id']
                        single_target_model['value']['TargetID'] = item['obj']['id']
                        # 判断目标类别
                        if item['obj']['type'] == 1:
                            single_target_model['value']['PropertyPages'][0]['PropertyPageDescript'] = 'pig'
                        else:
                            single_target_model['value']['PropertyPages'][0]['PropertyPageDescript'] = 'people'

                        # 保存目标坐标
                        left = float(rect['x'])
                        top = float(rect['y'])
                        right = float(rect['x']) + float(rect['w'])
                        bottom = float(rect['y']) + float(rect['h'])
                        single_target_model['value']['Vertex'][0]['fX'] = left
                        single_target_model['value']['Vertex'][0]['fY'] = top
                        single_target_model['value']['Vertex'][1]['fX'] = right
                        single_target_model['value']['Vertex'][1]['fY'] = top
                        single_target_model['value']['Vertex'][2]['fX'] = right
                        single_target_model['value']['Vertex'][2]['fY'] = bottom
                        single_target_model['value']['Vertex'][3]['fX'] = left
                        single_target_model['value']['Vertex'][3]['fY'] = bottom
                        targets_xy.append(copy.deepcopy(single_target_model))
                    target_index+=1
                single_frame_model['value']['mapTargets'] = targets_xy
                list_frames.append(copy.deepcopy(single_frame_model))

        self.data_dict['calibInfo']['VideoChannels'][0]['VideoInfo']['mapFrameInfos'] = list_frames

        # 把dict中的数据写入新的json，并将json文件保存
        json_data = json.dumps(self.data_dict, indent=1)
        with open(self.data_path + '_convert/Result/' + self.data_path.split("/")[-1] + '_convert.json', 'w') as f:
            f.write(json_data)


if __name__ == '__main__':
    a = QApplication(sys.argv)
    w = win()
    w.show()
    sys.exit(a.exec_())
