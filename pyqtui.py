import copy
import json
import os
import sys
import cv2
import numpy as np
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QLabel
from qtpy import QtCore


class win(QDialog):
    index = 0  # 读到哪一张
    num_pictures = 0  # 一共有多少张图
    img = np.ndarray(())  # 初始化一个img的ndarry，用于存储图像，同时也是用于显示
    img_with_mask=np.ndarray(()) # 初始化一个img，用于存储带遮罩的图像
    data_path = ""  # 选择数据文件夹
    imgs_to_save = {}  # 创建存储要保存的图片名的dict,key为图片index,value为要保存的目标
    original_img = np.ndarray(())  # 初始载入的图像
    targets_all = []  # 当前读取的图像的所有目标信息，做一个缓存
    targets_in_region = []  # 当前读取的图像的所有在区域内的目标信息，做一个缓存
    data_dict = "json"  # 读取的标注数据的json
    doc = "json"  # 最终要保存的json，加载自test.json这个模板文件
    cover_regions = []  # 所有的自主画遮罩的区域
    cover_region= [] # 正在画的自主画遮罩的区域
    is_drawing_flag=False #用于表示现在是否正在画框
    img_with_covering= np.ndarray(()) #有遮盖区域的图片
    def __init__(self):

        super().__init__()
        self.resize(1920, 1080)
        self.label = QLabel()

        # 布局设定
        layout = QGridLayout(self)
        layout.addWidget(self.label, 0, 1, 3, 4)
        # 选择数据文件夹
        self.data_path = QFileDialog.getExistingDirectory(self, "选取文件夹", './')



        # 读取json文件
        with open(self.data_path + '/Result/msg.json') as fd:
            with open("test.json", "r") as data_file:
                self.data_dict = json.load(data_file)

                self.data_dict['calibInfo']['VideoChannels'][0]['MediaInfo']['FilePath'] = self.data_path  # FilePath

                self.data_dict['calibInfo']['VideoChannels'][0]['MediaInfo']['FrameNum'] = len(os.listdir(self.data_path)) - 1  # FrameNumprint(os.listdir(self.data_path)[-1])
                self.data_dict['calibInfo']['VideoChannels'][0]['MediaInfo']['breakFrameNum'] = os.listdir(self.data_path)[-1]  # breakFrameNum
                self.doc = json.loads(fd.read())
                self.num_pictures = len(self.doc['RECORDS'])
        # 创建要保存的文件的文件夹
        if not os.path.exists(self.data_path + '_convert'):
            os.mkdir(self.data_path + '_convert')
            os.mkdir(self.data_path + '_convert/Result')

        reply = QtWidgets.QMessageBox.question(self,
                                               '图片整理工具',
                                               "是否定位到上一次查看的位置？",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            #断点续看
            self.load_index()
        else:
            pass
        reply = QtWidgets.QMessageBox.question(self,
                                               '图片整理工具',
                                               "是否加载上一次保存的缓存文件",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            # 加载上一次保存的缓存文件
            self.load_progress_file()
        else:
            pass

        # 加载第一张图
        self.loadImage()

    #加载当前图片
    def loadImage(self):
        # 调用存储文件
        self.cover_regions=[]#重置标定区域
        targets = json.loads(self.doc['RECORDS'][self.index]['algorithm_result'])
        region_points = \
            json.loads(self.doc['RECORDS'][self.index]['device_rule_result'])['alertInfo']['ruleInfo']['region'][
                'polygon']
        alert_region_points = []

        for point in region_points:
            alert_region_points.append([int(float(point['x']) * 1920), int(float(point['y']) * 1080)])
        alert_region_points = np.array([alert_region_points], np.int32)
        original_img = cv2.imread(self.data_path + '/' + self.doc['RECORDS'][self.index]['msg_id'] + '.jpg')
        self.original_img = copy.deepcopy(original_img)
        self.img_with_covering = copy.deepcopy(original_img)
        img = copy.deepcopy(original_img)
        img = cv2.polylines(img, alert_region_points, 1, (255,255,255),thickness=1)

        # 先绘制遮挡区域的mask
        mask = np.zeros(original_img.shape, np.uint8)
        mask = cv2.fillPoly(mask, alert_region_points, (255, 255, 255))
        # 先画一个遮罩方便后面判断target在不在框内
        img_cache = cv2.bitwise_and(original_img, mask)
        self.targets_all = []
        self.targets_in_region = []
        target_index = 0
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

                # 遮罩修正，去掉在区域内物体上覆盖的遮罩
                erase_mask_region = [
                    [dict_['x1'], dict_['y1']],
                    [dict_['x2'], dict_['y1']],
                    [dict_['x2'], dict_['y2']],
                    [dict_['x1'], dict_['y2']]

                ]
                # self.targets.append(erase_mask_region)
                erase_mask_region = np.array(erase_mask_region, np.int32)
                mask = cv2.fillPoly(mask, [erase_mask_region], (255, 255, 255))
                cv2.rectangle(img, (dict_['x1'], dict_['y1']), (dict_['x2'], dict_['y2']), (0, 255, 0), 2)
                # 区域内目标添加该目标索引
                self.targets_in_region.append(target_index)
            else:
                cv2.rectangle(img, (dict_['x1'], dict_['y1']), (dict_['x2'], dict_['y2']), (0, 0, 255), 2)

            #在框内写框的物体种类以及置信度
            if target['obj']['type'] == 1:
                img = cv2.putText(img, "pig"+str(target['obj']['confidence']), (x_center, y_center), cv2.FONT_ITALIC, 0.8, (255, 255, 255), 2)
            else:
                img = cv2.putText(img, "people"+str(target['obj']['confidence']), (x_center, y_center), cv2.FONT_ITALIC, 0.8, (255, 255, 255), 2)

            #target绘制完毕之后添加到target_all里
            self.targets_all.append(target_index)
            target_index += 1


        img_with_mask= cv2.bitwise_and(original_img, mask)
        self.img_with_mask = img_with_mask
        self.img = img
        self.displayImg(self.img)

    #在gui上显示当前图片
    def displayImg(self,img):
        shrink = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.QtImg = QtGui.QImage(shrink.data,
                                  shrink.shape[1],
                                  shrink.shape[0],
                                  shrink.shape[1] * 3,
                                  QtGui.QImage.Format_RGB888)

        self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        self.label.show()

    #重写按键响应事件用来实现图片切换和保存功能
    def keyPressEvent(self, keyevent):
        print(keyevent.text())
        if keyevent.text() == 'a' or keyevent.text() == 'A':
            print("上一张")
            if(self.index<=0):
                QtWidgets.QMessageBox.information(self, "图片整理工具", "已经是第一张了")
            else:
                self.index -= 1
                self.loadImage()


        if keyevent.text() == 'd' or keyevent.text() == 'D':
            print("下一张")
            if self.index >= self.num_pictures-1:
                QtWidgets.QMessageBox.information(self, "图片整理工具", "已经是最后一张了")
            else:
                self.index += 1
                self.loadImage()

        if keyevent.text() == 'w' or keyevent.text() == 'W':
            print("保存加遮罩的图片")
            # 在要保存的list内则显示已添加，否则直接添加进入list
            if self.index not in self.imgs_to_save.keys():
                self.imgs_to_save[self.index] = self.targets_in_region
                cv2.imwrite(self.data_path + '_convert/' + self.doc['RECORDS'][self.index]['msg_id'] + '.jpg',
                            self.img_with_mask)

            else:
                reply = QtWidgets.QMessageBox.question(self,
                                                       '图片整理工具',
                                                       "已经添加过该图片了，是否覆盖保存",
                                                       QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                       QtWidgets.QMessageBox.No)
                if reply == QtWidgets.QMessageBox.Yes:
                    self.imgs_to_save[self.index] = self.targets_in_region
                    cv2.imwrite(self.data_path + '_convert/' + self.doc['RECORDS'][self.index]['msg_id'] + '.jpg',
                                self.img_with_mask)
                else:
                    pass

        if keyevent.text() == ' ':
            print("保存原图")
            # 在要保存的list内则显示已添加，否则直接添加进入list
            print(str(self.imgs_to_save.keys()))
            if self.index not in self.imgs_to_save.keys():
                self.imgs_to_save[self.index] = self.targets_all
                cv2.imwrite(self.data_path + '_convert/' + self.doc['RECORDS'][self.index]['msg_id'] + '.jpg',
                            self.img_with_covering)

            else:
                reply = QtWidgets.QMessageBox.question(self,
                                                       '图片整理工具',
                                                       "已经添加过该图片了，是否覆盖保存",
                                                       QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                                       QtWidgets.QMessageBox.No)
                if reply == QtWidgets.QMessageBox.Yes:
                    self.imgs_to_save[self.index] = self.targets_all
                    cv2.imwrite(self.data_path + '_convert/' + self.doc['RECORDS'][self.index]['msg_id'] + '.jpg',
                                self.img_with_covering)
                else:
                    pass


        if(keyevent.text() == 's' or keyevent.text() == 'S' )  and self.is_drawing_flag==False:
            print("切换到画框模式")
            self.is_drawing_flag = True

        else:
            if (keyevent.text() == 's' or keyevent.text() == 'S' ) and self.is_drawing_flag == True:
                print("画框完毕")
                if(self.cover_region!=[]):
                    cover_region_cache=copy.deepcopy(np.array(self.cover_region,np.int32))
                    self.cover_regions.append(cover_region_cache)
                    self.load_cover_img()
                    self.cover_region=[]
                self.is_drawing_flag = False


    #重写鼠标滚动事件

    def wheelEvent(self, event):

        angle = event.angleDelta() / 8  # 返回QPoint对象，为滚轮转过的数值，单位为1/8度
        angleX = angle.x()  # 水平滚过的距离(此处用不上)
        angleY = angle.y()  # 竖直滚过的距离
        if angleY > 0:
            print("上一张")
            if (self.index <= 0):
                QtWidgets.QMessageBox.information(self, "图片整理工具", "已经是第一张了")
            else:
                self.index -= 1
                self.loadImage()
        else:  # 滚轮下滚
            print("下一张")
            if self.index >= self.num_pictures - 1:
                QtWidgets.QMessageBox.information(self, "图片整理工具", "已经是最后一张了")
            else:
                self.index += 1
                self.loadImage()

    #重写鼠标点击函数，用于自己画遮罩使用。
    def mousePressEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton and self.is_drawing_flag == True:  # 左键按下
            #往当前的单个遮盖区域内添加点
            self.cover_region.append([event.x(),event.y()])
            img = copy.deepcopy(self.img)
            cover_region_points = np.array([self.cover_region], np.int32)
            img = cv2.polylines(img,cover_region_points , 1, (0,0,0),thickness=3)
            self.displayImg(img)


    def load_cover_img(self):

        #要保存的图
        self.img_with_covering = cv2.fillPoly(self.img_with_covering,self.cover_regions, (0, 0, 0))
        #绘制的图
        self.img = cv2.fillPoly(self.img,self.cover_regions, (0, 0, 0))
        self.displayImg(self.img)


    #重写关闭程序事件
    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(self,
                                               '图片整理工具',
                                               "是否要退出程序？",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.savedata()
            self.save_progress_file()
            QtWidgets.QMessageBox.information(self, "图片整理工具", "保存数据完成，点击ok退出")
            event.accept()
        else:
            event.ignore()

    #关闭前对标注文件进行保存
    def savedata(self):
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

                target_index = 0
                # 遍历每个目标
                for item in targets:
                    # print(item)
                    if target_index in self.imgs_to_save[frame_index]:
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
                    target_index += 1
                single_frame_model['value']['mapTargets'] = targets_xy
                list_frames.append(copy.deepcopy(single_frame_model))

        self.data_dict['calibInfo']['VideoChannels'][0]['VideoInfo']['mapFrameInfos'] = list_frames

        # 把dict中的数据写入新的json，并将json文件保存
        json_data = json.dumps(self.data_dict, indent=1)
        with open(self.data_path + '_convert/Result/' + self.data_path.split("/")[-1] + '_convert.json', 'w') as f:
            f.write(json_data)

    #保存配置文件，方便下次打开时加载
    def save_progress_file(self):
        json_data = json.dumps(self.imgs_to_save, indent=1)
        with open(self.data_path + '/Result/img_to_save_cache.json', 'w') as f:
            f.write(json_data)

        with open(self.data_path + '/Result/index.txt', 'w') as f:
            f.write(str(self.index))


    #加载上一次退出时自动保存的配置文件
    def load_progress_file(self):
        if(os.path.exists(self.data_path + '/Result/img_to_save_cache.json')):
            with open(self.data_path + '/Result/img_to_save_cache.json', 'r') as f:
                json_dict=json.load(f)
                #不能直接复制，因为原来的int转json变成了str，需要转回来
                for key,value in json_dict.items():
                    self.imgs_to_save[int(key)]=value
        else:
            QtWidgets.QMessageBox.information(self, "图片整理工具", "未找到上一次使用的缓存")

    #定位到上一次退出时的浏览位置
    def load_index(self):
        if (os.path.exists(self.data_path + '/Result/img_to_save_cache.json')):
            with open(self.data_path + '/Result/index.txt', 'r') as f:
                self.index = int(f.read())
        else:
            QtWidgets.QMessageBox.information(self, "图片整理工具", "未找到上一次查看的位置")
if __name__ == '__main__':
    a = QApplication(sys.argv)
    w = win()
    w.show()
    sys.exit(a.exec_())


