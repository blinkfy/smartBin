COMPUTERTEST=False
SHOWTIMEANDNAME=False
SINGLEPROCESS=False
USEDEEPCAMERA=True
JETSON=True
USEKEY=True
ULTRASNOIC=True
KEY=12
LED=11
import sys
import cv2
import math
import json
import queue
import time
import random
import vlc
import torch
import gc
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from PyQt5.QtCore import QUrl,Qt,pyqtSignal,QTime,QThread,QSize,QFileInfo,QDateTime,QEvent,QTimer,QMutex,QObject
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QMediaPlaylist
from PyQt5.QtWidgets import QApplication, QWidget,QPushButton,QVBoxLayout,QHBoxLayout,QLabel,QMainWindow, QTableView, QAbstractItemView,QAction,QSplitter,QHeaderView,QTabWidget,QTableWidget,QScrollArea,QSpinBox,QDoubleSpinBox,QGroupBox,QToolButton,QGridLayout,QSlider,QTableWidgetItem,QStyledItemDelegate,QStyleOptionViewItem,QAbstractItemView,QStyle,QMessageBox,QFrame
from PyQt5.QtGui import QImage, QKeyEvent, QMouseEvent, QPixmap,QIcon,QStandardItem, QStandardItemModel,QFont,QGuiApplication,QPalette,QColor, QBrush
if not JETSON:
    COMPUTERTEST=True
if not COMPUTERTEST:
    import myprogram.motors as motors
    import ultrasonic
    import Jetson.GPIO as GPIO

if not JETSON:
    path="F:/ultralytics-main/myprogram/"
else:
    path="/home/jetson/Desktop/ultralytics-main/myprogram/"
class VlcSignals(QObject):
    endReached = pyqtSignal() 
class videoPlayer(QWidget):
    def __init__(self):
        super(videoPlayer,self).__init__()
        self.resize(800,500)
        self.setWindowTitle('垃圾分类你我他,文明社区靠大家')
        self.setWindowIcon(QIcon(path+'images/trash.png'))
        main_window.my_signal.connect(self.startplay)
        self.instance = vlc.Instance()
        self.media_player = self.instance.media_player_new()
        self.video_frame = QFrame()
        self.video_frame.setStyleSheet("background-color: black;")
        
        # 设置VLC输出窗口
        if sys.platform.startswith('win'):
            self.media_player.set_hwnd(int(self.video_frame.winId()))
        elif sys.platform.startswith('linux'):
            self.media_player.set_xwindow(int(self.video_frame.winId()))
        elif sys.platform.startswith('darwin'):
            self.media_player.set_nsobject(int(self.video_frame.winId()))
        self.v_layout=QVBoxLayout()
        self.v_layout.addWidget(self.video_frame)
        self.v_layout.maximumSize()
        self.setStyleSheet("background-color: black;")
        self.setLayout(self.v_layout)
        # 初始化信号系统
        self.signals = VlcSignals()
        self.signals.endReached.connect(self.restart_playback)
        event_manager = self.media_player.event_manager()
        event_manager.event_attach(vlc.EventType.MediaPlayerEndReached, self._end_reached)

        #加载视频文件
        media = self.instance.media_new(path+'images/rabbish.mp4')
        self.media_player.set_media(media)
        self.media_player.play()

    def _end_reached(self, event):
        """VLC内部回调（在独立线程中执行）"""
        self.signals.endReached.emit()  # 发射信号到主线程
    
    def restart_playback(self):
        """重新开始播放（在主线程中执行）"""
        self.media_player.stop()
        self.media_player.play()

    def toMainWindow(self):
        print('go to main window')
        self.media_player.stop()
        main_window.showFullScreen()
        main_window.my_thread.time=QDateTime.currentDateTime().toMSecsSinceEpoch()#更新时间
        self.hide()
        #main_window.exec_()

    def startplay(self):
        print('start play video')
        self.media_player.play()

    def mousePressEvent(self, QMouseEvent):
        if QMouseEvent.button() == Qt.LeftButton:
            print('mouse down')
            self.toMainWindow()

class HoverTableView(QTableView):
    showDetailsSignal=pyqtSignal(int)
    def __init__(self, parent=None):
        super(HoverTableView, self).__init__(parent)
        self.setMouseTracking(True)  # 开启鼠标跟踪
        self.viewport().installEventFilter(self)
        self.model = QStandardItemModel(self)
        if SHOWTIMEANDNAME:
            self.model.setHorizontalHeaderLabels(['名称','类型','数量','状态','时间'])
        else:
            self.model.setHorizontalHeaderLabels(['序号','垃圾种类','数量','分类成功与否'])
        self.setStyleSheet("font-size: 15pt;")
        self.setModel(self.model)
 
    def eventFilter(self, watched, event):
        if watched == self.viewport() and event.type() == QEvent.MouseMove:
            # 鼠标在视口上移动时触发
            index = self.indexAt(event.pos())
            if index.isValid():
                self.showDetailsSignal.emit(index.row())
        return super(HoverTableView, self).eventFilter(watched, event)
    
class MainWindow(QMainWindow):
    my_signal = pyqtSignal() #自定义信号
    def __init__(self):
        super(MainWindow,self).__init__()
        self.setWindowTitle('垃圾分类')
        self.setWindowIcon(QIcon(path+'images/trash.png'))
        self.setStyleSheet("font-size: 10pt;")
        self.menu_init()                                        #初始化菜单栏
        self.resize(QGuiApplication.primaryScreen().size())
        self.move(0,0)
        self.status_bar = self.statusBar()                      #添加状态栏
        self.status_bar.showMessage('正在加载模型')
        self.isplay=True
        self.is_perspect=False
        self.mousePoint=None
        self.control_key=False
        self.points=[]
        self.tableDetails=[]
        self.zoom_show=1.0
        self.rubbishDict={'bottle':'可回收垃圾','bottle2':'可回收垃圾','can':'可回收垃圾','papercup':'可回收垃圾','cap':'可回收垃圾','paper':'可回收垃圾','mu':'可回收垃圾','potato':'厨余垃圾','potatocut':'厨余垃圾','rabbitcut':'厨余垃圾','mooli':'厨余垃圾','battery':'有害垃圾','battery1':'有害垃圾','battery5':'有害垃圾','drug':'有害垃圾','drugbag':'有害垃圾','drugbox':'有害垃圾','capsule':'有害垃圾'}
        self.motor_name_list=['catch','barrier1','barrier2','barrier3','barrier4','platformT','platformR']
        self.jsonData={'points':[(0,0),(320,0),(320,320),(0,320)],'vflip':False,'hflip':False,'rotate90':False,'CLAHE':False,'sharpening':False,'fliter':False,'width':640,'height':640, "z": False, "catch": 0, "barrier1": 0, "barrier2": 0, "barrier3": 0, "barrier4": 0, "platformT": 0, "platformR": 0, "rotate": 0,"detect_threshold":40.0,"catch_threshold":10.0,"axis-width":31000,"axis-height":28000,"axis-x1":0,"axis-y1":0,"axis-x2":479,"axis-y2":479,"full_distance":10,"full_time":1}
        try:
            fp=open(path+'data.json','r')
            self.jsonData=json.load(fp)
            print(self.jsonData)
            fp.close()
            if not COMPUTERTEST:
                motors.platformR=self.jsonData['platformR']
                motors.rotate=self.jsonData['rotate']
                motors.axis_x=self.jsonData['axis-width']
                motors.axis_y=self.jsonData['axis-height']
        except Exception as e:
            print('open json error:',e)
            self.status_bar.showMessage('打开json错误')
        
        self.my_thread = RecognitionThread()
        self.my_thread.my_signal.connect(self.resetImg)
        self.my_thread.readCamera.img_tab2_signal.connect(self.resetImg2)
        self.my_thread.readCamera.img_tab2_signal2.connect(self.resetImg3)
        self.my_thread.readCamera.img_tab2_signal3.connect(self.resetDeepImg)
        self.my_thread.newItem_signal.connect(self.newItem_func)
        self.my_thread.mainWindowToPlayVideo_singal.connect(self.close)
        self.my_thread.updataVelocityToTab2_singal.connect(self.updata_move_velocity_to_tab2)
        self.my_thread.pause_play_singal.connect(self.pause_play)
        self.my_thread.start()
        self.my_thread.setPriority(QThread.HighPriority)
        self.my_thread.moveThread.keepLoop_singal.connect(self.keeploop)
        
        self.settingPage=SettingWindow()
        self.settingPage.hide()
        self.settingPage.image_label.resize(200,200)
        self.my_thread.readCamera.img_signal2.connect(self.settingPage.resetImg)
        self.layout_init()

    def menu_init(self):
        self.menu_menu = self.menuBar().addMenu('菜单')#创建一个菜单栏
        self.back_action = QAction('返回', self)
        self.back_action.setIcon(QIcon(path+'images/back.png'))
        self.back_action.setShortcut('Ctrl+B')
        self.back_action.setStatusTip('观看宣传片')
        self.menu_menu.addAction(self.back_action)
        self.back_action.triggered.connect(self.close)
        
        self.pause_action = QAction('暂停', self)
        self.pause_action.setText('暂停')
        self.pause_action.setIcon(QIcon(path+'images/pause.png'))
        self.pause_action.setShortcut('Ctrl+P')
        self.pause_action.setStatusTip('暂停摄像头')
        self.pause_action.setEnabled(False)
        self.menu_menu.addAction(self.pause_action)
        self.pause_action.triggered.connect(self.pause_play)

        self.setting_action = QAction('首选项', self)
        self.setting_action.setIcon(QIcon(path+'images/setting.png'))
        self.setting_action.setShortcut('Ctrl+,')
        self.setting_action.setStatusTip('设置')
        self.menu_menu.addAction(self.setting_action)
        self.setting_action.triggered.connect(self.toSettingPage)
        
        self.debug_action = QAction('调试', self)
        self.debug_action.setIcon(QIcon(path+'images/debugging.png'))
        self.debug_action.setShortcut('Ctrl+D')
        self.debug_action.setStatusTip('调试')
        self.menu_menu.addAction(self.debug_action)
        self.debug_action.triggered.connect(self.debug_func)
        
        self.manual_action = QAction('仅识别', self)
        self.manual_action.setIcon(QIcon(path+'images/changes.png'))
        self.manual_action.setShortcut('Ctrl+M')
        self.manual_action.setStatusTip('连续识别,不抓取')
        self.manual_action.setCheckable(True)
        self.manual_action.setChecked(False)
        self.menu_menu.addAction(self.manual_action)
        #self.manual_action.triggered.connect( )

        
        self.menu_menu.addSeparator()

        self.bilateralFilter_action = QAction('双边滤波', self)
        self.bilateralFilter_action.setIcon(QIcon(path+'images/blur.png'))
        self.bilateralFilter_action.setShortcut('Ctrl+F')
        self.bilateralFilter_action.setStatusTip('双边滤波')
        self.menu_menu.addAction(self.bilateralFilter_action)
        self.bilateralFilter_action.triggered.connect(lambda:self.bilateralFilter_bt.setChecked(not self.bilateralFilter_bt.isChecked()))
        
        self.sharpening_action = QAction('锐化', self)
        self.sharpening_action.setIcon(QIcon(path+'images/sharpen.png'))
        self.sharpening_action.setShortcut('Ctrl+S')
        self.sharpening_action.setStatusTip('卷积锐化')
        self.menu_menu.addAction(self.sharpening_action)
        self.sharpening_action.triggered.connect(lambda:self.sharpening_bt.setChecked(not self.sharpening_bt.isChecked()))
        
        self.equalize_action = QAction('CLAHE均衡化', self)
        self.equalize_action.setIcon(QIcon(path+'images/equalizer.png'))
        self.equalize_action.setShortcut('Ctrl+C')
        self.equalize_action.setStatusTip('CLAHE均衡化')
        self.menu_menu.addAction(self.equalize_action)
        self.equalize_action.triggered.connect(lambda:self.equalize_bt.setChecked(not self.equalize_bt.isChecked()))

        self.menu_menu.addSeparator()

        self.fullScreen_action = QAction('退出全屏', self)
        self.fullScreen_action.setIcon(QIcon(path+'images/exit-fullscreen.png'))
        self.fullScreen_action.setShortcut('F11')
        self.fullScreen_action.setStatusTip('退出全屏')
        self.fullScreen_action.setCheckable(True)
        self.fullScreen_action.setChecked(False)
        self.menu_menu.addAction(self.fullScreen_action)
        self.fullScreen_action.triggered.connect(self.fullscreenornot)

        self.exit_action = QAction('退出', self)
        self.exit_action.setIcon(QIcon(path+'images/exit.png'))
        self.exit_action.setShortcut('Ctrl+Q')
        self.exit_action.setStatusTip('退出程序')
        self.menu_menu.addAction(self.exit_action)
        self.exit_action.triggered.connect(self.exit)

    def layout_init(self):
        #===========================================================================Tab1
        self.image_label = QLabel(self)
        convertToQtFormat = QImage(path+'images/loading.png')
        #qtimg = convertToQtFormat.scaled(int(QGuiApplication.primaryScreen().size().width()*0.57),int(QGuiApplication.primaryScreen().size().width()*0.57), Qt.KeepAspectRatio)
        self.image_label.setPixmap(QPixmap.fromImage(convertToQtFormat))
        self.image_label.setScaledContents(True)
        self.lengthOfSide=min(int(QGuiApplication.primaryScreen().size().width()*0.57),self.height()-50)
        self.image_label.setFixedSize(self.lengthOfSide,self.lengthOfSide)
        
        self.table_v_layout=QVBoxLayout()
        self.model = QStandardItemModel(self)
        if SHOWTIMEANDNAME:
            self.model.setHorizontalHeaderLabels(['名称','类型','数量','状态','时间'])
        else:
            self.model.setHorizontalHeaderLabels(['序号','垃圾种类','数量','分类成功与否'])
        self.table = HoverTableView(self) 
        self.table.setStyleSheet("font-size: 15pt;")
        self.table.verticalHeader().setVisible(False)
        self.table.setModel(self.model)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)#宽度拉满
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)#禁用编辑
        self.table_v_layout.addWidget(self.table)
        self.table.showDetailsSignal.connect(self.showDetails)
        

        self.sum_table=QTableWidget(2, 4, self)
        self.sum_table.setItem(0,0,QTableWidgetItem('厨余垃圾'))
        self.sum_chicken=QTableWidgetItem('0')
        self.sum_table.setItem(0,1,self.sum_chicken)
        self.sum_table.setItem(0,2,QTableWidgetItem('其他垃圾'))
        self.sum_others=QTableWidgetItem('0')
        self.sum_table.setItem(0,3,self.sum_others)
        self.sum_table.setItem(1,0,QTableWidgetItem('有害垃圾'))
        self.sum_harmful=QTableWidgetItem('0')
        self.sum_table.setItem(1,1,self.sum_harmful)
        self.sum_table.setItem(1,2,QTableWidgetItem('可回收垃圾'))
        self.sum_recycle=QTableWidgetItem('0')
        self.sum_table.setItem(1,3,self.sum_recycle)
        self.fully_alarm=QTableWidgetItem('满载警报')
        self.fully_alarm.setFont(QFont('楷体',13, QFont.Bold,False))
        self.sum_table.setMaximumHeight(140)
        self.sum_table.setStyleSheet("font-size: 12pt;")
        self.sum_table.setShowGrid(False)
        self.sum_table.horizontalHeader().setHidden(True)
        self.sum_table.verticalHeader().setHidden(True)
        self.sum_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_v_layout.addWidget(self.sum_table)
        self.table_widget=QWidget(self)
        self.table_widget.setLayout(self.table_v_layout)
        
        self.status_v_layout=QVBoxLayout()
        self.back_bt=QPushButton('观看宣传片',self)
        self.back_bt.setIcon(QIcon(path+'images/trash.png'))
        self.back_bt.clicked.connect(self.back_action.trigger)
        self.status_v_layout.addWidget(self.back_bt)
        self.debug_bt=QPushButton('模拟测试',self)
        self.debug_bt.setIcon(QIcon(path+'images/debugging.png'))
        self.debug_bt.clicked.connect(self.debug_func)
        self.status_v_layout.addWidget(self.debug_bt)
        self.clear_bt=QPushButton('一键清空',self)
        self.clear_bt.setIcon(QIcon(path+'images/bin.png'))
        self.clear_bt.clicked.connect(self.clear_func)
        self.status_v_layout.addWidget(self.clear_bt)
        self.status=QLabel('状态：\n正在加载模型',self)
        self.status_v_layout.addWidget(self.status)
        self.move_status=QLabel('',self)
        self.status_v_layout.addWidget(self.move_status)
        self.status_bt=QPushButton('手动分类完成',self)
        self.status_bt.clicked.connect(self.keeploop)
        self.status_v_layout.addWidget(self.status_bt)
        self.rightWeight=QWidget(self)
        self.rightWeight.setLayout(self.status_v_layout)
        
        self.splitter=QSplitter(self)
        self.splitter.addWidget(self.table_widget)
        self.splitter.setShortcutEnabled(1)
        self.splitter.addWidget(self.image_label)
        self.splitter.addWidget(self.rightWeight)
        self.splitter.setSizes([int(QGuiApplication.primaryScreen().size().width()*0.4),int(QGuiApplication.primaryScreen().size().width()*0.55),int(QGuiApplication.primaryScreen().size().width()*0.05)])

        #==================================================================================Tab2↓
        self.img_all_scrollArea=QScrollArea(self)
        self.img_all_scrollArea.setWidgetResizable(True)
        self.img_all_widget=QWidget(self.img_all_scrollArea)
        self.img_v_layout=QVBoxLayout(self.img_all_widget)
        convertToQtFormat = QImage(path+'images/loading.png')
        #qtimg = convertToQtFormat.scaled(400,400, Qt.KeepAspectRatio)
        self.scroll_area = QScrollArea(self.img_all_widget) 
        self.image_label2 = QLabel(self.scroll_area)
        self.image_label2.setPixmap(QPixmap.fromImage(convertToQtFormat))
        self.image_label2.setScaledContents(True)
        self.scroll_area.setWidget(self.image_label2)
        self.orgimg_label=QLabel('原图：',self.img_all_widget)
        self.img_v_layout.addWidget(self.orgimg_label)
        self.img_v_layout.addWidget(self.scroll_area)
        self.processed_img_label=QLabel('预览图：',self.img_all_widget)
        if USEDEEPCAMERA:
            self.scroll_area2 = QScrollArea(self.img_all_widget)
            self.scroll_h_layout=QHBoxLayout(self.scroll_area2)
            self.processedimage_label = QLabel()
            self.processedimage_label.setPixmap(QPixmap.fromImage(convertToQtFormat))
            self.processedimage_label.setScaledContents(True)
            self.scroll_h_layout.addWidget(self.processedimage_label)
            self.depthimage_label = QLabel()
            self.depthimage_label.setPixmap(QPixmap.fromImage(convertToQtFormat))
            self.depthimage_label.setScaledContents(True)
            self.scroll_h_layout.addWidget(self.depthimage_label)
            self.scroll_area2.setLayout(self.scroll_h_layout)
        else:
            self.scroll_area2 = QScrollArea(self.img_all_widget) 
            self.processedimage_label = QLabel(self.scroll_area2)
            self.processedimage_label.setPixmap(QPixmap.fromImage(convertToQtFormat))
            self.processedimage_label.setScaledContents(True)
            self.scroll_area2.setWidget(self.processedimage_label)
        self.img_v_layout.addWidget(self.processed_img_label)
        self.img_v_layout.addWidget(self.scroll_area2)
        self.img_v_layout.addStretch(0)
        self.img_all_widget.setLayout(self.img_v_layout)
        self.img_all_scrollArea.setWidget(self.img_all_widget)

        self.set_v_layout=QVBoxLayout()
        self.groupbox_1 = QGroupBox('图像范围设置', self)
        self.groupbox_1.setStyleSheet('background:#EEEEEE')
        self.groupbox_1_vlayout=QVBoxLayout()
        self.zoom_h_layout=QHBoxLayout()
        self.zoom_in_bt=QToolButton(self)
        self.zoom_in_bt.setIcon(QIcon(path+'images/zoom-in.png'))
        self.zoom_in_bt.setShortcut('Ctrl++')
        self.zoom_in_bt.clicked.connect(lambda:self.zoom_func(1.1))
        self.zoom_out_bt=QToolButton(self)
        self.zoom_out_bt.setIcon(QIcon(path+'images/zoom-out.png'))
        self.zoom_out_bt.setShortcut('Ctrl+-')
        self.zoom_out_bt.clicked.connect(lambda:self.zoom_func(0.9))
        self.hflip_bt=QToolButton(self)
        self.hflip_bt.setIcon(QIcon(path+'images/flip-h.png'))
        self.hflip_bt.setCheckable(True)
        self.hflip_bt.setChecked(self.jsonData['hflip'])
        self.vflip_bt=QToolButton(self)
        self.vflip_bt.setIcon(QIcon(path+'images/flip-v.png'))
        self.vflip_bt.setCheckable(True)
        self.vflip_bt.setChecked(self.jsonData['vflip'])
        self.rotate_bt=QToolButton(self)
        self.rotate_bt.setIcon(QIcon(path+'images/rotate.png'))
        self.rotate_bt.setCheckable(True)
        self.rotate_bt.setChecked(self.jsonData['rotate90'])
        self.zoom_h_layout.addWidget(self.zoom_in_bt)
        self.zoom_h_layout.addWidget(self.zoom_out_bt)
        self.zoom_h_layout.addWidget(self.hflip_bt)
        self.zoom_h_layout.addWidget(self.vflip_bt)
        self.zoom_h_layout.addWidget(self.rotate_bt)
        self.groupbox_1_vlayout.addLayout(self.zoom_h_layout)

        self.perspect_bt=QPushButton('框选图像范围',self)
        self.perspect_bt.setStyleSheet('background:#E0E0E0')
        self.perspect_bt.setIcon(QIcon(path+'images/perspective.png'))
        self.perspect_bt.setToolTip('透视变换')
        self.perspect_bt.setStatusTip('透视变换')
        self.perspect_bt.setCheckable(True)
        self.perspect_bt.clicked.connect(self.perspect_func)
        self.groupbox_1_vlayout.addWidget(self.perspect_bt)
        self.confirm_bt=QPushButton('确定',self)
        self.confirm_bt.setStyleSheet('background:#E0E0E0')
        self.confirm_bt.clicked.connect(self.confirm_func)
        self.confirm_bt.hide()
        self.groupbox_1_vlayout.addWidget(self.confirm_bt)
        
        self.points_grid_layout=QGridLayout()
        self.points_grid_layout.addWidget(QLabel('左上点',self),0,0)
        self.points1_x_label=QLabel('x:',self)
        self.points_grid_layout.addWidget(self.points1_x_label,0,1)
        self.points1_x_spinbox=QSpinBox(self)
        self.points1_x_spinbox.setRange(0, 1920)
        self.points1_x_spinbox.setSingleStep(1) 				#设置步长
        self.points1_x_spinbox.setValue(self.jsonData['points'][0][0])
        self.points1_x_spinbox.setStyleSheet('background:#F6F6F6')
        self.points1_x_spinbox.valueChanged.connect(lambda value: self.points_change_func(0,0,value))
        self.points_grid_layout.addWidget(self.points1_x_spinbox,0,2)
        self.points1_y_label=QLabel('y:',self)
        self.points_grid_layout.addWidget(self.points1_y_label,0,3)
        self.points1_y_spinbox=QSpinBox(self)
        self.points1_y_spinbox.setRange(0, 1920)
        self.points1_y_spinbox.setSingleStep(1)
        self.points1_y_spinbox.setValue(self.jsonData['points'][0][1])
        self.points1_y_spinbox.setStyleSheet('background:#F6F6F6')
        self.points1_y_spinbox.valueChanged.connect(lambda value: self.points_change_func(0,1,value))
        self.points_grid_layout.addWidget(self.points1_y_spinbox,0,4)
        self.points_grid_layout.addWidget(QLabel('右上点',self),1,0)
        self.points2_x_label=QLabel('x:',self)
        self.points_grid_layout.addWidget(self.points2_x_label,1,1)
        self.points2_x_spinbox=QSpinBox(self)
        self.points2_x_spinbox.setRange(0, 1920)
        self.points2_x_spinbox.setSingleStep(1) 				#设置步长
        self.points2_x_spinbox.setValue(self.jsonData['points'][1][0])
        self.points2_x_spinbox.setStyleSheet('background:#F6F6F6')
        self.points2_x_spinbox.valueChanged.connect(lambda value: self.points_change_func(1,0,value))
        self.points_grid_layout.addWidget(self.points2_x_spinbox,1,2)
        self.points2_y_label=QLabel('y:',self)
        self.points_grid_layout.addWidget(self.points2_y_label,1,3)
        self.points2_y_spinbox=QSpinBox(self)
        self.points2_y_spinbox.setRange(0, 1920)
        self.points2_y_spinbox.setSingleStep(1)
        self.points2_y_spinbox.setValue(self.jsonData['points'][1][1])
        self.points2_y_spinbox.setStyleSheet('background:#F6F6F6')
        self.points2_y_spinbox.valueChanged.connect(lambda value: self.points_change_func(1,1,value))
        self.points_grid_layout.addWidget(self.points2_y_spinbox,1,4)
        self.points_grid_layout.addWidget(QLabel('左下点',self),2,0)
        self.points3_x_label=QLabel('x:',self)
        self.points_grid_layout.addWidget(self.points3_x_label,2,1)
        self.points3_x_spinbox=QSpinBox(self)
        self.points3_x_spinbox.setRange(0, 1920)
        self.points3_x_spinbox.setSingleStep(1) 				#设置步长
        self.points3_x_spinbox.setValue(self.jsonData['points'][2][0])
        self.points3_x_spinbox.setStyleSheet('background:#F6F6F6')
        self.points3_x_spinbox.valueChanged.connect(lambda value: self.points_change_func(2,0,value))
        self.points_grid_layout.addWidget(self.points3_x_spinbox,2,2)
        self.points3_y_label=QLabel('y:',self)
        self.points_grid_layout.addWidget(self.points3_y_label,2,3)
        self.points3_y_spinbox=QSpinBox(self)
        self.points3_y_spinbox.setRange(0, 1920)
        self.points3_y_spinbox.setSingleStep(1)
        self.points3_y_spinbox.setValue(self.jsonData['points'][2][1])
        self.points3_y_spinbox.setStyleSheet('background:#F6F6F6')
        self.points3_y_spinbox.valueChanged.connect(lambda value: self.points_change_func(2,1,value))
        self.points_grid_layout.addWidget(self.points3_y_spinbox,2,4)
        self.points_grid_layout.addWidget(QLabel('左上点',self),3,0)
        self.points4_x_label=QLabel('x:',self)
        self.points_grid_layout.addWidget(self.points4_x_label,3,1)
        self.points4_x_spinbox=QSpinBox(self)
        self.points4_x_spinbox.setRange(0, 1920)
        self.points4_x_spinbox.setSingleStep(1) 				#设置步长
        self.points4_x_spinbox.setValue(self.jsonData['points'][3][0])
        self.points4_x_spinbox.setStyleSheet('background:#F6F6F6')
        self.points4_x_spinbox.valueChanged.connect(lambda value: self.points_change_func(3,0,value))
        self.points_grid_layout.addWidget(self.points4_x_spinbox,3,2)
        self.points4_y_label=QLabel('y:',self)
        self.points_grid_layout.addWidget(self.points4_y_label,3,3)
        self.points4_y_spinbox=QSpinBox(self)
        self.points4_y_spinbox.setRange(0, 1920)
        self.points4_y_spinbox.setSingleStep(1)
        self.points4_y_spinbox.setValue(self.jsonData['points'][3][1])
        self.points4_y_spinbox.setStyleSheet('background:#F6F6F6')
        self.points4_y_spinbox.valueChanged.connect(lambda value: self.points_change_func(3,1,value))
        self.points_grid_layout.addWidget(self.points4_y_spinbox,3,4)
        self.groupbox_1_vlayout.addLayout(self.points_grid_layout)
        
        self.groupbox_1.setLayout(self.groupbox_1_vlayout)
        self.set_v_layout.addWidget(self.groupbox_1)

        self.groupbox_2 = QGroupBox('输入模型图像大小设置', self)
        self.groupbox_2.setStyleSheet('background:#EEEEEE')
        self.spinbox_h_layout=QHBoxLayout()
        self.imgwidth_label=QLabel('宽：',self)
        self.spinbox_h_layout.addWidget(self.imgwidth_label)
        self.imgwidth_spinbox=QSpinBox(self)
        self.imgwidth_spinbox.setRange(320, 1280)				#设置数值范围
        self.imgwidth_spinbox.setSingleStep(1) 				#设置步长
        self.imgwidth_spinbox.setValue(self.jsonData['width'])
        self.imgwidth_spinbox.setStyleSheet('background:#F6F6F6')
        self.spinbox_h_layout.addWidget(self.imgwidth_spinbox)
        self.spinbox_h_layout.addStretch(0)
        self.imgheight_label=QLabel('高：',self)
        self.spinbox_h_layout.addWidget(self.imgheight_label)
        self.imgheight_spinbox=QSpinBox(self)
        self.imgheight_spinbox.setRange(320, 1280)				#设置数值范围
        self.imgheight_spinbox.setSingleStep(1) 				#设置步长
        self.imgheight_spinbox.setValue(self.jsonData['height'])
        self.imgheight_spinbox.setStyleSheet('background:#F6F6F6')
        self.spinbox_h_layout.addWidget(self.imgheight_spinbox)
        self.groupbox_2.setLayout(self.spinbox_h_layout)
        self.set_v_layout.addWidget(self.groupbox_2)
        self.imgwidth_spinbox.valueChanged.connect(self.wh_change_func)
        self.imgheight_spinbox.valueChanged.connect(self.wh_change_func)

        self.groupbox_3 = QGroupBox('图像预处理', self)
        self.groupbox_3.setStyleSheet('background:#EEEEEE')
        self.process_v_layout=QVBoxLayout()
        self.bilateralFilter_bt=QPushButton('双边滤波',self.groupbox_3)
        self.bilateralFilter_bt.setIcon(QIcon(path+'images/blur.png'))
        self.bilateralFilter_bt.setCheckable(True)
        self.bilateralFilter_bt.setChecked(self.jsonData['fliter'])
        self.process_v_layout.addWidget(self.bilateralFilter_bt)
        self.sharpening_bt=QPushButton('  锐化',self.groupbox_3)
        self.sharpening_bt.setIcon(QIcon(path+'images/sharpen.png'))
        self.sharpening_bt.setCheckable(True)
        self.sharpening_bt.setChecked(self.jsonData['sharpening'])
        self.process_v_layout.addWidget(self.sharpening_bt)
        self.equalize_bt=QPushButton('CLAHE均衡化',self.groupbox_3)
        self.equalize_bt.setIcon(QIcon(path+'images/equalizer.png'))
        self.equalize_bt.setCheckable(True)
        self.equalize_bt.setChecked(self.jsonData['CLAHE'])
        self.process_v_layout.addWidget(self.equalize_bt)
        self.groupbox_3.setLayout(self.process_v_layout)
        self.set_v_layout.addWidget(self.groupbox_3)

        self.groupbox_4 = QGroupBox('运动速率阈值设置', self)
        self.groupbox_4.setStyleSheet('background:#EEEEEE')
        self.spinbox2_h_layout=QHBoxLayout()
        self.detect_threshold_label=QLabel('多垃圾：',self)
        self.spinbox2_h_layout.addWidget(self.detect_threshold_label)
        self.detect_threshold_spinbox=QDoubleSpinBox(self)
        self.detect_threshold_spinbox.setRange(1, 500000)				#设置数值范围
        self.detect_threshold_spinbox.setSingleStep(10) 				#设置步长
        self.detect_threshold_spinbox.setValue(self.jsonData['detect_threshold'])
        self.detect_threshold_spinbox.setStyleSheet('background:#F6F6F6')
        self.spinbox2_h_layout.addWidget(self.detect_threshold_spinbox)
        self.spinbox2_h_layout.addStretch(0)
        self.catch_threshold_label=QLabel('单垃圾：',self)
        self.spinbox2_h_layout.addWidget(self.catch_threshold_label)
        self.catch_threshold_spinbox=QDoubleSpinBox(self)
        self.catch_threshold_spinbox.setRange(1, 500000)				#设置数值范围
        self.catch_threshold_spinbox.setSingleStep(10) 				#设置步长
        self.catch_threshold_spinbox.setValue(self.jsonData['catch_threshold'])
        self.catch_threshold_spinbox.setStyleSheet('background:#F6F6F6')
        self.spinbox2_h_layout.addWidget(self.catch_threshold_spinbox)
        self.groupbox_4.setLayout(self.spinbox2_h_layout)
        self.set_v_layout.addWidget(self.groupbox_4)

        self.groupbox_5 = QGroupBox('坐标换算', self)
        self.groupbox_5.setStyleSheet('background:#EEEEEE')
        self.range_grid_layout=QGridLayout()
        self.target_zone_label=QLabel('目标范围',self)
        self.range_grid_layout.addWidget(self.target_zone_label,0,0)
        self.axis_width_label=QLabel('宽:',self)
        self.range_grid_layout.addWidget(self.axis_width_label,0,1)
        self.axiswidth_spinbox=QSpinBox(self)
        self.axiswidth_spinbox.setRange(100, 40000)				#设置数值范围
        self.axiswidth_spinbox.setSingleStep(10) 				#设置步长
        self.axiswidth_spinbox.setValue(self.jsonData['axis-width'])
        self.axiswidth_spinbox.setStyleSheet('background:#F6F6F6')
        self.axiswidth_spinbox.valueChanged.connect(lambda value: self.motor_wh_change(0,value))

        self.range_grid_layout.addWidget(self.axiswidth_spinbox,0,2)
        self.axisheight_label=QLabel('高:',self)
        self.range_grid_layout.addWidget(self.axisheight_label,0,3)
        self.axisheight_spinbox=QSpinBox(self)
        self.axisheight_spinbox.setRange(100, 40000)				#设置数值范围
        self.axisheight_spinbox.setSingleStep(10) 				#设置步长
        self.axisheight_spinbox.setValue(self.jsonData['axis-height'])
        self.axisheight_spinbox.setStyleSheet('background:#F6F6F6')
        self.axisheight_spinbox.valueChanged.connect(lambda value: self.motor_wh_change(1,value))
        self.range_grid_layout.addWidget(self.axisheight_spinbox,0,4)
        
        self.point1_label=QLabel('左上点',self)
        self.range_grid_layout.addWidget(self.point1_label,1,0)
        self.point1_x_label=QLabel('x:',self)
        self.range_grid_layout.addWidget(self.point1_x_label,1,1)
        self.point1_x_spinbox=QSpinBox(self)
        self.point1_x_spinbox.setRange(-10, self.jsonData['width']+9)
        self.point1_x_spinbox.setSingleStep(1) 				#设置步长
        self.point1_x_spinbox.setValue(self.jsonData['axis-x1'])
        self.point1_x_spinbox.setStyleSheet('background:#F6F6F6')
        self.range_grid_layout.addWidget(self.point1_x_spinbox,1,2)
        self.point1_y_label=QLabel('y:',self)
        self.range_grid_layout.addWidget(self.point1_y_label,1,3)
        self.point1_y_spinbox=QSpinBox(self)
        self.point1_y_spinbox.setRange(-10, self.jsonData['height']+9)
        self.point1_y_spinbox.setSingleStep(1)
        self.point1_y_spinbox.setValue(self.jsonData['axis-y1'])
        self.point1_y_spinbox.setStyleSheet('background:#F6F6F6')
        self.range_grid_layout.addWidget(self.point1_y_spinbox,1,4)
        
        self.point2_label=QLabel('右下点',self)
        self.range_grid_layout.addWidget(self.point2_label,2,0)
        self.point2_x_label=QLabel('x:',self)
        self.range_grid_layout.addWidget(self.point2_x_label,2,1)
        self.point2_x_spinbox=QSpinBox(self)
        self.point2_x_spinbox.setRange(0, self.jsonData['width'])
        self.point2_x_spinbox.setSingleStep(1) 				#设置步长
        self.point2_x_spinbox.setValue(self.jsonData['axis-x2'])
        self.point2_x_spinbox.setStyleSheet('background:#F6F6F6')
        self.range_grid_layout.addWidget(self.point2_x_spinbox,2,2)
        self.point2_y_label=QLabel('y:',self)
        self.range_grid_layout.addWidget(self.point2_y_label,2,3)
        self.point2_y_spinbox=QSpinBox(self)
        self.point2_y_spinbox.setRange(0, self.jsonData['height'])
        self.point2_y_spinbox.setSingleStep(1)
        self.point2_y_spinbox.setValue(self.jsonData['axis-y2'])
        self.point2_y_spinbox.setStyleSheet('background:#F6F6F6')
        self.range_grid_layout.addWidget(self.point2_y_spinbox,2,4)

        self.groupbox_5.setLayout(self.range_grid_layout)
        self.set_v_layout.addWidget(self.groupbox_5)
        self.groupbox_6 = QGroupBox('满载检测', self)
        self.groupbox_6.setStyleSheet('background:#EEEEEE')
        self.full_grid_layout=QGridLayout()
        self.full_grid_layout.addWidget(QLabel('距离阈值',self),0,0)
        self.full_distance_spinbox=QDoubleSpinBox(self)
        self.full_distance_spinbox.setRange(0, 200)
        self.full_distance_spinbox.setSingleStep(1)
        self.full_distance_spinbox.setValue(self.jsonData['full_distance'])
        self.full_distance_spinbox.setStyleSheet('background:#F6F6F6')
        self.full_grid_layout.addWidget(self.full_distance_spinbox,0,1)
        self.full_distance_m_spinbox=QDoubleSpinBox(self)
        self.full_distance_m_spinbox.setRange(0, 200)
        self.full_distance_m_spinbox.setSingleStep(1)
        self.full_distance_m_spinbox.setValue(self.jsonData['full_distance']+15)
        self.full_distance_m_spinbox.setStyleSheet('background:#F6F6F6')
        self.full_grid_layout.addWidget(self.full_distance_m_spinbox,0,2)
        self.full_grid_layout.addWidget(QLabel('时间阈值',self),0,3)
        self.full_time_spinbox=QDoubleSpinBox(self)
        self.full_time_spinbox.setRange(0, 5)
        self.full_time_spinbox.setSingleStep(0.1)
        self.full_time_spinbox.setValue(self.jsonData['full_time'])
        self.full_time_spinbox.setStyleSheet('background:#F6F6F6')
        self.full_grid_layout.addWidget(self.full_time_spinbox,0,4)
        self.full_showdistance=QLabel('-',self),QLabel('-',self),QLabel('-',self),QLabel('-',self)
        self.full_grid_layout.addWidget(self.full_showdistance[0],1,0)
        self.full_grid_layout.addWidget(self.full_showdistance[1],1,1)
        self.full_grid_layout.addWidget(self.full_showdistance[2],1,2)
        self.full_grid_layout.addWidget(self.full_showdistance[3],1,3)
        self.groupbox_6.setLayout(self.full_grid_layout)
        self.set_v_layout.addWidget(self.groupbox_6)
        self.depth_label=QLabel('深度：NA',self)
        self.set_v_layout.addWidget(self.depth_label)
        self.set_v_layout.addStretch(0)
    
        self.tab2_h_layout=QHBoxLayout()
        self.tab2_h_layout.addWidget(self.img_all_scrollArea)
        self.tab2_h_layout.addLayout(self.set_v_layout)
        self.tab2_h_layout.setStretch(0,8)
        self.tab2_h_layout.setStretch(1,2)
        self.tab2_widget=QWidget(self)
        self.tab2_widget.setLayout(self.tab2_h_layout)
#==============================================================================  Tab3↓
        self.tab3_widget=QWidget(self)
        self.tab3_widget.setStyleSheet('font-size: 15pt;')
        self.grid_layout = QGridLayout()
        self.grid_layout.addWidget(QLabel('电机名称'), 0, 0, 1, 1,Qt.AlignCenter)
        self.grid_layout.addWidget(QLabel('当前位置'), 0, 1, 1, 1,Qt.AlignCenter)
        self.grid_layout.addWidget(QLabel('设置位置'), 0, 2, 1, 1,Qt.AlignCenter)
        self.grid_layout.addWidget(QLabel('微调/指令'), 0, 3, 1, 1,Qt.AlignCenter)

        #x,y电机↓
        self.grid_layout.addWidget(QLabel('电机x'), 1, 0, 1, 1,Qt.AlignCenter)
        self.grid_layout.addWidget(QLabel('电机y'), 2, 0, 1, 1,Qt.AlignCenter)
        self.motorx_value_label=QLabel(str(1))
        self.motory_value_label=QLabel(str(1))
        self.grid_layout.addWidget(self.motorx_value_label,1,1,1,1,Qt.AlignCenter)
        self.grid_layout.addWidget(self.motory_value_label,2,1,1,1,Qt.AlignCenter)
        self.motorx_spinbox=QSpinBox(self)
        self.motory_spinbox=QSpinBox(self)
        self.motorx_spinbox.setRange(1,40000)
        self.motorx_spinbox.setSingleStep(100)
        self.motory_spinbox.setRange(1,40000)
        self.motory_spinbox.setSingleStep(100)
        self.motorx_spinbox.setValue(1)
        self.motory_spinbox.setValue(1)
        self.grid_layout.addWidget(self.motorx_spinbox,1,2,1,1,Qt.AlignCenter)
        self.grid_layout.addWidget(self.motory_spinbox,2,2,1,1,Qt.AlignCenter)
        self.motorx_confirm_bt=QPushButton('确认',self)
        self.motory_confirm_bt=QPushButton('确认',self)
        self.motorx_confirm_bt.clicked.connect(lambda: self.motor_confirm_func(0))
        self.motory_confirm_bt.clicked.connect(lambda: self.motor_confirm_func(1))
        self.grid_layout.addWidget(self.motorx_confirm_bt,1,3,1,1,Qt.AlignCenter)
        self.grid_layout.addWidget(self.motory_confirm_bt,2,3,1,1,Qt.AlignCenter)

        #气动↓
        # self.grid_layout.addWidget(QLabel('气动'), 3, 0, 1, 1,Qt.AlignCenter)
        # self.pneumatic_value_label=QLabel('收回')
        # self.grid_layout.addWidget(self.pneumatic_value_label,3,1,1,1,Qt.AlignCenter)
        # self.pneumatic_stretch_bt=QPushButton('伸出',self)
        # self.pneumatic_retract_bt=QPushButton('收回',self)
        # self.pneumatic_stretch_bt.clicked.connect(lambda: self.pneumatic_bt_func(0))
        # self.pneumatic_retract_bt.clicked.connect(lambda: self.pneumatic_bt_func(1))
        # self.pneumatic_bt_HLayout=QHBoxLayout()
        # self.pneumatic_bt_HLayout.addWidget(self.pneumatic_stretch_bt)
        # self.pneumatic_bt_HLayout.addWidget(self.pneumatic_retract_bt)
        # self.grid_layout.addLayout(self.pneumatic_bt_HLayout,3,2,1,1,Qt.AlignCenter)

        #爪子上下↓
        self.grid_layout.addWidget(QLabel('爪子上下'), 3, 0, 1, 1,Qt.AlignCenter)
        self.updown_value_label=QLabel(f"3950")
        self.grid_layout.addWidget(self.updown_value_label,3,1,1,1,Qt.AlignCenter)
        self.updown_slider=QSlider(Qt.Horizontal,self)
        self.updown_slider.setRange(0,4095)
        self.updown_slider.setValue(3950)
        self.updown_slider.setFixedWidth(360)
        self.updown_slider.valueChanged.connect(lambda value: self.slider_func(2,value))
        self.grid_layout.addWidget(self.updown_slider,3,2,1,1,Qt.AlignCenter)
        self.updown_spinbox=QSpinBox(self)
        self.updown_spinbox.setRange(-4000,4000)
        self.updown_spinbox.setValue(0)
        self.updown_spinbox.valueChanged.connect(lambda value: self.spinbox_func(2,value))
        self.grid_layout.addWidget(self.updown_spinbox,3,3,1,1,Qt.AlignCenter)
        #爪子旋转↓
        self.grid_layout.addWidget(QLabel('爪子旋转'), 4, 0, 1, 1,Qt.AlignCenter)
        self.rotate_value_label=QLabel(f"0+{self.jsonData['rotate']}")
        self.grid_layout.addWidget(self.rotate_value_label,4,1,1,1,Qt.AlignCenter)
        self.rotate_slider=QSlider(Qt.Horizontal,self)
        self.rotate_slider.setRange(-90,90)
        self.rotate_slider.setValue(0)
        self.rotate_slider.setFixedWidth(360)
        self.rotate_slider.valueChanged.connect(lambda value: self.slider_func(0,value))
        self.grid_layout.addWidget(self.rotate_slider,4,2,1,1,Qt.AlignCenter)
        self.rotate_spinbox=QSpinBox(self)
        self.rotate_spinbox.setRange(-500,500)
        self.rotate_spinbox.setValue(self.jsonData['rotate'])
        self.rotate_spinbox.valueChanged.connect(lambda value: self.spinbox_func(0,value))
        self.grid_layout.addWidget(self.rotate_spinbox,4,3,1,1,Qt.AlignCenter)

        #四周挡板↓
        self.grid_layout.addWidget(QLabel('挡板'), 5, 0, 1, 1,Qt.AlignCenter)
        self.around_value_label=QLabel('关闭')
        self.grid_layout.addWidget(self.around_value_label,5,1,1,1,Qt.AlignCenter)
        self.around_open_bt=QPushButton('打开',self)
        self.around_close_bt=QPushButton('关闭',self)
        self.around_open_bt.clicked.connect(lambda: self.around_bt_func(1))
        self.around_close_bt.clicked.connect(lambda: self.around_bt_func(0))
        self.around_bt_HLayout=QHBoxLayout()
        self.around_bt_HLayout.addWidget(self.around_open_bt)
        self.around_bt_HLayout.addWidget(self.around_close_bt)
        self.grid_layout.addLayout(self.around_bt_HLayout,5,2,1,1,Qt.AlignCenter)
        
        #平台旋转↓
        self.grid_layout.addWidget(QLabel('平台旋转'), 6, 0, 1, 1,Qt.AlignCenter)
        self.platformR_value_label=QLabel(f"0+{self.jsonData['platformR']}")
        self.grid_layout.addWidget(self.platformR_value_label,6,1,1,1,Qt.AlignCenter)
        self.platformR_slider=QSlider(Qt.Horizontal,self)
        self.platformR_slider.setRange(0,4096)
        self.platformR_slider.setValue(2048)
        self.platformR_slider.setFixedWidth(360)
        self.platformR_slider.valueChanged.connect(lambda value: self.slider_func(1,value))
        self.grid_layout.addWidget(self.platformR_slider,6,2,1,1,Qt.AlignCenter)
        self.platformR_spinbox=QSpinBox(self)
        self.platformR_spinbox.setRange(-1000,1000)
        self.platformR_spinbox.setValue(self.jsonData['platformR'])
        self.platformR_spinbox.valueChanged.connect(lambda value: self.spinbox_func(1,value))
        self.grid_layout.addWidget(self.platformR_spinbox,6,3,1,1,Qt.AlignCenter)
        
        self.tab3_widget.setLayout(self.grid_layout)
#==============================================================================
        self.tab=QTabWidget(self)
        self.tab.addTab(self.splitter,QIcon(path+'images/trash.png'),'状态')
        self.tab.addTab(self.tab2_widget,QIcon(path+'images/setting.png'),'图像设置')
        self.tab.addTab(self.tab3_widget,QIcon(path+'images/servo.png'),'电机设置')
        self.tab.tabBar().currentChanged.connect(self.tabChange)
        self.setCentralWidget(self.tab)

    def tabChange(self,index):
        if COMPUTERTEST and index==2:
            self.tab.setCurrentIndex(0)
        print(f"Tab changed to index {index}")
    
    def wh_change_func(self):
        self.point1_x_spinbox.setRange(-10,self.imgwidth_spinbox.value()+9)
        self.point1_y_spinbox.setRange(-10,self.imgheight_spinbox.value()+9)
        self.point2_x_spinbox.setRange(-10,self.imgwidth_spinbox.value()+9)
        self.point2_y_spinbox.setRange(-10,self.imgheight_spinbox.value()+9)

    def perspect_func(self):#按下透视变换的按钮
        self.is_perspect=not self.is_perspect
        if self.is_perspect:
            self.status_bar.showMessage('透视变换')
        else:
            self.status_bar.showMessage('退出透视变换')
            self.confirm_bt.hide()
            self.points.clear()

    def confirm_func(self):
        self.is_perspect=False
        self.perspect_bt.setChecked(False)
        self.confirm_bt.hide()
        if len(self.points)==4:
            self.jsonData['points']=self.points.copy()
            self.status_bar.showMessage('透视变换完成')
            self.points1_x_spinbox.setValue(self.jsonData['points'][0][0])
            self.points1_y_spinbox.setValue(self.jsonData['points'][0][1])
            self.points2_x_spinbox.setValue(self.jsonData['points'][1][0])
            self.points2_y_spinbox.setValue(self.jsonData['points'][1][1])
            self.points3_x_spinbox.setValue(self.jsonData['points'][2][0])
            self.points3_y_spinbox.setValue(self.jsonData['points'][2][1])
            self.points4_x_spinbox.setValue(self.jsonData['points'][3][0])
            self.points4_y_spinbox.setValue(self.jsonData['points'][3][1])
        self.points.clear()
    
    def points_change_func(self,index,xy,val):
        self.jsonData['points'][index][xy]=val

    def motor_confirm_func(self,index):
        if index==0:
            if not COMPUTERTEST and self.my_thread.moveThread.move==False:
                motors.goto_x_position(self.motorx_spinbox.value())
            self.motorx_value_label.setText(str(self.motorx_spinbox.value()))
        elif index==1:
            if not COMPUTERTEST and self.my_thread.moveThread.move==False:
                motors.goto_y_position(self.motory_spinbox.value())
            self.motory_value_label.setText(str(self.motory_spinbox.value()))
        
    def pneumatic_bt_func(self,val):
        if val==0:
            self.pneumatic_value_label.setText('收回')
            if not COMPUTERTEST and self.my_thread.moveThread.move==False:
                data1_to_send = bytes.fromhex('01 05 00 A0 FF 00 8C 18')          #一路开
                motors.ser3.write(data1_to_send)
                time.sleep(0.02)
        elif val==1:
            self.pneumatic_value_label.setText('伸出')
            if not COMPUTERTEST and self.my_thread.moveThread.move==False:
                data2_to_send = bytes.fromhex('01 05 00 A0 00 00 CD E8')          #一路关
                motors.ser3.write(data2_to_send)
                time.sleep(0.02)

    def around_bt_func(self,val):
        if val and not COMPUTERTEST:
            self.around_value_label.setText('打开')
            motors.open_around()
        elif not COMPUTERTEST:
            self.around_value_label.setText('关闭')
            motors.close_around()

    def motor_wh_change(self,index,val):
        if index==0:
            if not COMPUTERTEST and self.my_thread.moveThread.move==False:
                motors.axis_x=val
            elif not COMPUTERTEST:self.axiswidth_spinbox.setValue(motors.axis_x)
        elif index==1:
            if not COMPUTERTEST and self.my_thread.moveThread.move==False:
                motors.axis_y=val
            elif not COMPUTERTEST:self.axisheight_spinbox.setValue(motors.axis_y)
    
    def slider_func(self,index,value):
        if index==0:
            if not COMPUTERTEST and self.my_thread.moveThread.move==False:
                motors.servo_2_rotate(value)
                self.rotate_value_label.setText(str(value)+'+'+str(self.rotate_spinbox.value()))
        elif index==1:
            if not COMPUTERTEST and self.my_thread.moveThread.move==False:
                if self.around_value_label.text()=='关闭':
                    choice=QMessageBox.warning(self,'警告','四周挡板可能还为打开，是否确认调整平台？',QMessageBox.Yes|QMessageBox.No)
                    if choice==QMessageBox.Yes:
                        motors.servo_sport(7,value,50,50)
                        self.platformR_value_label.setText(str(value)+'+'+str(self.platformR_spinbox.value()))
                else:
                    motors.servo_sport(7,value,50,50)
                    self.platformR_value_label.setText(str(value)+'+'+str(self.platformR_spinbox.value()))
        elif index==2:
            if not COMPUTERTEST and self.my_thread.moveThread.move==False:
                motors.servo_sport(3,value,90,250)    
                self.updown_value_label.setText(str(value)+'+'+str(self.rotate_spinbox.value()))
        print(index,value)
            
    def spinbox_func(self,index,value):
        if index==0:
            if not COMPUTERTEST and self.my_thread.moveThread.move==False:
                motors.rotate=value
                motors.servo_2_rotate(self.rotate_slider.value())
                self.rotate_value_label.setText(str(self.rotate_slider.value())+'+'+str(value))
        elif index==1:
            if not COMPUTERTEST and self.my_thread.moveThread.move==False:
                motors.platformR=value
                motors.servo_sport(7,self.platformR_slider.value(),50,50)
                self.platformR_value_label.setText(str(self.platformR_slider.value())+'+'+str(value))
                
    def zoom_func(self,ratio):
        if self.tab.currentIndex()==1:
            self.zoom_show*=ratio
            if self.zoom_show<0.2:
                self.zoom_show=0.2
            elif self.zoom_show>5:
                self.zoom_show=5
            self.status_bar.showMessage(f'缩放  {self.zoom_show*100:.1f}%')

    def debug_func(self):
        self.my_thread.anyNew=True
        random.seed(QTime.currentTime().second())
        x,y,angle=random.randint(0,self.imgwidth_spinbox.value()-1),random.randint(0,self.imgheight_spinbox.value()-1),random.randint(-90,90)
        axis_x=(x-main_window.point1_x_spinbox.value())*main_window.axiswidth_spinbox.value()//(main_window.point2_x_spinbox.value()-main_window.point1_x_spinbox.value())
        axis_y=(y-main_window.point1_y_spinbox.value())*main_window.axisheight_spinbox.value()//(main_window.point2_y_spinbox.value()-main_window.point1_y_spinbox.value())
        self.newItem_func('测试',1,x,y,angle,axis_x,axis_y,30,True,300.0)
        self.my_thread.moveThread.x,self.my_thread.moveThread.y,self.my_thread.moveThread.angle,self.my_thread.moveThread.k,self.my_thread.moveThread.num=axis_y,axis_x,angle,random.randint(1,4),2
        self.my_thread.moveThread.width=20
        self.my_thread.moveThread.move=True
        print('set the move true')
        
    def clear_func(self):
        self.model.clear()
        if SHOWTIMEANDNAME:
            self.model.setHorizontalHeaderLabels(['名称','类型','数量','状态','时间'])
        else:
            self.model.setHorizontalHeaderLabels(['序号','垃圾种类','数量','分类成功与否'])
        self.sum_others.setText(str(0))
        self.sum_chicken.setText(str(0))
        self.sum_harmful.setText(str(0))
        self.sum_recycle.setText(str(0))
        self.tableDetails.clear()

    def keeploop(self,succeed):#按下按钮才继续检查(显示信息)
        if self.model.rowCount()>0 and succeed:
            if SHOWTIMEANDNAME:
                self.model.setItem(self.model.rowCount()-1,3,QStandardItem('存储正确'))
                self.model.setItem(self.model.rowCount()-1,4,QStandardItem(QTime.currentTime().toString('hh:mm:ss')))
                self.model.item(self.model.rowCount()-1,2).setTextAlignment(Qt.AlignCenter)
                self.model.item(self.model.rowCount()-1,3).setTextAlignment(Qt.AlignCenter)
            else:
                self.model.setItem(self.model.rowCount()-1,3,QStandardItem('存储正确'))
                self.model.item(self.model.rowCount()-1,3).setTextAlignment(Qt.AlignCenter)
            rubbishCount=dict()
            for i in range(self.model.rowCount()):
                rubbishCount[self.model.item(i,1).text()]=rubbishCount.get(self.model.item(i,1).text(),0)+1
            
            if self.my_thread.anyNew==True:
                if self.model.item(self.model.rowCount()-1,1).text()=='其他垃圾':
                    self.sum_others.setText(str(int(self.sum_others.text())+1))
                if self.model.item(self.model.rowCount()-1,1).text()=='厨余垃圾':
                    self.sum_chicken.setText(str(int(self.sum_chicken.text())+1))
                if self.model.item(self.model.rowCount()-1,1).text()=='有害垃圾':
                    self.sum_harmful.setText(str(int(self.sum_harmful.text())+1))
                if self.model.item(self.model.rowCount()-1,1).text()=='可回收垃圾':
                    self.sum_recycle.setText(str(int(self.sum_recycle.text())+1))
        #self.status.setText('状态：\n检测中')
        self.status_bar.showMessage('检测中')
        self.my_thread.anyNew=False
        
    def resetImg(self,image,anyNew,subval):#更新tab1的图片
        self.image_label.setPixmap(QPixmap.fromImage(image))
        #if not self.settingPage.isHidden():
            #self.settingPage.image_label.setPixmap(QPixmap.fromImage(image))
        
        self.move_status.setText(f'运动速率{int(subval)}') 

    def resetImg2(self,image):#更新tab2的图片
        if self.hflip_bt.isChecked():
            cv2.flip(image,1,image)
        if self.vflip_bt.isChecked():
            cv2.flip(image,0,image)
        if self.rotate_bt.isChecked():
            image=cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.mousePoint:
            cv2.circle(image,self.mousePoint,4,(0,255,0),-1,16)
        for point in self.points:
            cv2.circle(image,point,3,(255,0,0),-1,16)
        if len(self.points)>1:
            cv2.polylines(image,[np.array(self.points)],True,(0,255,0),1,16)
        for point in self.jsonData['points']:
            cv2.circle(image,point,3,(255,255,255),-1,16)
        if len(self.jsonData['points'])==4:
            cv2.polylines(image,[np.array(self.jsonData['points'])],True,(255,0,0),1,16)
        convertToQtFormat = QImage(image,image.shape[1],image.shape[0],3*image.shape[1], QImage.Format_RGB888)
        self.image_label2.setPixmap(QPixmap.fromImage(convertToQtFormat))
        self.image_label2.resize(int(convertToQtFormat.width()*self.zoom_show),int(convertToQtFormat.height()*self.zoom_show))
        self.scroll_area.setFixedHeight(int(convertToQtFormat.height()*self.zoom_show)+5)
        self.scroll_area.setFixedWidth(min(int(convertToQtFormat.width()*self.zoom_show)+5,int(self.centralWidget().width()*0.77)))
    
    def resetImg3(self,image):
        cv2.circle(image,(self.point1_x_spinbox.value(),self.point1_y_spinbox.value()),5,(255,255,0),-1)
        cv2.circle(image,(self.point2_x_spinbox.value(),self.point2_y_spinbox.value()),5,(255,255,0),-1)
        convertToQtFormat = QImage(image,image.shape[1],image.shape[0],3*image.shape[1], QImage.Format_RGB888)
        self.processedimage_label.setPixmap(QPixmap.fromImage(convertToQtFormat))
        self.processedimage_label.resize(int(convertToQtFormat.width()*self.zoom_show),int(convertToQtFormat.height()*self.zoom_show))
        self.scroll_area2.setFixedHeight(int(convertToQtFormat.height()*self.zoom_show)+5)
        self.scroll_area2.setFixedWidth(min(int(convertToQtFormat.width()*2*self.zoom_show)+5,int(self.centralWidget().width()*0.77)))

    def resetDeepImg(self,image,mdepth):
        convertToQtFormat = QImage(image,image.shape[1],image.shape[0],image.shape[1], QImage.Format_Grayscale8)
        self.depthimage_label.setPixmap(QPixmap.fromImage(convertToQtFormat))
        self.depthimage_label.resize(int(convertToQtFormat.width()*self.zoom_show),int(convertToQtFormat.height()*self.zoom_show))
        self.depth_label.setText(f'深度:{mdepth:.2f}mm')
    
    def updata_move_velocity_to_tab2(self,sub):
        if sub!=0:
            self.groupbox_4.setTitle(f'运动速率阈值设置({sub:.2f})')

    def newItem_func(self,itemName,num,x,y,angle,axis_x,axis_y,width,success,depth=None):#从线程传入检测到的东西名称
        if success or len(self.tableDetails)==0:
            if SHOWTIMEANDNAME:
                self.item_list=[QStandardItem(itemName),QStandardItem(self.rubbishDict.get(itemName,'其他垃圾')),QStandardItem(f'{num}'),QStandardItem('等待处理'),QStandardItem(QTime.currentTime().toString('hh:mm:ss'))]#添加到表格
            else:
                self.item_list=[QStandardItem(f'{self.model.rowCount()+1}'),QStandardItem(self.rubbishDict.get(itemName,'其他垃圾')),QStandardItem(f'{num}'),QStandardItem('等待处理')]#添加到表格
            self.model.appendRow(self.item_list)
            self.table.scrollToBottom()
            self.tableDetails.append(f"{QTime.currentTime().toString('hh:mm:ss')}\n名称：{itemName}\n坐标:{x},{y}\n{axis_x},{axis_y}\n角度：{angle:.2f}\n宽度：{width}\n深度：{int(depth)}mm\n")#保存详细信息
        else:
            last_row = self.model.rowCount() - 1
            if SHOWTIMEANDNAME:
                self.model.setItem(last_row, 0, QStandardItem(itemName))
                self.model.setItem(last_row, 1, QStandardItem(self.rubbishDict.get(itemName, '其他垃圾')))
                self.model.setItem(last_row, 2, QStandardItem(f'{num}'))
                self.model.setItem(last_row, 3, QStandardItem('等待处理'))
                self.model.setItem(last_row, 4, QStandardItem(QTime.currentTime().toString('hh:mm:ss')))
            else:
                self.model.setItem(last_row, 0, QStandardItem(f'{self.model.rowCount()}'))
                self.model.setItem(last_row, 1, QStandardItem(self.rubbishDict.get(itemName, '其他垃圾')))
                self.model.setItem(last_row, 2, QStandardItem(f'{num}'))
                self.model.setItem(last_row, 3, QStandardItem('等待处理'))
            self.tableDetails.append(self.tableDetails.pop()+f"\nretry:\n{QTime.currentTime().toString('hh:mm:ss')}\n名称：{itemName}\n坐标:{x},{y}\n{axis_x},{axis_y}\n角度：{angle:.2f}\n宽度：{width}\n深度：{int(depth)}mm\n")
        self.model.item(self.model.rowCount()-1,2).setTextAlignment(Qt.AlignCenter)
        self.model.item(self.model.rowCount()-1,1).setTextAlignment(Qt.AlignCenter)
        self.model.item(self.model.rowCount()-1,0).setTextAlignment(Qt.AlignCenter)
        
    def showDetails(self,row=-1):
        if row<0 or row>=len(self.tableDetails):
            self.status_bar.showMessage(f'行坐标错误 {row}')
            print('行坐标错误:',row)
            return
        self.status.setText(f'第{row+1}行:\n{self.tableDetails[row]}')
        self.status_bar.showMessage(self.tableDetails[row])
        self.table.selectRow(row)

    def pause_play(self):#开始与暂停
        if self.isplay:
            self.isplay=False
            self.my_thread.isupdate=False
            self.pause_action.setIcon(QIcon(path+'images/play.png'))
            self.pause_action.setText('开始')
            self.pause_action.setStatusTip('开始识别')
        else:
            self.isplay=True
            self.my_thread.isupdate=True
            self.pause_action.setIcon(QIcon(path+'images/pause.png'))
            self.pause_action.setText('暂停')
            self.pause_action.setStatusTip('暂停识别')
    
    def fullscreenornot(self):
        print('F11',self.fullScreen_action.isChecked())
        if self.fullScreen_action.isChecked():
            self.fullScreen_action.setText('进入全屏')
            self.showMaximized()
            self.fullScreen_action.setIcon(QIcon(path+"images/fullscreen.png"))
        else:
            self.fullScreen_action.setText('退出全屏')
            self.showFullScreen()
            self.fullScreen_action.setIcon(QIcon(path+"images/exit-fullscreen.png"))
        print(self.fullScreen_action.isChecked())
    
    def exit(self):
        self.my_thread.moveThread.loop=False
        self.my_thread.readCamera.loop=False
        self.my_thread.loop=False
        self.my_thread.quit()
        self.jsonData['vflip']=self.vflip_bt.isChecked()
        self.jsonData['hflip']=self.hflip_bt.isChecked()
        self.jsonData['rotate90']=self.rotate_bt.isChecked()
        self.jsonData['CLAHE']=self.equalize_bt.isChecked()
        self.jsonData['sharpening']=self.sharpening_bt.isChecked()
        self.jsonData['fliter']=self.bilateralFilter_bt.isChecked()
        self.jsonData['width']=self.imgwidth_spinbox.value()
        self.jsonData['height']=self.imgheight_spinbox.value()
        self.jsonData['full_distance']=self.full_distance_spinbox.value()
        self.jsonData['full_time']=self.full_time_spinbox.value()
        self.jsonData['detect_threshold']=self.detect_threshold_spinbox.value()
        self.jsonData['catch_threshold']=self.catch_threshold_spinbox.value()
        self.jsonData['axis-width']=self.axiswidth_spinbox.value()
        self.jsonData['axis-height']=self.axisheight_spinbox.value()
        self.jsonData['axis-x1']=self.point1_x_spinbox.value()
        self.jsonData['axis-y1']=self.point1_y_spinbox.value()
        self.jsonData['axis-x2']=self.point2_x_spinbox.value()
        self.jsonData['axis-y2']=self.point2_y_spinbox.value()
        #self.jsonData['z']=True if self.pneumatic_value_label.text()=='伸出' else False
        self.jsonData['rotate']=self.rotate_spinbox.value()
        self.jsonData['platformR']=self.platformR_spinbox.value()
        fp=open(path+'data.json','w')
        json.dump(self.jsonData,fp,ensure_ascii=False,indent=4)
        fp.close()
        self.close()
        demo.media_player.stop()
        demo.close()
        self.my_thread.wait()
        if not COMPUTERTEST:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(LED, GPIO.OUT)
            GPIO.output(LED,0)
            GPIO.cleanup()
            motors.ser_close()
        exit(0)
    
    def toSettingPage(self):
        self.settingPage.show()
        
    def closeEvent(self, QCloseEvent):
        demo.showFullScreen()
        self.my_signal.emit()
        QCloseEvent.accept()

    def mousePressEvent(self, QMouseEvent): 
        if not self.tab.currentIndex()==1:return
        x = round((QMouseEvent.x()-40+self.scroll_area.horizontalScrollBar().value())/self.zoom_show)
        y = round((QMouseEvent.y()-201+self.scroll_area.verticalScrollBar().value()+self.img_all_scrollArea.verticalScrollBar().value())/self.zoom_show)
        if x<0:x=0
        if y<0:y=0
        print('xy',x,y)
        self.mousePoint=(x,y)

    def mouseMoveEvent(self,QMouseEvent):
        if not self.mousePoint:return
        x = round((QMouseEvent.x()-40+self.scroll_area.horizontalScrollBar().value())/self.zoom_show)
        y = round((QMouseEvent.y()-201+self.scroll_area.verticalScrollBar().value()+self.img_all_scrollArea.verticalScrollBar().value())/self.zoom_show)
        if x<0:x=0
        if y<0:y=0
        print('xy',x,y)
        self.mousePoint=(x,y)

    def mouseReleaseEvent(self, QMouseEvent): 
        self.mousePoint=None

    def wheelEvent(self,wheelEvent):
        if self.control_key:
            if wheelEvent.angleDelta().y() > 0:
                self.zoom_func(1.1)
            else:
                self.zoom_func(0.9)

    def mouseDoubleClickEvent(self, QMouseEvent):
        x = round((QMouseEvent.x()-40+self.scroll_area.horizontalScrollBar().value())/self.zoom_show)
        y = round((QMouseEvent.y()-201+self.scroll_area.verticalScrollBar().value()+self.img_all_scrollArea.verticalScrollBar().value())/self.zoom_show)
        if self.is_perspect and QMouseEvent.button() == Qt.LeftButton:
            if len(self.points)<4:
                self.points.append((x,y))
                self.points.sort(key=lambda x:x[1])
                print(self.points)
                self.points[:2]=sorted(self.points[:2],key=lambda x:x[0]) 
                self.points[2:]=sorted(self.points[2:],key=lambda x:x[0],reverse=True) 
                print(self.points,'\n')
                if len(self.points)==4:
                    self.confirm_bt.show()
    
    def keyPressEvent(self, QKeyEvent):
        print('key press: ',QKeyEvent.key())
        if QKeyEvent.key() == Qt.Key_Escape:
            if self.is_perspect:
                self.is_perspect=False
                self.points.clear()
                self.status_bar.showMessage('退出透视变换')
                self.perspect_bt.setChecked(False)
            if not self.confirm_bt.isHidden():
                self.confirm_bt.hide()
            self.points.clear()
        elif QKeyEvent.key()==16777220:#回车
            if not self.confirm_bt.isHidden():
                self.confirm_func()
        elif QKeyEvent.key() == Qt.Key_Control:
            self.control_key=True
            self.scroll_area.verticalScrollBar().setEnabled(False)
            self.scroll_area2.verticalScrollBar().setEnabled(False)
            self.img_all_scrollArea.verticalScrollBar().setEnabled(False)

    def keyReleaseEvent(self, QKeyEvent):
        print('key release: ',QKeyEvent.key())
        if QKeyEvent.key() == Qt.Key_Control:
            self.control_key=False
            self.scroll_area.verticalScrollBar().setEnabled(True)
            self.scroll_area2.verticalScrollBar().setEnabled(True)
            self.img_all_scrollArea.verticalScrollBar().setEnabled(True)

class SettingWindow(QWidget):
    def __init__(self):
        super(SettingWindow,self).__init__()
        self.setWindowIcon(QIcon(path+'images/setting.png'))
        self.setWindowTitle('首选项')
        self.move(1400,200)
        
        self.all_v_layout=QVBoxLayout()
        self.imageTab=QWidget(self)
        self.imageTab_h_layout=QHBoxLayout()
        self.image_label = QLabel(self)
        convertToQtFormat = QImage(path+'images/loading.png')
        qtimg = convertToQtFormat.scaled(400,400, Qt.KeepAspectRatio)
        self.image_label.setPixmap(QPixmap.fromImage(qtimg))
        self.image_label.setScaledContents(True)
        self.image_label.setMaximumWidth(800)
        self.image_label.setMaximumHeight(900)
        self.imageTab_h_layout.addWidget(self.image_label)
        self.imageTab.setLayout(self.imageTab_h_layout)
        
        self.motorTab=QWidget(self)
        self.fullTab=QWidget(self)
        self.fullTab_h_layout=QHBoxLayout()
        self.full_label = QLabel(self)
        convertToQtFormat = QImage(path+'images/full.jpeg')
        qtimg = convertToQtFormat.scaled(1400,1400, Qt.KeepAspectRatio)
        self.full_label.setPixmap(QPixmap.fromImage(qtimg))
        self.full_label.setScaledContents(True)
        self.full_label.setMaximumWidth(1400)
        self.full_label.setMaximumHeight(1400)
        self.fullTab_h_layout.addWidget(self.full_label)
        self.fullTab.setLayout(self.fullTab_h_layout)
        self.tab=QTabWidget(self)
        self.tab.addTab(self.imageTab,'图像设置')
        self.tab.addTab(self.motorTab,'电机设置')
        self.tab.addTab(self.fullTab,'满载警报')
        
        self.all_v_layout.addWidget(self.tab)
        self.bt_h_layout=QHBoxLayout()
        self.confirmbt=QPushButton('确定',self)
        self.confirmbt.clicked.connect(self.confirm)
        self.cancelbt=QPushButton('取消',self)
        self.cancelbt.clicked.connect(self.hide)
        self.bt_h_layout.addWidget(self.confirmbt)
        self.bt_h_layout.addWidget(self.cancelbt)
        self.all_v_layout.addLayout(self.bt_h_layout)
        self.setLayout(self.all_v_layout)
        
    def confirm(self):
        self.hide()
    def resetImg(self,image):
        self.image_label.setPixmap(QPixmap.fromImage(image))
        self.image_label.setFixedHeight(self.image_label.width()*image.height()//image.width())

class RecognitionThread(QThread):
    my_signal = pyqtSignal(QImage,bool,float)
    newItem_signal = pyqtSignal(str,int,int,int,float,int,int,int,bool,float)
    playVideoToMainWindow_singal=pyqtSignal()
    mainWindowToPlayVideo_singal=pyqtSignal()
    updataVelocityToTab2_singal=pyqtSignal(float)
    pause_play_singal=pyqtSignal()
    def __init__(self):
        super(RecognitionThread, self).__init__()
        self.loop=True
        self.rubbish=[]
        self.ishandled=1
        self.isupdate=1
        self.anyNew=False
        self.lastOut=False          #上次是否是在边上
        self.processcnt=0
        self.catchorder=['papercup','can','mu','bottle','bottle2','brick','stone','battery','battery1','battery5','drug','drugbag','drugbox','potato','cap','glove','potatocut','rabbitcut','mooli','paper','capsule','china']
        self.weight={}              #用于防止一直抓一个的权重
        self.move_history=queue.Queue()
        #self.startLight=Light()
        self.readCamera=ReadCamera()
        self.moveThread = Move()
        #self.moveThread.my_signal.connect(self.resetImg)
        self.moveThread.start()
        self.mutex=QMutex()
        
    def run(self):
        print('loading yolo')
        if JETSON:
            self.model=YOLO("/home/jetson/Desktop/ultralytics-main/runs/segment/bestg.pt")
        else:
            self.model=YOLO("F:/ultralytics-main/runs/segment/train3.29/weights/best3.29.pt")
        print('reading camera')
        self.msleep(100)
        self.readCamera.start()
        while self.readCamera.upnew==False and self.loop: 
            self.msleep(1)
        frame=self.readCamera.frame
        if USEDEEPCAMERA:depthimg=self.readCamera.depthimg
        demo.hide()
        main_window.hide()
        print('predicting')
        gc.collect()
        torch.cuda.empty_cache()
        try:
            results=self.model.predict(frame,agnostic_nms=True,conf=0.5,verbose=False)
        except Exception as e:
            print('first try err',e)
            self.readCamera.loop=False
            self.readCamera.quit()
            self.readCamera.wait()
            self.readCamera.start()
            self.msleep(10)
            self.readCamera.upnew=False
            self.readCamera.getnew=True
            while self.readCamera.upnew==False and self.loop: 
                self.msleep(1)
            frame=self.readCamera.frame
            if USEDEEPCAMERA:depthimg=self.readCamera.depthimg
            print('predicting2')
            results=self.model.predict(frame,agnostic_nms=True,conf=0.5,verbose=False)
        self.readCamera.upnew=False
        self.readCamera.getnew=True
        if not COMPUTERTEST:
            #等待按下启动
            try:
                GPIO.setmode(GPIO.BOARD)
                GPIO.setup(KEY,GPIO.IN)
                GPIO.setup(LED,GPIO.OUT,initial=0)
                print("wait for press")
                self.msleep(10)
                while (not GPIO.input(KEY)) and USEKEY:
                    self.msleep(100)
                print("key down")
                keycnt=0
                while GPIO.input(KEY) and USEKEY:
                    GPIO.output(LED,1)
                    self.msleep(100)
                    keycnt+=1
                    if keycnt>20: break
                print("key up")
            except:
                print("KEY error,start")
            self.msleep(100)
            motors.servo5_change(0)
            motors.servo_sport(5,3000,90,250)
            motors.servo_sport(6,2048,30,50)
            motors.servo_sport(7,2048,30,50)
            motors.servo_2_rotate(-10)
            self.msleep(100)
            motors.close_around()
            #motors.motor_sport(1,2004,1500)
        demo.showFullScreen()
        main_window.status.setText('正在识别')
        
        print('start recognise')
        le=0
        lastnum,llastnum=0,0#上两次识别到的数量
        lastnotcatchnum=0#不抓时候的数量
        lastPanprocess=False
        lastmove,llastmove=0,0#上两次运动速率
        lastitem,lastBitem=None,None#上一次识别到的物品
        catchnum,catchBnum,whole_catchnum=0,0,0  #上一次抓取时里面的垃圾数
        lastitems=[]
        mean_depth=0.0
        self.time=QDateTime.currentDateTime().toMSecsSinceEpoch()#更新时间
        self.sharptime=QDateTime.currentDateTime().toMSecsSinceEpoch()#更新时间
        self.sharpcnt=0
        while self.loop:
            if (not COMPUTERTEST) and GPIO.input(KEY):
                while GPIO.input(KEY):
                    self.msleep(100)
                self.pause_play_singal.emit()
            self.mutex.lock()
            while self.readCamera.upnew==False and self.loop: 
                self.usleep(100)
            frame=self.readCamera.frame
            if USEDEEPCAMERA:depthimg=self.readCamera.depthimg
            self.readCamera.upnew=False
            self.readCamera.getnew=True

            if self.anyNew==False:
                results=self.model.predict(frame,agnostic_nms=True,conf=0.5,verbose=False) #推导
                frame=results[0].plot()                             #获取画框后的图片
                thistime=QDateTime.currentDateTime().toMSecsSinceEpoch()#更新时间
                if results[0].masks:                                #如果检测到垃圾
                    if main_window.isHidden():                      #且在宣传片页面，跳转到主窗口
                        self.playVideoToMainWindow_singal.emit()
                    self.time=thistime#更新时间
                    self.sharptime=thistime
                elif (not main_window.isHidden()) and thistime-self.time>180000:
                    self.mainWindowToPlayVideo_singal.emit()        #长时间未识别到垃圾，返回播放宣传片
                self.sharpcnt+=1
                if (not main_window.isHidden()) and thistime-self.sharptime>=7000:
                    main_window.bilateralFilter_bt.setChecked(True)
                    main_window.sharpening_bt.setChecked(True)
                    self.sharptime=thistime
                    self.sharpcnt=0
                if self.sharpcnt>=5 and main_window.sharpening_bt.isChecked():
                    main_window.bilateralFilter_bt.setChecked(False)
                    main_window.sharpening_bt.setChecked(False)
                    self.sharptime=thistime
                try:
                    xylist=results[0].boxes.xyxy
                    masks=results[0].masks.xy
                    probs=results[0].boxes.conf
                    information=[]                                  #存储每一个物体
                    for i in range(len(masks)):
                        mask=np.array(masks[i],int)
                        M = cv2.moments(mask)                       #通过矩计算重心
                        x = int(M['m10']/M['m00'])
                        y = int(M['m01']/M['m00'])
                        xyxy=xylist[i]
                        itemName=results[0].names[int(results[0].boxes.cls[i])]
                        # for xy in mask:
                        #     cv2.circle(frame,(int(xy[0]),int(xy[1])),1,(0,255,0))
                        try:
                            rows,cols = frame.shape[:2]#直线拟合
                            dx,dy,__x_,__y_ = cv2.fitLine(mask, cv2.DIST_L2,0,0.01,0.01) #返回直线的点斜
                            dx,dy=*dx,*dy
                            cv2.circle(frame,(x,y),5,(0,0,255),-1)
                            lefty = int((-x*dy/dx) + y)
                            righty = int(((cols-x)*dy/dx)+y)
                            cv2.line(frame,(cols-1,righty),(0,lefty),(255,255,0))
                            angle=-math.atan2(dy,dx)*180/math.pi
                            print(f'{itemName}: ({x},{y}) {angle:.1f}° {probs[i]*100:.1f}%',end=' ')
                            cv2.putText(frame,f'{angle:.1f}',(x+20,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,0))
                        except Exception as e:
                            print(f'fit line error of {itemName}: ',e)
                            angle=90.0
                        isout=0 if (x<80 or x>400 or y<80 or y>400) else 1
                        if USEDEEPCAMERA:
                            bmask=np.zeros_like(depthimg,dtype=np.uint8)
                            cv2.drawContours(bmask,[mask],0,255,-1)
                            region_depths=depthimg[bmask==255]
                            valid_region_depths=region_depths[(region_depths>0)&(region_depths<360)]
                            if len(valid_region_depths)>0:
                                top_depth=np.percentile(valid_region_depths,15)
                                #mean_depth=np.mean(valid_region_depths)
                                #mean_depth=np.percentile(valid_region_depths,85)+(np.percentile(valid_region_depths,85)-np.percentile(valid_region_depths,15))*min(x*1.3,480)/480
                                mean_depth=(top_depth+675)/3.0
                                mean_depth+=4
                                if mean_depth>340:mean_depth=340
                            else:
                                top_depth=332
                                mean_depth=340
                            print(f'depth: {mean_depth:.1f}mm')
                        else: print('')
                        weight=self.weight.get(itemName,1)
                        information.append((x,y,angle,itemName,i,isout,probs[i],weight,mean_depth,top_depth))             #存储起来以便排序
                       
                    rcdict={'可回收垃圾':1,'有害垃圾':2,'厨余垃圾':3,'其他垃圾':4}
                    #information.sort(key=lambda x:((-1 if rcdict[main_window.rubbishDict.get(x[3],'其他垃圾')] in [2,3] else 0) if (1 if x[7]>=3 else x[5])==0 else 1,(self.catchorder.index(x[3])-10*x[6]+15)*x[7]))#按照先边缘，再大小和准确度的顺序,并乘以权重
                    information.sort(key=lambda x:((-1 if rcdict[main_window.rubbishDict.get(x[3],'其他垃圾')] in [2,3] else 0) if (1 if x[7]>=3 else x[5])==0 else 1,(x[9]/6-10*x[6]+15)*x[7]))#按照先抓的次数上限，然后边缘；再深度和准确度,并乘以权重的顺序

                    (x,y,angle,itemName,i,isout,prob,weight,mean_depth,top_depth)=information[0]
                    
                    if self.moveThread.move==False:
                        target_equal=[results[0].names[int(x)] for x in results[0].boxes.cls.tolist()].count(itemName)==lastitems.count(itemName)  #目标类垃圾数相同
                        #判断抓取： 0是判断运动速率下降，1、2是上两次数量一致，3是5次识别中运动的
                        if llastmove+30>=lastmove+10>=self.readCamera.move and len(masks)==lastnum==llastnum and target_equal and (sum(self.move_history.queue)==0):                               #上传识别到的垃圾
                            #判断是否直接倾倒
                            # panprocess=True                            
                            # onename=main_window.rubbishDict.get(information[0][3],'其他垃圾')
                            # for i in range(len(information)):
                            #     if onename!=main_window.rubbishDict.get(information[i][3],'其他垃圾'):panprocess=False
                            #     if information[i][6]<0.6 and len(masks)>1:panprocess=False
                            
                            #显示和执行
                            if ((not SINGLEPROCESS) or itemName!='capsule' or 13>max(self.readCamera.move,lastmove,llastmove)):#如果是单垃圾模式但是抓，运动判断更严格, and capsule shoule move slower
                                self.processcnt+=1
                                if lastnotcatchnum==0:      #判断上次处理是否成功
                                    caughtsuccedd=True
                                else:
                                    if len(masks)<whole_catchnum:                   #总数减少
                                        vote=1
                                    else:
                                        vote=0
                                    if [results[0].names[int(x)] for x in results[0].boxes.cls.tolist()].count(lastitem)<catchnum:      #小类中减少
                                        vote+=1
                                    if [main_window.rubbishDict.get(results[0].names[int(x)],'其他垃圾') for x in results[0].boxes.cls.tolist()].count(lastBitem)<catchBnum:  #大类中减少
                                        vote+=1
                                    if vote>=2: caughtsuccedd=True
                                    else:       caughtsuccedd=False
                                #caughtsuccedd=not (len(masks)==catchnum and lastnotcatchnum>0 and sorted(lastname)==sorted(results[0].boxes.cls))      #所有垃圾一致视为失败
                                #caughtsuccedd=lastnotcatchnum==0 or (not [results[0].names[int(x)] for x in results[0].boxes.cls.tolist()].count(lastitem)>=catchnum)  #同种垃圾数量一致视为失败
                                #caughtsuccedd=lastnotcatchnum==0 or (not [main_window.rubbishDict.get(results[0].names[int(x)],'其他垃圾') for x in results[0].boxes.cls.tolist()].count(lastitem)>=catchnum)#同类垃圾数量一致视为失败
                                
                                panprocess=True if (len(masks)==1 and (SINGLEPROCESS or ((lastnotcatchnum==1 or lastnotcatchnum==2 and caughtsuccedd) and (itemName in ['capsule','china'] or caughtsuccedd==False and self.weight.get(itemName,1)>2))) and (self.moveThread.notTakein or self.weight.get(itemName,1)==1)) else False
                                if isout==0:     #一定范围外抓，0是外
                                    panprocess=False
                                if (not caughtsuccedd and self.processcnt%2==0) and not self.lastOut and lastPanprocess:
                                    panprocess=False          #上次没倒下去就抓
                                self.lastOut=(isout==0)
                                lastPanprocess=panprocess
                                print(f'{itemName} last catch succeedd:{caughtsuccedd} {lastnotcatchnum}')
                                if caughtsuccedd==False and self.moveThread.notTakein:          #抓且没抓起来->增加权重排序靠后
                                    self.weight[itemName]=self.weight.get(itemName,1)*1.5
                                
                                x_max=main_window.axiswidth_spinbox.value()
                                y_max=main_window.axisheight_spinbox.value()
                                axis_x=(x-main_window.point1_x_spinbox.value())*x_max//(main_window.point2_x_spinbox.value()-main_window.point1_x_spinbox.value())
                                axis_y=(y-main_window.point1_y_spinbox.value())*y_max//(main_window.point2_y_spinbox.value()-main_window.point1_y_spinbox.value())
                            
                                #获取物体最小宽度
                                mask=np.array(masks[i],int)
                                min_width=480
                                m_angle=angle
                                for anglei in range(max(-90,int(angle)-45),min(90,int(angle)+45),5):    #步长5度范围90度
                                    M=cv2.getRotationMatrix2D((0,0),-anglei, 1)#旋转变换矩阵
                                    rotated=np.dot(mask,M.T[:2,:2])
                                    topmost=rotated[rotated[:,1].argmin()][1]
                                    bottommost=rotated[rotated[:,1].argmax()][1]
                                    width=(bottommost-topmost)
                                    if width<min_width:
                                        min_width=width
                                        m_angle=anglei
                                angle=m_angle
                                for anglei in range(max(-90,int(angle)-3),min(90,int(angle)+2)):        #步长1度范围5度
                                    M=cv2.getRotationMatrix2D((0,0),-anglei, 1)
                                    rotated=np.dot(mask,M.T[:2,:2])
                                    topmost=rotated[rotated[:,1].argmin()][1]
                                    bottommost=rotated[rotated[:,1].argmax()][1]
                                    width=(bottommost-topmost)
                                    if width<min_width:
                                        min_width=width
                                        m_angle=anglei
                                width=min_width
                                angle=m_angle
                                # if itemName=='papercup':
                                    # angle=angle+(90 if angle<0 else -90)

                                #计算判断是否需要先将垃圾分开所需的距离
                                minl1,minl2=480,480
                                angle2=angle+(90 if angle<0 else -90)
                                xc1=x+width*math.cos(angle2*math.pi/180)/2
                                yc1=y-width*math.sin(angle2*math.pi/180)/2
                                xc2=x-width*math.cos(angle2*math.pi/180)/2
                                yc2=y+width*math.sin(angle2*math.pi/180)/2
                                cv2.circle(frame,(int(xc1),int(yc1)),3,(0,255,0),-1)
                                cv2.circle(frame,(int(xc2),int(yc2)),3,(0,255,0),-1)
                                for j in range(len(masks)):
                                    if j==i or results[0].names[int(results[0].boxes.cls[j])] not in ['paper','battery']:
                                        continue
                                    maskj=masks[j]
                                    maskj=masks[j]
                                    l12=(xc1-maskj[:,0])**2+(yc1-maskj[:,1])**2
                                    l22=(xc2-maskj[:,0])**2+(yc2-maskj[:,1])**2
                                    minl1=np.sqrt(l12.min())
                                    minl2=np.sqrt(l22.min())
                                print('catch',itemName,'width:',(bottommost-topmost)*103/227,'mm',f'nearest:{min(minl1,minl2)*103/277:.1f}mm',f'weight:{weight}' if weight>1 else '')

                                self.newItem_signal.emit(itemName,len(masks) if panprocess else 1,int(x),int(y),angle,int(axis_x),int(axis_y),int((bottommost-topmost)*103//227),caughtsuccedd and self.moveThread.notTakein,mean_depth)
                                rcname=main_window.rubbishDict.get(itemName,'其他垃圾')
                                self.moveThread.x,self.moveThread.y,self.moveThread.angle,self.moveThread.k,self.moveThread.pan=axis_y,axis_x,angle,rcdict[rcname],panprocess
                                self.moveThread.itemname=itemName
                                self.moveThread.width=width*103/227+(8 if itemName=='china' else 0)  #85/277精确值；这里让他张大一点
                                if itemName in ['paper','bottle','bottle2','papercup','mu']:
                                    self.moveThread.height=mean_depth+4
                                else:
                                    self.moveThread.height=mean_depth
                                if minl1<minl2:
                                    self.moveThread.cy=(xc1-main_window.point1_x_spinbox.value())*x_max//(main_window.point2_x_spinbox.value()-main_window.point1_x_spinbox.value())
                                    self.moveThread.cx=(yc1-main_window.point1_y_spinbox.value())*y_max//(main_window.point2_y_spinbox.value()-main_window.point1_y_spinbox.value())
                                    self.moveThread.cl=minl1*103/277
                                else:
                                    self.moveThread.cy=(xc2-main_window.point1_x_spinbox.value())*x_max//(main_window.point2_x_spinbox.value()-main_window.point1_x_spinbox.value())
                                    self.moveThread.cx=(yc2-main_window.point1_y_spinbox.value())*y_max//(main_window.point2_y_spinbox.value()-main_window.point1_y_spinbox.value())
                                    self.moveThread.cl=minl2*103/277
                                self.moveThread.move=True
                                print('set the move true')

                                lastitem =itemName
                                lastBitem=main_window.rubbishDict.get(itemName,'其他垃圾')
                                catchnum =[results[0].names[int(x)] for x in results[0].boxes.cls.tolist()].count(lastitem)
                                catchBnum=[main_window.rubbishDict.get(results[0].names[int(x)],'其他垃圾') for x in results[0].boxes.cls.tolist()].count(lastBitem)
                                whole_catchnum=len(masks)
                                if not main_window.manual_action.isChecked():            #且没有打开仅识别，不再继续检测
                                    self.anyNew=True
                                lastnotcatchnum=len(masks)
                        if lastnotcatchnum>0:lastnotcatchnum=len(masks)
                    llastnum=lastnum
                    lastnum=len(masks)
                    lastitems=[results[0].names[int(x)] for x in results[0].boxes.cls.tolist()]
                except AttributeError as e:
                    if self.moveThread.move==False:
                        lastnotcatchnum=0
                        self.weight.clear()
                except Exception as e:
                    print('Exception1 occurred: ',e)
                recognited=frame.copy()
            else:
                self.msleep(33)
            llastmove=lastmove
            lastmove=self.readCamera.move
            if ((not SINGLEPROCESS) and main_window.detect_threshold_spinbox.value()<=self.readCamera.move) or (SINGLEPROCESS and main_window.catch_threshold_spinbox.value()<=self.readCamera.move):  #判断运动
                if self.move_history.qsize()>=5:
                    self.move_history.get()
                self.move_history.put(1)
            else:
                if self.move_history.qsize()>=5:
                    self.move_history.get()
                self.move_history.put(0)

            try:
                if not main_window.isHidden():
                    nowtabindex=main_window.tab.currentIndex()
                    if nowtabindex==0:
                        if not main_window.isplay:
                            pause_img=cv2.imread(path+'images/video-pause.png')
                            pause_img=cv2.resize(pause_img,(frame.shape[1],frame.shape[0]))
                            frame=cv2.addWeighted(frame,0.4,pause_img, 0.6,-153)
                        elif sum(self.move_history.queue)>=3:
                            move_img=cv2.imread(path+'images/move-arrow.png')
                            move_img=cv2.resize(move_img,(frame.shape[1],frame.shape[0]))
                            frame=cv2.addWeighted(frame,0.4,move_img, 0.6,-153)
                        convertToQtFormat = QImage(cv2.addWeighted(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB), 0.8, recognited, 0.2, 0),frame.shape[1],frame.shape[0],3*frame.shape[1], QImage.Format_RGB888)
                        self.my_signal.emit(convertToQtFormat,self.anyNew,self.readCamera.move)
                    elif nowtabindex==1:
                        self.updataVelocityToTab2_singal.emit(self.readCamera.move)
            except Exception as e:
                print('Exception2 occurred: ',e)
            while self.isupdate==False and self.loop:
                self.readCamera.video.read()
                self.msleep(50)
                if (not COMPUTERTEST) and GPIO.input(KEY):
                    while GPIO.input(KEY):
                        self.msleep(100)
                    self.pause_play_singal.emit()
                    # if main_window.isHidden():                      #且在宣传片页面，跳转到主窗口
                    #     self.playVideoToMainWindow_singal.emit()
                    self.sharptime=QDateTime.currentDateTime().toMSecsSinceEpoch()
                    self.time=self.sharptime
            self.mutex.unlock()

        # self.startLight.pwm32.stop()
        # self.startLight.pwm33.stop()
        # self.startLight.quit()
        self.readCamera.loop=False
        self.moveThread.loop=False
        self.readCamera.quit()
        self.moveThread.quit()
        self.readCamera.wait()
        self.moveThread.wait()
        #self.startLight.wait()

class ReadCamera(QThread):
    read_singal=pyqtSignal() 
    img_signal2 = pyqtSignal(QImage)
    img_tab2_signal = pyqtSignal(np.ndarray)
    img_tab2_signal2 = pyqtSignal(np.ndarray)
    img_tab2_signal3 = pyqtSignal(np.ndarray,float)
    def __init__(self):
        super(ReadCamera, self).__init__()
        self.getnew=True
        self.loop=True
        self.upnew=False
        self.move=0
        # self.full=False
        # self.fullHistory=queue.Queue()
        # self.fullHistory.put(0)
        # self.fullHistory.put(0)
        # self.fullHistory.put(0)
        # self.fullHistory.put(0)
        # self.fullHistory.put(0)
        # self.start_full_time=0
        # self.average_full_distance=12
        self.mutex = QMutex()

    def getProcessedFrame(self):
        if USEDEEPCAMERA:
            frames=self.pipline.wait_for_frames()
            frames=self.align.process(frames)
            frame=frames.get_color_frame()
            depth_frame=frames.get_depth_frame()
            if not frame or not depth_frame:
                return False,None
            depth_frame=self.temporal.process(depth_frame)
            depth_image = np.asanyarray(depth_frame.get_data())
            frame = np.asanyarray(frame.get_data())
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
        else:
            flag,frame=self.video.read()
            if not flag and frame is None:
                return False,None
        
        if not main_window.settingPage.isHidden():              #如果在设置页面，把摄像头的初始图像传过去
            convertToQtFormat = QImage(np.ndarray.copy(frame),frame.shape[1],frame.shape[0],3*frame.shape[1], QImage.Format_RGB888)
            self.img_signal2.emit(convertToQtFormat)
        #print(frame.shape)
        if main_window.tab.currentIndex()==1:                   #如果是Tab2，把摄像头的初始图像传过去
            self.img_tab2_signal.emit(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            
        if main_window.hflip_bt.isChecked():                    #图像预处理
            cv2.flip(frame,1,frame)
            if USEDEEPCAMERA:cv2.flip(depth_image,1,depth_image)
        if main_window.vflip_bt.isChecked():
            cv2.flip(frame,0,frame)
            if USEDEEPCAMERA:cv2.flip(depth_image,0,depth_image)
        if main_window.rotate_bt.isChecked():
            frame=cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
            if USEDEEPCAMERA:depth_image=cv2.rotate(depth_image,cv2.ROTATE_90_COUNTERCLOCKWISE)
        if len(main_window.jsonData['points'])==4:
            pts2=np.float32([(0,0),(main_window.imgwidth_spinbox.value()-1,0),(main_window.imgwidth_spinbox.value()-1,main_window.imgheight_spinbox.value()-1),(0,main_window.imgheight_spinbox.value()-1)])
            M=cv2.getPerspectiveTransform(np.float32(main_window.jsonData['points']),pts2)
            frame=cv2.warpPerspective(frame,M,(main_window.imgwidth_spinbox.value(),main_window.imgheight_spinbox.value()))
            if USEDEEPCAMERA:depth_image=cv2.warpPerspective(depth_image,M,(main_window.imgwidth_spinbox.value(),main_window.imgheight_spinbox.value()))
        else:
            frame=np.ndarray.copy(frame[:,(frame.shape[1]-frame.shape[0])//2:(frame.shape[1]+frame.shape[0])//2])
            if USEDEEPCAMERA:depth_image=np.ndarray.copy(depth_image[:,(depth_image.shape[1]-depth_image.shape[0])//2:(depth_image.shape[1]+depth_image.shape[0])//2])
        if main_window.bilateralFilter_bt.isChecked():
            frame=cv2.bilateralFilter(frame,11,30,15)

        try:                                                    #计算运动速率
            qianjing=self.bjfg.apply(frame)
            qianjing=cv2.morphologyEx(qianjing,cv2.MORPH_OPEN,np.ones((3,3)))
            contours,hi=cv2.findContours(qianjing,3,2)
            self.move=0
            for c in contours:
                self.move+=cv2.contourArea(c)
        except:
            print('error with doing MOG2')
            
        if main_window.sharpening_bt.isChecked():
            frame=cv2.filter2D(frame,8,np.array([[-0.1,-0.5,-0.1],[-0.5,3.4,-0.5],[-0.1,-0.5,-0.1]]))
        if main_window.equalize_bt.isChecked():
            yuv=cv2.cvtColor(frame,cv2.COLOR_RGB2YUV)
            yuv[:,:,0]=cv2.createCLAHE(2,(8,8)).apply(yuv[:,:,0])
            cv2.cvtColor(yuv,cv2.COLOR_YUV2RGB,frame)
        if USEDEEPCAMERA:
            return True,frame,depth_image
        else:
            return True,frame

    def run(self):
        if USEDEEPCAMERA:
            self.pipline=rs.pipeline()
            config=rs.config()
            config.enable_stream(rs.stream.color,1920,1080,rs.format.bgr8,30)
            config.enable_stream(rs.stream.depth,1280,720,rs.format.z16,30)
            self.pipline.start(config)
            self.align=rs.align(rs.stream.color)
            self.temporal = rs.temporal_filter()
            #self.msleep(100)
            cframes=[]
            for i in range(3):
                frames=self.pipline.wait_for_frames()
                frames=self.align.process(frames)
                cframes.append(frames.get_color_frame())
            if cframes[0]==cframes[1]==cframes[2]:
                self.pipline.stop()
                self.sleep(0.5)
                self.pipline=rs.pipeline()
                config=rs.config()
                config.enable_stream(rs.stream.color,1920,1080,rs.format.bgr8,30)
                config.enable_stream(rs.stream.depth,1280,720,rs.format.z16,30)
                self.pipline.start(config)
                self.align=rs.align(rs.stream.color)
                self.temporal = rs.temporal_filter()
                cframes=[]
                for i in range(3):
                    frames=self.pipline.wait_for_frames()
                    frames=self.align.process(frames)
                    cframes.append(frames.get_color_frame())
                if cframes[0]==cframes[1]==cframes[2]:
                    print('camera init error')
                    MainWindow.exit()
            def isOpened():
                return True
            self.video=type('video',(object,),{'isOpened':isOpened,'read':lambda: (True,cframes[0].get_data())})
        else:
            self.video=cv2.VideoCapture(0,cv2.CAP_V4L2)
            if not self.video.isOpened():
                self.video=cv2.VideoCapture(1)
            if not self.video.isOpened():
                print('camera init error')
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.video.set(cv2.CAP_PROP_FPS, 30)
        
        self.font=cv2.FONT_HERSHEY_SIMPLEX#create~~MOG2是背景分割器
        self.bjfg=cv2.createBackgroundSubtractorKNN(history=10,detectShadows=False)
        e2=0
        main_window.pause_action.setEnabled(True)
        while self.loop:
            #while self.getnew==False and self.loop:self.usleep(100)
            if USEDEEPCAMERA:
                flag,frame,depth_image=self.getProcessedFrame()
                if main_window.tab.currentIndex()==1:#如果是Tab2，显示预处理图片
                    self.img_tab2_signal2.emit(frame.copy())
                    self.img_tab2_signal3.emit(cv2.convertScaleAbs(cv2.add(depth_image,-250),alpha=2.5),depth_image[depth_image.shape[0]//2,depth_image.shape[1]//2])
            else:
                flag,frame=self.getProcessedFrame()
                if main_window.tab.currentIndex()==1:#如果是Tab2，显示预处理图片
                    self.img_tab2_signal2.emit(frame.copy())
            if not flag:
                continue
            
            self.mutex.lock()
            self.frame=frame
            if USEDEEPCAMERA:self.depthimg=depth_image

            self.upnew=True
            self.getnew=False
            self.mutex.unlock()
        if USEDEEPCAMERA:
            self.pipline.stop()
        else:
            self.video.release()
                

# class Light(QThread):               
#     def __init__(self):
#         super(Light, self).__init__()
#     def run(self):
#         GPIO.setmode(GPIO.BOARD)
#         GPIO.setup(15,GPIO.OUT,initial=GPIO.LOW)
#         GPIO.setup(33,GPIO.OUT,initial=GPIO.LOW)
#         self.pwm32=GPIO.PWM(15,1000)
#         self.pwm33=GPIO.PWM(33,1000)
#         self.pwm33.start(0)
#         self.pwm32.start(0)
#         for i in range(101):
#             self.pwm33.ChangeDutyCycle(i//2)
#             self.pwm32.ChangeDutyCycle(i)
#             time.sleep(0.01)

class Move(QThread):
    keepLoop_singal=pyqtSignal(bool)
    def __init__(self):
        super(Move,self).__init__()
        self.loop=True
        self.move=False
        self.x,self.y,self.angle,self.k,self.pan=1,1,0,1,False
        self.height=0
        self.width=55
        self.itemname=''
        self.cx,self.cy,self.cl=0,0,480
        self.mutex=QMutex()
        self.full=False
        self.fullHistory=queue.Queue()
        self.fullHistory.put(0)
        self.fullHistory.put(0)
        self.fullHistory.put(0)
        self.fullHistory.put(0)
        self.fullHistory.put(0)
        self.fullHistory.put(0)
        self.fullHistory.put(0)
        self.fullHistory.put(0)
        self.fullHistory.put(0)
        self.start_full_time=0
        self.average_full_distance=102
        self.lastMoveTime=time.time()+20
        self.notTakein=True
        self.yasuotimes=0
        
    def run(self):
        while self.loop:
            self.mutex.lock()
            if self.move:
                print('move running')
                if not COMPUTERTEST:
                    if self.pan:
                        if SINGLEPROCESS:self.msleep(350)
                        motors.whole_process_pan(self,self.k,self.x,self.y)
                        self.notTakein=True
                        self.msleep(500)
                    elif self.cl<7.5 and self.itemname=='paper' or self.cl<5 and self.itemname=='battery':
                        motors.separate_garbage(self.height,self.angle,self.cx,self.cy,self.width)
                        self.notTakein=False
                    else:
                        if self.width>81:self.width=81
                        elif self.width<10:self.width==10
                        self.notTakein=motors.all_grab(self,self.x,self.y,self.angle,self.k,self.width,self.height)
                    self.keepLoop_singal.emit(self.notTakein)
                    main_window.my_thread.sharptime=QDateTime.currentDateTime().toMSecsSinceEpoch()
                self.move=False
                self.lastMoveTime=time.time()
            self.mutex.unlock()
            
            if not COMPUTERTEST and ULTRASNOIC and SINGLEPROCESS and time.time()-self.lastMoveTime>20:
                try:
                    full_distance=motors.get_distances()
                    if main_window.tab.currentIndex()==1:
                        for i,num in enumerate(full_distance):
                            main_window.full_showdistance[i].setText(f'{num:.1f}')
                    if min(full_distance)<main_window.full_distance_spinbox.value() or max(full_distance)>main_window.full_distance_m_spinbox.value():
                        self.fullHistory.get()
                        self.fullHistory.put(1)
                        if time.time()-self.start_full_time>main_window.full_time_spinbox.value():
                            if not self.full:
                                QTimer.singleShot(0, self.show_setting_window)
                            self.full=True
                    else:
                        self.start_full_time=time.time()
                        self.fullHistory.get()
                        self.fullHistory.put(0)
                        #print(self.full,sum(self.fullHistory.queue),main_window.settingPage.tab.currentIndex(),main_window.settingPage.isHidden())
                        if self.full and sum(self.fullHistory.queue)<1 and main_window.settingPage.tab.currentIndex()==2:
                            if not main_window.settingPage.isHidden():
                                main_window.settingPage.tab.setCurrentIndex(0)
                                main_window.settingPage.hide()
                            self.full=False
                except Exception as e:
                    print('Detect full error: ',e)
            
    def show_setting_window(self):
        main_window.settingPage.tab.setCurrentIndex(2)
        main_window.settingPage.show()
            

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window=MainWindow()
    demo = videoPlayer()
    main_window.my_thread.playVideoToMainWindow_singal.connect(demo.toMainWindow)
    demo.showFullScreen()
    demo.hide()
    #demo.showMaximized()
    sys.exit(app.exec_())