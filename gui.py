import sys
from PyQt5.QtWidgets import QProgressBar, QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QPushButton, QLabel,QVBoxLayout, QSizePolicy, QToolButton
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSlot,QTimer, QRect,pyqtSignal
import os
import cv2
import dlib
import numpy as np
import argparse
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import logging
import sys
import numpy as np
from keras.models import Model
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

sys.setrecursionlimit(2 ** 20)
np.random.seed(2 ** 10)

class HoverButton(QToolButton):

    def __init__(self, parent=None):
        super(HoverButton, self).__init__(parent)
        self.setMouseTracking(True)

    def enterEvent(self,event):
        #print("Enter")
        self.setStyleSheet("color: #333;border: 2px solid #555;border-radius: 11px;padding: 5px;background-color: QRadialGradient(cx: 0.3, cy: -0.4,fx: 0.3, fy: -0.4,radius: 1.35, stop: 0 #fff, stop: 1 #bbb);font-size: 15px;padding-left: 5px;padding-right: 5px;")

    def leaveEvent(self,event):
        self.setStyleSheet("color: #333;border: 2px solid #555;border-radius: 11px;padding: 5px;background-color: QRadialGradient(cx: 0.3, cy: -0.4,fx: 0.3, fy: -0.4,radius: 1.35, stop: 0 #fff, stop: 1 #888);font-size: 15px;padding-left: 5px;padding-right: 5px;")
        #self.setStyleSheet("background-color:yellow;")
        #print("Leave")
        
class QutieBar(QProgressBar):
    value = 0
    def reset(progressBar):
        progressBar.value=0
        progressBar.setValue(progressBar.value)


    @pyqtSlot()
    def increaseValue(progressBar):
        progressBar.setTextVisible(True)
        progressBar.setValue(progressBar.value)
        progressBar.value = progressBar.value+1

class WideResNet:
    def __init__(self, image_size, depth=16, k=8):
        self._depth = depth
        self._k = k
        self._dropout_probability = 0
        self._weight_decay = 0.0005
        self._use_bias = False
        self._weight_init = "he_normal"

        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)

    # Wide residual network http://arxiv.org/abs/1605.07146
    def _wide_basic(self, n_input_plane, n_output_plane, stride):
        def f(net):
            # format of conv_params:
            #               [ [kernel_size=("kernel width", "kernel height"),
            #               strides="(stride_vertical,stride_horizontal)",
            #               padding="same" or "valid"] ]
            # B(3,3): orignal <<basic>> block
            conv_params = [[3, 3, stride, "same"],
                           [3, 3, (1, 1), "same"]]

            n_bottleneck_plane = n_output_plane

            # Residual block
            for i, v in enumerate(conv_params):
                if i == 0:
                    if n_input_plane != n_output_plane:
                        net = BatchNormalization(axis=self._channel_axis)(net)
                        net = Activation("relu")(net)
                        convs = net
                    else:
                        convs = BatchNormalization(axis=self._channel_axis)(net)
                        convs = Activation("relu")(convs)

                    convs = Conv2D(n_bottleneck_plane, kernel_size=(v[0], v[1]),
                                          strides=v[2],
                                          padding=v[3],
                                          kernel_initializer=self._weight_init,
                                          kernel_regularizer=l2(self._weight_decay),
                                          use_bias=self._use_bias)(convs)
                else:
                    convs = BatchNormalization(axis=self._channel_axis)(convs)
                    convs = Activation("relu")(convs)
                    if self._dropout_probability > 0:
                        convs = Dropout(self._dropout_probability)(convs)
                    convs = Conv2D(n_bottleneck_plane, kernel_size=(v[0], v[1]),
                                          strides=v[2],
                                          padding=v[3],
                                          kernel_initializer=self._weight_init,
                                          kernel_regularizer=l2(self._weight_decay),
                                          use_bias=self._use_bias)(convs)

            # Shortcut Connection: identity function or 1x1 convolutional
            #  (depends on difference between input & output shape - this
            #   corresponds to whether we are using the first block in each
            #   group; see _layer() ).
            if n_input_plane != n_output_plane:
                shortcut = Conv2D(n_output_plane, kernel_size=(1, 1),
                                         strides=stride,
                                         padding="same",
                                         kernel_initializer=self._weight_init,
                                         kernel_regularizer=l2(self._weight_decay),
                                         use_bias=self._use_bias)(net)
            else:
                shortcut = net

            return add([convs, shortcut])

        return f


    # "Stacking Residual Units on the same stage"
    def _layer(self, block, n_input_plane, n_output_plane, count, stride):
        def f(net):
            net = block(n_input_plane, n_output_plane, stride)(net)
            for i in range(2, int(count + 1)):
                net = block(n_output_plane, n_output_plane, stride=(1, 1))(net)
            return net

        return f

#    def create_model(self):
    def __call__(self):
        logging.debug("Creating model...")

        assert ((self._depth - 4) % 6 == 0)
        n = (self._depth - 4) / 6

        inputs = Input(shape=self._input_shape)

        n_stages = [16, 16 * self._k, 32 * self._k, 64 * self._k]

        conv1 = Conv2D(filters=n_stages[0], kernel_size=(3, 3),
                              strides=(1, 1),
                              padding="same",
                              kernel_initializer=self._weight_init,
                              kernel_regularizer=l2(self._weight_decay),
                              use_bias=self._use_bias)(inputs)  # "One conv at the beginning (spatial size: 32x32)"

        # Add wide residual blocks
        block_fn = self._wide_basic
        conv2 = self._layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=(1, 1))(conv1)
        conv3 = self._layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=(2, 2))(conv2)
        conv4 = self._layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=(2, 2))(conv3)
        batch_norm = BatchNormalization(axis=self._channel_axis)(conv4)
        relu = Activation("relu")(batch_norm)

        # Classifier block
        pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding="same")(relu)
        flatten = Flatten()(pool)
        predictions_g = Dense(units=2, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax")(flatten)
        predictions_a = Dense(units=101, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax")(flatten)

        model = Model(inputs=inputs, outputs=[predictions_g, predictions_a])

        return model
 
class App(QWidget):
    fileName = ''
    count=0
    def __init__(self):
        super().__init__()
        self.title = 'Age Prediction from facial images'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()
        
    
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setStyleSheet("color:#000;border: none;background: QRadialGradient(cx: 0.3, cy: -0.4,fx: 0.3, fy: -0.4,radius: 1.35,stop: 0 #a6a6a6, stop: 1 #333333);")
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.button = HoverButton()
        self.button.setText('Input file:')
        self.button.move(100,70) 
        self.button.clicked.connect(self.on_click)
        self.button.setStyleSheet("color: #333;border: 2px solid #555;border-radius: 11px;padding: 5px;background-color: QRadialGradient(cx: 0.3, cy: -0.4,fx: 0.3, fy: -0.4,radius: 1.35, stop: 0 #fff, stop: 1 #888);font-size: 15px;padding-left: 5px;padding-right: 5px;")
        self.button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum);

        
        self.button2 = HoverButton()
        self.button2.setText('Predict')
        self.button2.move(100,70) 
        self.button2.clicked.connect(self.predict)
        self.button2.setStyleSheet("color: #333;border: 2px solid #555;border-radius: 11px;padding: 5px;background-color: QRadialGradient(cx: 0.3, cy: -0.4,fx: 0.3, fy: -0.4,radius: 1.35, stop: 0 #fff, stop: 1 #888);font-size: 15px;padding-left: 5px;padding-right: 5px;")
        self.button2.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum);

        # a figure instance to plot on
        self.figure = plt.figure(frameon=True)
        self.l1 = QLabel()
        self.l1.setPixmap(QPixmap("logo.png"))
        #self.l1.setGeometry(self.left,self.top,self.width,550)
        #self.l1.setFixedHeight(550)
        self.l1.setScaledContents(True)
        self.l1.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        self.button3 = HoverButton()
        self.button3.setText('Live Detection from Camera')
        self.button3.move(100,70) 
        self.button3.setStyleSheet("color: #333;border: 2px solid #555;border-radius: 11px;padding: 5px;background-color: QRadialGradient(cx: 0.3, cy: -0.4,fx: 0.3, fy: -0.4,radius: 1.35, stop: 0 #fff, stop: 1 #888);font-size: 15px;padding-left: 5px;padding-right: 5px;")
        self.button3.clicked.connect(self.predictVideo)
        self.button3.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Maximum);

        vbox = QVBoxLayout()
        vbox.addWidget(self.l1)
        vbox.addWidget(self.button)
        vbox.addWidget(self.button2)
        vbox.addWidget(self.button3)
        self.button2.hide()
        self.bar = QutieBar(self)
        self.bar.setTextVisible(True)
        self.bar.setAlignment(Qt.AlignCenter)
        vbox.addWidget(self.bar)
        self.setLayout(vbox)
        self.timer = QTimer()
        
        
        #self.bar.connect(self.timer, SIGNAL("timeout()"), self.bar, SLOT("increaseValue()")) 
        
        self.show()
    
    def plot(self,img):
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.l1.setPixmap(QPixmap(qImg))
        print('Image printed')
        self.button.show()
        self.button2.hide()
        #self.progressBar.setRange(0,1)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        global fileName
        fileName, _ = QFileDialog.getOpenFileName(self,"Select input image", "","Image files (*.jpg *.jpeg *.png *.tif *.cms);;JPEG (*.jpg *.jpeg);;TIFF (*.tif);;PNG (*.png);;CMS (*.cms)",options=options)
        if fileName:
            self.timer = QTimer()
            self.bar.reset()
            self.l1.setPixmap(QPixmap(fileName))
            self.button.hide()
            self.button2.show()
            #self.predict(fileName)

    @pyqtSlot()
    def on_click(self):
        self.openFileNameDialog()

    def draw_label(self, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,font_scale=2, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)
        return image


    def predict(self):
        self.timer.start(1)
        depth = 16
        k = 8
        weight_file = "weights.18-4.06.hdf5"

        detector = dlib.get_frontal_face_detector()

        img_size = 64
        model = WideResNet(img_size, depth=depth, k=k)()
        model.load_weights(weight_file)

        img = cv2.imread(fileName)

        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))
        #print(faces.shape)
        for i, d in enumerate(detected):
            self.timer.timeout.connect(self.bar.increaseValue)
            
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - 0.4 * w), 0)
            yw1 = max(int(y1 - 0.4 * h), 0)
            xw2 = min(int(x2 + 0.4 * w), img_w - 1)
            yw2 = min(int(y2 + 0.4 * h), img_h - 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            faces[i,:,:,:] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
            results = model.predict(np.expand_dims(faces[i],axis=0))
            #print((np.expand_dims(faces[i],axis=0).shape))
            #plt.imshow(faces)
            predicted_genders = results[0]
            #print(predicted_genders)
            ages=np.zeros(101)
            for j in range(101):
                ages[j]=j
            top3=(sorted(zip(results[1][0],ages), reverse=True)[:3])
            print (top3)
            answer_age=top3[0][1]
            label = "{}".format(int(answer_age))
            img = self.draw_label(img, (d.left(), d.top()), label)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.plot(img)
        
    def predictVideo(self):
        #args = get_args()
        depth = 16
        k = 8
        weight_file = "weights.18-4.06.hdf5"

        if not weight_file:
            weight_file = "weights.18-4.06.hdf5"

        # for face detection
        detector = dlib.get_frontal_face_detector()

        # load model and weights
        img_size = 64
        model = WideResNet(img_size, depth=depth, k=k)()
        model.load_weights(weight_file)

        # capture video
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                print("error: failed to capture image")
                return -1

            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = np.shape(input_img)

            # detect faces using dlib detector
            detected = detector(input_img, 1)
            faces = np.empty((len(detected), img_size, img_size, 3))
            #print(faces.shape)
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - 0.4 * w), 0)
                yw1 = max(int(y1 - 0.4 * h), 0)
                xw2 = min(int(x2 + 0.4 * w), img_w - 1)
                yw2 = min(int(y2 + 0.4 * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i,:,:,:] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
            #print(len(detected),faces.shape)
            if len(detected) > 0:
                # predict ages and genders of the detected faces
                results = model.predict(faces)
                #print(results[0])
                ages=np.zeros(101)
                for i in range(101):
                    ages[i]=i
                top3=(sorted(zip(results[1][0],ages), reverse=True)[:3])
                #print (top3)
                answer_age=top3[0][1]
                predicted_genders = results[0]
                
            for i, d in enumerate(detected):
                label = "{}".format(int(answer_age))
                self.draw_label(img, (d.left(), d.top()), label)


            cv2.imshow("result", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


     

 
   
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())