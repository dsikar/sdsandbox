# https://docs.python.org/3/tutorial/classes.html
import cv2
import conf
import numpy as np

class RecordVideo():
    """
    Record video class, used to record videos from tcpflow log or still images
    """

    def __init__(self, model, videoname, img_cnt):
        """
        Record video while running predictions
        Inputs
            model: string, model name
            videoname: string, name to save video as
            img_cnt: number of images that will be added side by side
        """
        model = model.split('/')
        self.modelname = model[-1]
        videoname = videoname + '.avi'
        self.img_cnt = img_cnt
        self.VIDEO_WIDTH, self.VIDEO_HEIGHT = conf.VIDEO_WIDTH, conf.VIDEO_HEIGHT # 800, 600
        self.IMAGE_WIDTH, self.IMAGE_HEIGHT = conf.IMAGE_STILL_WIDTH, conf.IMAGE_STILL_HEIGHT
        self.VIDEO_WIDTH = self.IMAGE_WIDTH * img_cnt
        self.video = cv2.VideoWriter(videoname, 0, 11, (self.VIDEO_WIDTH, self.VIDEO_HEIGHT))  # assumed 11fps approximately
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        # video line spacing
        self.images = []

    def add_image(self, image, text):
        # self.img_arr_1 =
        #cv2.putText(image, model, (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # Predicted steering angle
        #pst = sa[len(sa) - 1][0]
        #pst *= conf.norm_const
        #simst = "Predicted steering angle: {:.2f}".format(pst)
        #cv2.putText(image, simst, (50, 115), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        self.images.append([image, text])

    def add_frame(self):
        y_offset = 50;
        x_offset = 50
        step = 65;
        img_cnt = len(self.images)
        # prepare images
        for i in range(0, img_cnt):
            # resize image
            image = self.images[i][0]
            image = cv2.resize(image, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT), cv2.INTER_AREA)
            # add text
            lines = self.images[i][1]
            for line in lines:
                cv2.putText(image, line, (x_offset, y_offset), self.font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                y_offset += step
                # print(line)
            # set start to top for printing lines in next frame
            y_offset = x_offset
            # store processed image
            self.images[i][0] = image
        # concatenate
        output_image = self.images[0][0]
        for i in range(1, img_cnt):
            output_image = np.concatenate((output_image, self.images[i][0]), axis=1)
        # append
        try:
            self.video.write(np.uint8(output_image)) # catch error Assertion failed) image.depth() == CV_8U
        except Exception as e:
            print("Exception raise: " + str(e))
        # blank images
        self.images = []

    def save_video(self):
        # save video as videoname
        cv2.destroyAllWindows()
        self.video.release()
