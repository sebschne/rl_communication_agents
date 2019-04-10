import numpy
import numpy as np
import scipy.ndimage
import matplotlib.pylab
import cv2
import imutils
import threading
import argparse
from naoqi import ALProxy
import rospy
from std_msgs.msg import String
import vision_definitions
import yarp


# Initialise YARP

roi = {"upper_x" : 420, "lower_x": 180, "upper_y":380, "lower_y" :120 }

class RobotVision():
    def __init__(self,robot, nao_ip = "127.0.0.1", nao_port = 9559, roi=roi ):
        self.list = []
        self.max_len=10
        self.pub = rospy.Publisher('chatter', String, queue_size=10)
        rospy.init_node('talker', anonymous=True)
        rate = rospy.Rate(10) # 10hz

        if robot == 'icub':
            yarp.Network.init()

            self.input_port = yarp.Port()
            self.input_port.open("/python-image-port")
            yarp.Network.connect("/icubSim/cam/left", "/python-image-port")
        elif robot == "nao":

            ip_addr = nao_ip
            port_num = nao_port

            # get NAOqi module proxy
            videoDevice = ALProxy('ALVideoDevice', ip_addr, port_num)

            # subscribe top camera
            AL_kTopCamera = 0
            AL_kQVGA = 1            # 320x240
            AL_kBGRColorSpace = 13
            captureDevice = videoDevice.subscribeCamera(
                "test", AL_kTopCamera, AL_kQVGA, AL_kBGRColorSpace, 10)

            # create image
            width = 320
            height = 240
            image = np.zeros((height, width, 3), np.uint8)
            while True:

                # get image
                result = videoDevice.getImageRemote(captureDevice);

                if result == None:
                    print 'cannot capture.'
                elif result[6] == None:
                    print 'no image data string.'
                else:

                    # translate value to mat
                    values = map(ord, list(result[6]))
                    i = 0
                    for y in range(0, height):
                        for x in range(0, width):
                            image.itemset((y, x, 0), values[i + 0])
                            image.itemset((y, x, 1), values[i + 1])
                            image.itemset((y, x, 2), values[i + 2])
                            i += 3

                    # show image
                    cv2.imshow("pepper-top-camera-320x240", self.detect_colour(image))
                green = self.list.count(1)
                not_green = self.list.count(0)
                green_perc = 0
                if green+not_green > 0:
                    green_perc = green/(green+not_green)

                if green_perc > 0.7:
                    self.pub.publish('green')
                else:
                    self.pub.publish('')

                # exit by [ESC]
                if cv2.waitKey(33) == 27:
                    break

        self.roi = roi
        # while(True):
        #
        #
        #
        #     self.yarp_to_python()
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

        while True:
            if robot == "icub":
                self.yarp_to_python()
            elif robot == "nao":
                self.naoqi_to_python()
            green = self.list.count(1)
            not_green = self.list.count(0)
            green_perc = 0
            if green+not_green > 0:
                green_perc = green/(green+not_green)

            if green_perc > 0.7:
                self.pub.publish('green')
            else:
                self.pub.publish('')



            # hello_str = "hello world %s" % rospy.get_time()

            rate.sleep()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



    def __del__(self):

        self.input_port.close()

    def naoqi_to_python(self):
        resolution = vision_definitions.kQQVGA
        self.camProxy.setResolution(self.nameId, resolution)

        print 'getting images in remote'
        result= self.camProxy.getImageRemote(self.nameId)
        print len(result)
        if result == None:
            print 'cannot capture.'
        elif result[6] == None:
            print 'no image data string.'
        else:
            # translate value to mat
            values = map(ord, list(result[6]))
            i = 0
            for y in range(0, self.height):
                for x in range(0, self.width):
                    self.image.itemset((y, x, 0), values[i + 0])
                    self.image.itemset((y, x, 1), values[i + 1])
                    self.image.itemset((y, x, 2), values[i + 2])
                    i += 3

                    # show image
                cv2.imshow("pepper-top-camera-320x240", self.image)

    def yarp_to_python(self):
        # Create a port and connect it to the iCub simulator virtual camera


        # Create numpy array to receive the image and the YARP image wrapped around it
        img_array = numpy.zeros((240, 320, 3), dtype=numpy.uint8)
        yarp_image = yarp.ImageRgb()
        yarp_image.resize(320, 240)
        yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])

        # Alternatively, if using Python 2.x, try:
        #yarp_image.setExternal(img_array.__array_interface__['data'][0], img_array.shape[1], img_array.shape[0])

        # Read the data from the port into the image
        self.input_port.read(yarp_image)

        # display the image that has been read
    #    matplotlib.pylab.imshow(img_array)
        cv2.imshow('frame', self.detect_colour(img_array))

      #      print "showing image", img_array
        # Cleanup

    def get_enclosingcircle(self,cnts_list,frame,colorid):
        center= None
        point=0
        c=0
        if colorid=="green":
            color=(0,255,0)
        if colorid=="red":
            color=(0,0,255)
        if colorid=="blue":
            color=(255,0,0)
        if colorid=="yellow":
            color=(0,255,255)

        if len(cnts_list)>=1:
            # self.list.append('green')


            for cnts in cnts_list:
                #cnts=cnts_list
                if c<cv2.contourArea(cnts):#max(cnts,key=cv2.contourArea)
                    ((x,y),radius)=cv2.minEnclosingCircle(cnts)
                    if (roi["lower_x"] <= x <= roi["upper_x"]) and (roi["lower_y"] <= y <= roi["upper_y"]):
                        if len(self.list)>self.max_len:
                            self.list.pop(0)
                            self.list.append(1)
                        else:
                            self.list.append(0)


                    c= cv2.contourArea(cnts)
                    contorn=cnts
            if c>0:


                point=point+1
                M= cv2.moments(contorn)
                center=(int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
                if radius > 0:
                    cv2.circle(frame,(int(x),int(y)),int(radius),color,2)
                    cv2.circle(frame,center,5,(0,0,255),-1)
                    text=str(point)
                    cv2.putText(frame,text,center,cv2.FONT_HERSHEY_PLAIN,1.0,(255,0,0))
        elif len(cnts_list)<1:
            if len(self.list)>self.max_len:
                self.list.pop(0)
                self.list.append(0)
            else:
                self.list.append(0)


        return frame

    def detect_colour(self,image):
        # old
            # greenLower=(35,86,6)
            # greenUpper=(64,255,255)
        # new
            greenLower = (33,80,6)
            greenUpper = (100,255,255)

            redLower=(1,86,6)
            redUpper=(18,255,255)
            blueLower=(110,86,6)
            blueUpper=(150,255,255)
            yellowLower=(24,86,6)
            yellowUpper=(28,255,255)

            frame= imutils.resize(image,width=600)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            #Green Mask
            mask1 = cv2.inRange(hsv, greenLower, greenUpper)
            mask1 = cv2.erode(mask1, None, iterations=2)
            mask1 = cv2.dilate(mask1, None, iterations=2)

            #Blue Mask
            mask2 = cv2.inRange(hsv, redLower, redUpper)
            mask2 = cv2.erode(mask2, None, iterations=2)
            mask2 = cv2.dilate(mask2, None, iterations=2)
            #Red Mask
            mask3 = cv2.inRange(hsv, blueLower, blueUpper)
            mask3 = cv2.erode(mask3, None, iterations=2)
            mask3 = cv2.dilate(mask3, None, iterations=2)
            #Yellow Mask
            mask4 = cv2.inRange(hsv, yellowLower, yellowUpper)
            mask4 = cv2.erode(mask4, None, iterations=2)
            mask4 = cv2.dilate(mask4, None, iterations=2)

            #Calculating Location
            l1=(150,171)
            l2=(5,215)
            l3=(553,297)
            l4=(549,366)

            cnts_list1=cv2.findContours(mask1.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

            #cnts_list2=cv2.findContours(mask2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
            #cnts_list3=cv2.findContours(mask3.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
            #cnts_list4=cv2.findContours(mask4.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
            cv2.rectangle(frame,(roi["upper_x"],roi["upper_y"]),(roi["lower_x"],roi["lower_y"]),(0,255,0),3)

            frame=self.get_enclosingcircle(cnts_list1,frame,"green")
            # frame=get_enclosingcircle(cnts_list2,frame,"red")
            # frame=get_enclosingcircle(cnts_list3,frame,"blue")
            #frame=self.get_enclosingcircle(cnts_list4,frame,"yellow")
            return frame

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', help='which robot: icub|nao', default = "icub")

    parser.add_argument('--nao_ip', help='ip address of nao', default = "127.0.0.1")
    parser.add_argument('--nao_port', help='port of nao', default = 9559, type=int)
    args = parser.parse_args()

    visio = RobotVision(args.robot,args.nao_ip,args.nao_port)
    # input_port = yarp.Port()
    # input_port.open("/python-image-port")
    # yarp.Network.connect("/icubSim/cam/left", "/python-image-port")
    #
    # while(True):
    # # Capture frame-by-frame
    #
    #     yarp_to_python()
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         input_port.close()
    #         break

    # Example demonstrating how to use the SobelFilter class implemented above
    # This assumes iCub simulator is running with world camera at /icubSim/cam
    # and an instance of yarpview with port name /view01

    #image_filter = SobelFilter("/sobel:in", "/sobel:out")

    #try:
    #    assert yarp.Network.connect("/sobel:out", "/view01")
    #    assert yarp.Network.connect("/icubSim/cam", "/sobel:in")

    #    image_filter.run()
    #finally:
    #    image_filter.cleanup()
