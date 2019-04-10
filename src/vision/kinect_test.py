#import the necessary modules
import cv2
import numpy as np
import imutils
from cv2 import *
import cv2 as cv
import time
import yarp
import numpy
import matplotlib.pylab as plt

# Camera settings
#cam = VideoCapture(0)   # arg -> index of camera

cam = cv2.VideoCapture(0)
print cam.isOpened()

# Initialise YARP
yarp.Network.init()

# Initialize Ports and Variables
# Create a port and connect it to an external port
input_port = yarp.Port()
input_port.open("/python-image-port")
yarp.Network.connect("/WebCamimage/out", "/python-image-port")

def yarp_send():
    img_array = numpy.random.uniform(0., 255., (240, 320)).astype(numpy.float32)
    yarp_image = yarp.ImageFloat()
    yarp_image.setExternal(img_array.__array_interface__['data'][0], img_array.shape[1], img_array.shape[0])
    output_port = yarp.Port()
    output_port.open("/python-image-port")
    yarp.Network.connect("/python-image-port", "/view01")
    output_port.write(yarp_image)
    # Cleanup
    output_port.close()
    return image

def yarp_to_python():
    # Create a port and connect it to the iCub simulator virtual camera
    input_port = yarp.Port()
    input_port.open("/python-image-port")
    yarp.Network.connect("/icubSim/cam", "/python-image-port")

    # Create numpy array to receive the image and the YARP image wrapped around it
    img_array = numpy.zeros((240, 320, 3), dtype=numpy.uint8)
    yarp_image = yarp.ImageRgb()
    yarp_image.resize(320, 240)
    yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
    
    # Alternatively, if using Python 2.x, try:
    # yarp_image.setExternal(img_array.__array_interface__['data'][0], img_array.shape[1], img_array.shape[0])

    # Read the data from the port into the image
    input_port.read(yarp_image)

    # display the image that has been read
    matplotlib.pylab.imshow(img_array)

    # Cleanup
    input_port.close()

def yarp_get():
    img_array = numpy.ones((480, 640, 3), dtype=numpy.uint8)

    source = img_array
    bitmap = cv.CreateImageHeader((source.shape[1], source.shape[0]), cv.IPL_DEPTH_8U, 3)
    cv.SetData(bitmap, source.tostring(), source.dtype.itemsize * 3 * source.shape[1])
    img_array = bitmap

    yarp_image = yarp.ImageRgb()
    yarp_image.resize(640, 480)
    yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
#    print img_array.__array_interface__['data'][0]
    #yarp_image.setExternal(img_array.__array_interface__['data'][0], img_array.shape[1], img_array.shape[0])

    input_port.read(yarp_image)

#    print img_array.getIplImage()
    plt.imshow(img_array)
    #plt.show()
    return img_array

#function to get RGB image from webcam
def get_video():
    s, array = cam.read()
    return array

#function to get the enclosing circle of a region
def get_enclosingcircle(cnts_list,frame,colorid):
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

    if len(cnts_list)>1:
        for cnts in cnts_list:
            #cnts=cnts_list
            if c<cv2.contourArea(cnts):#max(cnts,key=cv2.contourArea)
                ((x,y),radius)=cv2.minEnclosingCircle(cnts)
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
#                print "Color: ",colorid, " location: ",center
    return frame

if __name__ == "__main__":
    while 1:
        #get a frame from RGB camera

        start = time.time()
        #image = get_video()
       # print image
        image = yarp_get()
        end = time.time()
#    print ("time1")
#        print(end - start)

        start = time.time()
    #Tracking Point
        greenLower=(35,86,6)
        greenUpper=(64,255,255)
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
        cnts_list2=cv2.findContours(mask2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts_list3=cv2.findContours(mask3.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts_list4=cv2.findContours(mask4.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

        frame=get_enclosingcircle(cnts_list1,frame,"green")
        frame=get_enclosingcircle(cnts_list2,frame,"red")
        frame=get_enclosingcircle(cnts_list3,frame,"blue")
        frame=get_enclosingcircle(cnts_list4,frame,"yellow")
        end = time.time()
        #       print ("time2")
        #        print(end - start)

        #display RGB image
        cv2.imshow('RGB image',frame)
        #display depth image
        #cv2.imshow('Depth image',depth)

        # quit program when 'esc' key is pressed
        #time.sleep(1)
        k = cv2.waitKey(5) & 0xFF
        #if k == 27:
        #    break
cv2.destroyAllWindows()
