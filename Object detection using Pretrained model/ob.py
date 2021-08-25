import cv2
cap=cv2.VideoCapture("dhaka_traffic.mp4")
names=[]
classfile='coco.txt'
with open(classfile,'rt') as f:
    names=f.read().rstrip('\n').split('\n')
config_path='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model='frozen_inference_graph.pb '

net =cv2.dnn_DetectionModel(frozen_model,config_path)
net.setInputSize(320,320)
net.setInputScale(1.0/127)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)


while True:
    success,img=cap.read()
    classids,confs,bbox=net.detect(img,confThreshold=0.65)
    if len(classids)!=0:
        for x,y,z in zip(classids.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,z,color=(255,255,255),thickness=4 )
            cv2.putText(img,(names[x-1]).title(),(z[0]+10,z[1]+30),
            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            


    cv2.imshow("image",img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

