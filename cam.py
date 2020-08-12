import cv2
'''
camera = cv2.VideoCapture(0)

success,pic =camera.read()

camera.release()
'''
pic = cv2.imread('jj.jpg')
face_model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faces = face_model.detectMultiScale(pic)


for face in faces:
        x,y,w,h=face
        print(x,y,w,h)
        pic = cv2.rectangle(pic,(x,y),(x+w,y+h),(0,0,255),1)
cv2.imshow("face detection",pic)
cv2.waitKey(0)
cv2.destroyAllWindows()