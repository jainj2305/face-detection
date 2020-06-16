import cv2
import matplotlib.pyplot as plt

camera = cv2.VideoCapture(0)  #0 is passed for primary camera you can change it with external camera by passing 1 2 ..

success,pic =camera.read()  #starts camera of your system

camera.release()

if(success):
	pic = cv2.cvtColor(pic,4)
	plt.imshow(pic)
	plt.show()
	print(pic.shape)
	

face_model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  #detect faces
faces = face_model.detectMultiScale(pic)

print(faces)

for face in faces:
        x,y,w,h=face
        print(x,y,w,h)
        pic = cv2.rectangle(pic,(x,y),(x+w,y+h),(0,255,0),3)
plt.imshow(pic)
plt.axis("off")
plt.show()
