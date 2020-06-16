import cv2
import matplotlib.pyplot as plt

camera = cv2.VideoCapture(0)

success,pic =camera.read()

camera.release()

if(success):
	pic = cv2.cvtColor(pic,4)
	plt.imshow(pic)
	plt.show()
	print(pic.shape)
	

face_model = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faces = face_model.detectMultiScale(pic)

print(faces)

for face in faces:
        x,y,w,h=face
        print(x,y,w,h)
        pic = cv2.rectangle(pic,(x,y),(x+w,y+h),(0,255,0),3)
plt.imshow(pic)
plt.axis("off")
plt.show()
