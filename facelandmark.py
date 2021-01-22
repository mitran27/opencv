import cv2
import dlib
import numpy 
""" to import dlib basic requirment are visual studio with c++ build tools
cmake with its path in evironment variables """
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()




def get_landmarks(im):
    im=im.copy()
    rects =detector(im,1)
    
    for face in rects:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(im, (x, y), (x + w, y + h), (0,0,255), 2)
      
   
    
    cv2.imshow('face', im)
    cv2.waitKey(0)
    print(numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()]))
   
   
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for index, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(1), pos,fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.4,color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

image = cv2.imread('./images/trump.jpg')
cv2.imshow('initial', image)
cv2.waitKey(0)
landmarks = get_landmarks(image)

image_with_landmarks = annotate_landmarks(image, landmarks)



cv2.imshow('landmarks', image_with_landmarks)
cv2.waitKey(0)

cv2.destroyAllWindows()