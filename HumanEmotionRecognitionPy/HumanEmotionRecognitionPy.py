import cv2
import glob
import random
import numpy as np
import os.path

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

emotions = ["neutral", "anger", "disgust", "fear", "joy", "sadness", "surprise"]
fishface = cv2.createFisherFaceRecognizer() #Initialize fisher face classifier

def dataset_preprocessing(emotion):
    files = glob.glob("JAFFESet\\%s\\*" %emotion) #Get list of all images with emotion

    filenumber = 0
    print "Polishing Dataset..."

    for f in files:
        frame = cv2.imread(f) #Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
        
        #Detect face using 4 different classifiers
        face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
        face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

        #Go over detected faces, stop at first detected face, return empty if no face.
        if len(face) == 1:
            facefeatures = face
        elif len(face_two) == 1:
            facefeatures = face_two
        elif len(face_three) == 1:
            facefeatures = face_three
        elif len(face_four) == 1:
            facefeatures = face_four
        else:
            facefeatures = ""
        
        #Cut and save face
        for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
            
            gray = gray[y:y+h, x:x+w] #Cut the frame to size
            
            try:
                out = cv2.resize(gray, (350, 350)) #Resize face so all images have same size
                cv2.imwrite("JAFFESet_final\\%s\\%s.jpg" %(emotion, filenumber), out) #Write image
            except:
               pass #If error, pass file
        filenumber += 1 #Increment image number
    print "Done!"

def detect_faces(gray):
    out = [] 
       
    #Detect face using 4 different classifiers
    face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

    #Go over detected faces, stop at first detected face, return empty if no face.
    if len(face) > 0:
        facefeatures = face
    elif len(face_two) > 0:
        facefeatures = face_two
    elif len(face_three) > 0:
        facefeatures = face_three
    elif len(face_four) > 0:
        facefeatures = face_four
    else:
        facefeatures = ""

    
    x = 0
    #Cut and save face
    for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
        
        roi_gray = gray[y:y+h, x:x+w]
        out.append((cv2.resize(roi_gray, (350, 350)),x, y, w, h))

        #cv2.imshow(str(x),out)
        x += 1
    return out


def get_files(model,emotion):
    
    files = glob.glob("%s\\%s\\*" %(model,emotion))
    return files

def make_set(model,setName):
    data = []
    labels = []
    
    for emotion in emotions:
        files = get_files(model,emotion)
        n = 0

        #Append data to training and prediction list, and generate labels 0-7
        for item in files:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            out = cv2.resize(gray, (350, 350))
            data.append(out) #append image array to training data list
            labels.append(emotions.index(emotion))
            n += 1
            
        #print (emotions.index(emotion))
        print ("%i images detected in the %s folder" % (n, emotion))
        #cv2.namedWindow(emotion, cv2.WINDOW_NORMAL)
        #cv2.imshow(emotion,gray)
        

    fishface.train(data, np.asarray(labels))
    fishface.save(setName)

def emotion_recognizer(face):
    pred, conf = fishface.predict(face)
    if pred == 0:
        return (pred, conf)
    elif pred == 1:
        return (pred, conf)
    elif pred == 2:
        return (pred, conf)
    elif pred == 3:
        return (pred, conf)
    elif pred == 4:
        return (pred, conf)
    elif pred == 5:
        return (pred, conf)
    elif pred == 6:
        return (pred, conf)

def run_recognizer(image):
    setName = ""
    model = ""
    modelN = raw_input("Choose model: (0 for CK set, 1 for GoogleSet set, 2 for JAFFE set, 3 for MUG set)")
    if int(modelN) == 0:
        setName = "model_CK.xml"
        model = "CKset_final"
        print "CK set selected."
    elif int(modelN) == 1:
         setName = "model_Google.xml"
         model = "GoogleSet_final"
         print "Google set selected."
    elif int(modelN) == 2:
         setName = "model_JAFFE.xml"
         model = "JAFFESet_final"
         print "JAFFE set selected."
    elif int(modelN) == 3:
         setName = "model_MUG.xml"
         model = "MUGSet_final"
         print "MUG set selected."

    if (os.path.isfile(setName) and os.path.getsize(setName) > 0):
        print "Loading model..."
        fishface.load(setName)

        print "Loading image..."
        frame = cv2.imread(image) #Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
        x = 0
        print "Detecting faces and emotions..."
        result = detect_faces(gray)
        for face,x, y, w, h in result:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            emo,conf = emotion_recognizer(face)
            cv2.putText(frame,"%s" %(emotions[emo]), (x,y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0),1)
            #cv2.imshow("Detected %s - %i" % (emotions[emo],x), frame) 
            cv2.imshow("Detected faces and emotions", frame) 
            x += 1
        
    else:
        print "Model not found."
        print "Generating model..."
        make_set(model,setName)
        run_recognizer(image)



#run_recognizer("16.jpg")
run_recognizer("happy-people.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()

#for emotion in emotions: 
#    dataset_preprocessing(emotion) 