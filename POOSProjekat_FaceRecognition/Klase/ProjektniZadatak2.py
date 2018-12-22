import cv2
import os
import numpy as np
from sklearn.metrics import confusion_matrix

# ### Training Data

# Za poboljsanje treba se dodati vise fotografija u DataSet
# training-data (Organizacija foldera)
# |-------------- s1
# |               |-- 1.jpg
# |               |-- ...
# |               |-- 12.jpg
# |-------------- s2
# |               |-- 1.jpg
# |               |-- ...
# |               |-- 12.jpg
# ```
#
# Folder_`test-data`_ sluzi ta testiranje recognizera kasnije nakon treniranja

# As OpenCV face recognizer accepts labels as integers so we need to define a mapping between integer labels and persons actual names so below I am defining a mapping of persons integer labels and their respective names.
#
# **Note:** As we have not assigned `label 0` to any person so **the mapping for label 0 is empty**.



# Moguci slucajevi, odnosno osobe za koje se moze prepoznati lice
subjects = ["Other", "Robert Downey Jr.", "Angelina Jolie"]
predicted_subjects = []
val_subjects = ["Angelina Jolie", "Other", "Angelina Jolie", "Other", "Robert Downey Jr.", "Robert Downey Jr.", "Other", "Angelina Jolie", "Robert Downey Jr."]


# ### Prepare training data

# Potrebno je izvrsiti pripremu podataka jer openCV zahtjeva prilagođene podatke. It accepts two vectors, one vector is of faces of all the persons and the second vector is of integer labels for each face so that when processing a face the face recognizer knows which person that particular face belongs too.
#
# For example, if we had 2 persons and 2 images for each person.
#
# ```
# PERSON-1    PERSON-2
#
# img1        img1
# img2        img2
# ```
#
# Then the prepare data step will produce following face and label vectors.
#
# ```
# FACES                        LABELS
#
# person1_img1_face              1
# person1_img2_face              1
# person2_img1_face              2
# person2_img2_face              2
# ```
#
#
# Preparing data step can be further divided into following sub-steps.
#
# 1. Read all the folder names of subjects/persons provided in training data folder. So for example, in this tutorial we have folder names: `s1, s2`.
# 2. For each subject, extract label number. **Do you remember that our folders have a special naming convention?** Folder names follow the format `sLabel` where `Label` is an integer representing the label we have assigned to that subject. So for example, folder name `s1` means that the subject has label 1, s2 means subject label is 2 and so on. The label extracted in this step is assigned to each face detected in the next step.
# 3. Read all the images of the subject, detect face from each image.
# 4. Add each face to faces vector with corresponding subject label (extracted in above step) added to labels vector.
#
# **[There should be a visualization for above steps here]**

# Did you read my last article on [face detection](https://www.superdatascience.com/opencv-face-detection/)? No? Then you better do so right now because to detect faces, I am going to use the code from my previous article on [face detection](https://www.superdatascience.com/opencv-face-detection/). So if you have not read it, I encourage you to do so to understand how face detection works and its coding. Below is the same code.



# Detekcija lica korištenjem OpenCV-a

def detect_face(img):
    # Slike se pretvaraju u gray-scale .OpenCV detektor radi sa gray-scale slikama
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Koristenje OpenCV detektora, ovdje se koristi LBP jer je brz
    # Moze se koristiti Haar koji je sporiji
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    # let's detect multiscale (some images may be closer to camera than others) images
    # Rezultat je lista lica
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=5);

    # Slucaj da nije otkriveno lice
    if (len(faces) == 0):
        return None, None

    # Ukoliko postoji samo jedno lice ekstraktovati podrucje lica
    (x, y, w, h) = faces[0]

    # vratiti samo dio slike oko lica
    return gray[y:y + w, x:x + h], faces[0]


# I am using OpenCV's **LBP face detector**. On _line 4_, I convert the image to grayscale because most operations in OpenCV are performed in gray scale, then on _line 8_ I load LBP face detector using `cv2.CascadeClassifier` class. After that on _line 12_ I use `cv2.CascadeClassifier` class' `detectMultiScale` method to detect all the faces in the image. on _line 20_, from detected faces I only pick the first face because in one image there will be only one face (under the assumption that there will be only one prominent face). As faces returned by `detectMultiScale` method are actually rectangles (x, y, width, height) and not actual faces images so we have to extract face image area from the main image. So on _line 23_ I extract face area from gray image and return both the face image area and face rectangle.




# Funckija cita slike za treniranje, detektuje lica na svim slikama
# vraca dvije liste iste duzine gdje jedna lista predstavlja lica a druga labele za lica
def prepare_training_data(data_folder_path):
    # ------STEP-1--------
    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    # let's go through each directory and read images within it
    for dir_name in dirs:

        # our subject directories start with letter 's' so
        # ignore any non-relevant directories if any

        if not dir_name.startswith("s"):
            continue;

        # ------STEP-2--------
        # extract label number of subject from dir_name
        # format of dir name = slabel
        # , so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))

        # build path of directory containin images for current subject subject
        # sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        # ------STEP-3--------
        # go through each image name, read image,
        # detect face and add face to list of faces
        for image_name in subject_images_names:

            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;

            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)

            # display an image window to show the image
            #cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            #cv2.waitKey(100)

            # detect face
            face, rect = detect_face(image)

            # ------STEP-4--------
            # for the purpose of this tutorial
            # we will ignore faces that are not detected
            if face is not None:
                # add face to list of faces
                faces.append(face)
                # add label for this face
                labels.append(label)
            else:
                print(image_name)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels


# I have defined a function that takes the path, where training subjects' folders are stored, as parameter. This function follows the same 4 prepare data substeps mentioned above.
#
# **(step-1)** On _line 8_ I am using `os.listdir` method to read names of all folders stored on path passed to function as parameter. On _line 10-13_ I am defining labels and faces vectors.
#
# **(step-2)** After that I traverse through all subjects' folder names and from each subject's folder name on _line 27_ I am extracting the label information. As folder names follow the `sLabel` naming convention so removing the  letter `s` from folder name will give us the label assigned to that subject.
#
# **(step-3)** On _line 34_, I read all the images names of of the current subject being traversed and on _line 39-66_ I traverse those images one by one. On _line 53-54_ I am using OpenCV's `imshow(window_title, image)` along with OpenCV's `waitKey(interval)` method to display the current image being traveresed. The `waitKey(interval)` method pauses the code flow for the given interval (milliseconds), I am using it with 100ms interval so that we can view the image window for 100ms. On _line 57_, I detect face from the current image being traversed.
#
# **(step-4)** On _line 62-66_, I add the detected face and label to their respective vectors.

# But a function can't do anything unless we call it on some data that it has to prepare, right? Don't worry, I have got data of two beautiful and famous celebrities. I am sure you will recognize them!
#
# ![training-data](visualization/tom-shahrukh.png)
#
# Let's call this function on images of these beautiful celebrities to prepare data for training of our Face Recognizer. Below is a simple code to do that.

# In[5]:

# let's first prepare our training data
# data will be in two lists of same size
# one list will contain all the faces
# and other list will contain respective labels for each face
print("Preparing data...")
faces, labels = prepare_training_data("../DataSetPOOS/training-data")
print("Data prepared")
print(labels)

# print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

# This was probably the boring part, right? Don't worry, the fun stuff is coming up next. It's time to train our own face recognizer so that once trained it can recognize new faces of the persons it was trained on. Read? Ok then let's train our face recognizer.

# ### Train Face Recognizer

# As we know, OpenCV comes equipped with three face recognizers.
#
# 1. EigenFace Recognizer: This can be created with `cv2.face.createEigenFaceRecognizer()`
# 2. FisherFace Recognizer: This can be created with `cv2.face.createFisherFaceRecognizer()`
# 3. Local Binary Patterns Histogram (LBPH): This can be created with `cv2.face.LBPHFisherFaceRecognizer()`
#
# I am going to use LBPH face recognizer but you can use any face recognizer of your choice. No matter which of the OpenCV's face recognizer you use the code will remain the same. You just have to change one line, the face recognizer initialization line given below.

# In[6]:

# create our LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# or use EigenFaceRecognizer by replacing above line with
#face_recognizer = cv2.face.EigenFaceRecognizer_create()

# or use FisherFaceRecognizer by replacing above line with
# face_recognizer = cv2.face.FisherFaceRecognizer_create()


# Now that we have initialized our face recognizer and we also have prepared our training data, it's time to train the face recognizer. We will do that by calling the `train(faces-vector, labels-vector)` method of face recognizer.

# In[7]:

# train our face recognizer of our training faces

#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
face_recognizer.save('../trainner/trainner.yml')

# **Did you notice** that instead of passing `labels` vector directly to face recognizer I am first converting it to **numpy** array? This is because OpenCV expects labels vector to be a `numpy` array.
#
# Still not satisfied? Want to see some action? Next step is the real action, I promise!

# ### Prediction

# Now comes my favorite part, the prediction part. This is where we actually get to see if our algorithm is actually recognizing our trained subjects's faces or not. We will take two test images of our celeberities, detect faces from each of them and then pass those faces to our trained face recognizer to see if it recognizes them.
#
# Below are some utility functions that we will use for drawing bounding box (rectangle) around face and putting celeberity name near the face bounding box.

# In[8]:

# function to draw rectangle on image
# according to given (x, y) coordinates and
# given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# function to draw text on give image starting from
# passed (x, y) coordinates.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


# First function `draw_rectangle` draws a rectangle on image based on passed rectangle coordinates. It uses OpenCV's built in function `cv2.rectangle(img, topLeftPoint, bottomRightPoint, rgbColor, lineWidth)` to draw rectangle. We will use it to draw a rectangle around the face detected in test image.
#
# Second function `draw_text` uses OpenCV's built in function `cv2.putText(img, text, startPoint, font, fontSize, rgbColor, lineWidth)` to draw text on image.
#
# Now that we have the drawing functions, we just need to call the face recognizer's `predict(face)` method to test our face recognizer on test images. Following function does the prediction for us.



# this function recognizes the person in image passed
# and draws a rectangle around detected face with name of the
# subject
def predict(test_img):
    # kopija slike kako se ne bi stetio original
    img = test_img.copy()
    # pronalazak lica na slici
    face, rect = detect_face(img)

    # predict the image using our face recognizer
    #FACE_RECOGNIZER URAĐEN SA EXPORTOM .save, I OVU FUNKCIJU CIJELU PREBACITI U DRUGI PY
    label, confidence = face_recognizer.predict(face)
    # get name of respective label returned by face recognizer
    label_text = subjects[label]

    # crtanje kocke oko detektovanog lica
    draw_rectangle(img, rect)
    # ispis imena
    draw_text(img, label_text, rect[0], rect[1] - 5)
    predicted_subjects.append(label_text)

    return img


# Testiranje nad testnim slikama se vrsi pozivajuci funkciju predict

print("Predicting images...")

# load test images (testiranje rucno)
# test_img1 = cv2.imread("../test/Robert-Downey-Jr-120109-Sherlock-Holmes-5.jpg")
# test_img2 = cv2.imread("../test/51I5N-2UzhL.jpg")

# Pozivanje funkcije predict (testiranje rucno)
# predicted_img1 = predict(test_img1)
# predicted_img2 = predict(test_img2)

# Prikaz rezultata
#cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
#cv2.imshow(subjects[0], cv2.resize(predicted_img2, (400, 500)))

##############################
### TESTIRANJE PERFORMANSI ###
##############################

def acc(ind, mat):
    TP = mat[ind][ind]
    TN = 0
    ALL = 0
    for i in range(len(mat[0])):
        for j in range(len(mat[0])):
            ALL += mat[i][j]
            if i == ind or j == ind:
                continue
            TN += mat[i][j]
    return (TP + TN) / ALL

def sens(ind, mat):
    TP = mat[ind][ind]
    FN = 0
    for i in range(len(mat[0])):
        if i != ind:
            FN += mat[ind][i]
    return TP/(TP+FN)

def spec(ind, mat):
    TN = 0
    FP = 0
    for i in range(len(mat[0])):
        if i == ind:
            continue
        for j in range(len(mat[0])):
            if j == ind:
                continue
            TN += mat[i][j]
        FP += mat[i][ind]
    return TN/(TN+FP)





########
# Main #
########

test_path = "../test/"
dirs = os.listdir(test_path)
images_names = os.listdir(test_path)
print(images_names)
slike = []

#Acc brojPogodaka / Total

brojac = 0





for image_name in images_names:
    image_path = test_path + "/" + image_name
    image = cv2.imread(image_path)
    predicted_img = predict(image)
    slike.append(predicted_img)
    #cv2.imshow(subjects[0], cv2.resize(predicted_img, (400, 500)))
    #cv2.waitKey(5000)

for slika in slike:
    cv2.imshow("Face classified", cv2.resize(slika, (400, 500)))
    cv2.waitKey(0)


for k in range(len(val_subjects)):
    if val_subjects[k] != predicted_subjects[k]:
        brojac+=1


acc_ukupno = (len(predicted_subjects) - brojac) / len(val_subjects)

predicted = []
print(predicted_subjects)


confusion = confusion_matrix(val_subjects, predicted_subjects)
#print(confusion)
for i in range(len(confusion)):
    print("Class " + str(i) + ": ", end=" ")
    print("sensitivity: " + str(sens(i, confusion)) + ", specificity: " + str(spec(i, confusion)) + ", accuracy: " + str(acc(i, confusion)))


print("Prediction complete")




#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.waitKey(1000)
#cv2.destroyAllWindows()








