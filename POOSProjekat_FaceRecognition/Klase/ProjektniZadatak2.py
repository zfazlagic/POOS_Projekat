import cv2
import os
import numpy as np
from sklearn.metrics import confusion_matrix

##### Training Data

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
# Folder_`test-data`_ sluzi za testiranje recognizera kasnije nakon treniranja

# As OpenCV face recognizer accepts labels as integers so we need to define a mapping between integer labels and persons actual names so below I am defining a mapping of persons integer labels and their respective names.




# Moguci slucajevi, odnosno osobe za koje se moze prepoznati lice
subjects = ["Other", "Robert Downey Jr.", "Angelina Jolie"]
# Niz u kojem se cuvaju informacije o prepoznatim licima
predicted_subjects = []
# Testni slucajevi (sluzi samo za testiranje)
val_subjects = ["Angelina Jolie", "Other", "Angelina Jolie", "Other", "Robert Downey Jr.", "Robert Downey Jr.", "Other", "Angelina Jolie", "Robert Downey Jr."]


# ### Pripremanje podataka

# Potrebno je izvrsiti pripremu podataka jer openCV zahtjeva prilagođene podatke. It accepts two vectors, one vector is of faces of all the persons and the second vector is of integer labels for each face so that when processing a face the face recognizer knows which person that particular face belongs too.
#
# Slucaj za dvije osobe, koje imaju dvije slike po klasi
#
# ```
# PERSON-1    PERSON-2
#
# img1        img1
# img2        img2
# ```
#
# Potrebno je izdvojiti lica iz navedenih slika i kreirati labele
#
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

# Detekcija lica korištenjem OpenCV-a

def detect_face(img):
    # Slike se pretvaraju u gray-scale .OpenCV detektor radi sa gray-scale slikama. U ovom slucaju dodatna poboljasanja nisu koristena jer bi se mogao narusiti kvalitet slike koje su trenutno zadovoljavajuce
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Koristenje OpenCV detektora, ovdje se koristi LBP jer je brzi
    # Moze se koristiti i Haar (koji je sporiji)
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    # Multiscale -- neke slike mogu biti blize kameri nego ostale
    # Rezultat je lista lica
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=6);

    # Slucaj da nije otkriveno lice
    if (len(faces) == 0):
        return None, None

    # Ukoliko postoji samo jedno lice ekstraktovati podrucje lica
    (x, y, w, h) = faces[0]

    # vratiti samo dio slike oko lica
    return gray[y:y + w, x:x + h], faces[0]


# Funckija cita slike za treniranje, detektuje lica na svim slikama
# vraca dvije liste iste duzine gdje jedna lista predstavlja lica a druga labele za lica

def prepare_training_data(data_folder_path):

    # dohvatanje direktorija -> jedan direktorij za jedan subjekt prepoznavanja
    dirs = os.listdir(data_folder_path)

    # lista koja sadrzi lica
    faces = []
    # lista koja sadrzi labele
    labels = []

    # prolazak kroz direktorije i citanje slika iz direktorija
    for dir_name in dirs:


        # ignorisu se direktoriji koji ne pocinju slovom s

        if not dir_name.startswith("s"):
            continue;


        # uzima se broj za labelu iz naziva direktorija subjekta koji ce se koristiti za odredjivanje iz liste subjects
        # format imena direktorija je slabel

        label = int(dir_name.replace("s", ""))


        # kreiranje putanje direktorija koji sadrzi slike za određeni subjekat
        subject_dir_path = data_folder_path + "/" + dir_name

        # dohvati imena slika koja su u direktoriju
        subject_images_names = os.listdir(subject_dir_path)


        #prolazak kroz slike, otkrivanje lica i dodavanja u listu lica
        for image_name in subject_images_names:

            # ignorisanje sistemskih datoteka, u slucaju da se desi da postoji slucajno
            if image_name.startswith("."):
                continue;

            #kreiranje putanje za slike
            image_path = subject_dir_path + "/" + image_name

            # citanje slike
            image = cv2.imread(image_path)

            # Prikazati sliku za treniranje
            #cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            #cv2.waitKey(100)

            # detekcija lica
            face, rect = detect_face(image)


            # Lica koja nisu otkrivena biti ce ignorisana i ispisana kako bi se mogao u buducnosti popraviti DataSet
            if face is not None:
                # dodati lice u listu lica
                faces.append(face)
                # dodati labelu za lice u listu labela
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





# jedna lista sadrzi lica, a druga lista sadrzi listu labela za lica. Obje liste su iste duzine
# potrebno je pripremiti podatke za treniranje

print("Preparing data...")
faces, labels = prepare_training_data("../DataSetPOOS/training-data")
print("Data prepared")
print(labels)

# Ukupan broj lica i labela
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))


# ### Treniranje Face Recognizer-a

# Postoje razlicite vrste Face Recognizera
#
# 1. EigenFace Recognizer: This can be created with `cv2.face.createEigenFaceRecognizer()`
# 2. FisherFace Recognizer: This can be created with `cv2.face.createFisherFaceRecognizer()`
# 3. Local Binary Patterns Histogram (LBPH): This can be created with `cv2.face.LBPHFisherFaceRecognizer()`
##########################################################################################################



# kreiranje LBPH face recognizer-a
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# ili EigenFaceRecognizer
#face_recognizer = cv2.face.EigenFaceRecognizer_create()

# ili FisherFaceRecognizer
# face_recognizer = cv2.face.FisherFaceRecognizer_create()

# treniranje face_recognizera

face_recognizer.train(faces, np.array(labels))

#export face recognizera na navedenu lokaciju
face_recognizer.write('../trainner/trainner.yml')

# Predikcija ( u ovom kodu koristila se kako bi stekli utisak da li smo dobro izvrsili treniranje 2-p-z


#Sljedece dvije funkcije koristene su iz openCV paketa
# funkcija za crtanje koristeci koordinate, visinu i sirinu

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# funkcija za ispis teksta

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

# Funkcija vrsi prepoznavanje lica te ispis imena i obiljezavanje lica
def predict(test_img):
    # kopija slike kako se ne bi ostetio original
    img = test_img.copy()
    # pronalazak lica na slici
    face, rect = detect_face(img)

    # predikcija je uradjena koristenjem face_recognizera
    #FACE_RECOGNIZER URAĐEN SA EXPORTOM .save, I OVU FUNKCIJU CIJELU PREBACITI U DRUGI PY
    label, confidence = face_recognizer.predict(face)
    # ime se uzima koristenjem labele kao indexa iz niza subjekata
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

#################################

########
# Main #
########

test_path = "../test/"
dirs = os.listdir(test_path)
images_names = os.listdir(test_path)
print(images_names)
slike = []
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


### Performanse main ###

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









