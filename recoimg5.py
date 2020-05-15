from time import time

import glob
from sys import argv
import cv2
import pickle

import shutil
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn import preprocessing, svm
import numpy as np

## Inputs

k1 = 50
relearnK = False

c1 = 20
svc_kernel = 'rbf' # 'linear'

working_dir = ".\\images"

# phase subdirs
learn_subdir       = "train"
test_subdir        = "test"
validation_subdir  = "valid"

# classes
cat1 = "chat4"
cat2 = "chien3"
cat3 = "formule1"
cat4 = "motoGrandPrix"

baryName = ".\\data\\" # saved files dir

# verbose options

verbose_params = True

verbose_steps = True
verbose_files = False
verbose_state = False

verbose_kmeans = False
verbose_score = True
verbose_confusion = True

## Sequence variables

LEARN = 0
VALIDATE = 1
TEST = 2

stepNameDict = {LEARN: "train",   VALIDATE: "valid",      TEST: "test"}
subDirDict   = {LEARN: learn_subdir, VALIDATE: validation_subdir, TEST: test_subdir}

SEQUENCE = [LEARN, VALIDATE, TEST]

## Running sequence

if (verbose_params):
    print("k1 = %d"%(k1))
    print("c1 = %3f"%(c1))
    print("kernel = %s"%(svc_kernel))

debug = False

for STEP in SEQUENCE:

    if (verbose_steps):
        print("\n"+stepNameDict[STEP]+"...\n")

    subdir = subDirDict[STEP]

    ## Establishing groundTruth
    path_cat_1 = ("%s\\%s\\%s"+"/*.%s") % (working_dir,subdir,cat1,"jpg")
    path_cat_2 = ("%s\\%s\\%s"+"/*.%s") % (working_dir,subdir,cat2,"jpg")
    path_cat_3 = ("%s\\%s\\%s"+"/*.%s") % (working_dir,subdir,cat3,"jpeg")
    path_cat_4 = ("%s\\%s\\%s"+"/*.%s") % (working_dir,subdir,cat4,"jpeg")

    if (verbose_files):
        print("Reading" + 4*"\n\t%s" % (path_cat_1,path_cat_2,path_cat_3,path_cat_4))

    listImg = glob.glob(path_cat_1)
    tmp1 = len(listImg)
    groundTruth = [0]*tmp1

    listImg += glob.glob(path_cat_2)
    tmp2 = len(listImg) - tmp1
    groundTruth += [1]*tmp2

    listImg += glob.glob(path_cat_3)
    tmp3 = len(listImg) - ( tmp1 + tmp2 )
    groundTruth += [2]*tmp3

    listImg += glob.glob(path_cat_4)
    tmp4 = len(listImg) - ( tmp1 + tmp2 + tmp3 )
    groundTruth += [3]*tmp4

    if (verbose_files): print("Done reading" + 4*" %d" % (tmp1, tmp2, tmp3, tmp4))

    ## retrieving SIFT descriptors

    dimImg = [] # nb of sift per file
    lesSift = np.empty(shape=(0, 128), dtype=float) # array of all SIFTS from all images

    for s in listImg:
        if (verbose_files):
            print("###",s,"###")
        image = cv2.imread(s)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp,des = sift.detectAndCompute(gray,None)
        if (verbose_files):
            print("SIFT: ", len(kp))
        dimImg.append(len(des))
        lesSift = np.append(lesSift,des,axis=0)

    ## Computing k-means

    if (verbose_state): print("Done sifting")

    sifts = lesSift ###### preprocessing.normalize(lesSift)

    if ((STEP==LEARN) and relearnK):
        t0 = time()
        kmeans1 = KMeans(n_clusters=k1, random_state=0).fit(sifts)
        print("KMeans completed in %0.3fs" % (time() - t0))

        labels = kmeans1.labels_

        #sauvegarde de l'objet
        with open(baryName+'multiclass.bary', 'wb') as output:
            pickle.dump(kmeans1, output, pickle.HIGHEST_PROTOCOL)

    if (STEP==VALIDATE or STEP==TEST or (not relearnK)):
        #chargement de l'objet
        with open(baryName+'multiclass.bary', 'rb') as input:
            kmeans1 = pickle.load(input)

        # kmeans1 = KMeans(n_clusters=k1, random_state=0, n_init=1, init=bary1).fit(sifts)

        # print("before predict", kmeans1.labels_)
        labels = kmeans1.predict(sifts)
        # print("after predict", labels)



    if (verbose_kmeans): print("result of kmeans k=%d"%(k1), labels)

    ## Writing BOWs from k-means result

    bows = np.empty(shape=(0,k1),dtype=float) #BOW initialization

    i = 0
    for nb in dimImg: # for each sound (file)
        tmpBow = [0]*k1
        j = 0
        while j < nb: # for each SIFT of this sound (file)
            tmpBow[labels[i]] += 1
            j+=1
            i+=1
        copyBow = tmpBow.copy()
        bows = np.append(bows, [copyBow], 0)
    #if verbose:
        # print("BOWs : ", bows)

    # ** Here was the culprit **
    BOWs = bows ###### preprocessing.normalize(bows)

    ## Classification
    # Todo : Tweak 'decision_function_shape'
    classif = svm.SVC(C=c1,kernel=svc_kernel, decision_function_shape='ovr')


    if (STEP==LEARN):
        classif.fit(BOWs, groundTruth) # apprentissage
        #sauvegarde de l'objet
        with open(baryName+'multiclassSVC.svm', 'wb') as output:
            pickle.dump(classif, output, pickle.HIGHEST_PROTOCOL)

    if (STEP==VALIDATE or STEP==TEST):
        #chargement de l'objet
        with open(baryName+'multiclassSVC.svm', 'rb') as input:
            classif = pickle.load(input)

    if (verbose_score):
        print("groundTruth: ", np.array(groundTruth))
        res = classif.predict(BOWs)
        print("prediction : ",res)

        # Todo : Tweak 'average'
        F1 = f1_score(groundTruth, res, average=None) #F1
        print("F-measure =",F1)

        score = classif.score(BOWs, groundTruth)
        print(stepNameDict[STEP],"score = ", score)

    if (verbose_confusion):
        confusion = confusion_matrix(groundTruth, res, labels=[0,1,2,3])
        print(confusion)