
#
#
#The method to run this code is  BoW.py --train_path dataset/train --test_path dataset/test --no_clusters 100 --kernel precomputed
#
#
import argparse
import cv2
import numpy as np 
import os
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from skimage.feature import hog

def getFiles(train, path):
    images = []
    count = 0
    for folder in os.listdir(path):
        for file in  os.listdir(os.path.join(path, folder)):
            images.append(os.path.join(path, os.path.join(folder, file)))

    if(train is True):
        np.random.shuffle(images)
    
    return images

def getDescriptors_gray(sift, img):

    kp, des = sift.detectAndCompute(img[:,:,0], None)
    img=cv2.drawKeypoints(img[:,:,0],kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('sift_keypoints.jpg',img)
    return des

def getDescriptors_rgb(sift, img):

    #kp, des = sift.detectAndCompute(img, None)
    sift = cv2.SIFT_create()
    kp1 = sift.detect(img[:,:,0],None)
    kp2 = sift.detect(img[:,:,1],None)
    kp3 = sift.detect(img[:,:,2],None)
    
    kp=np.concatenate ((kp1,kp2,kp3))
    
    kp, des = sift.compute(img,kp)
    return des
    
    
def getDescriptors_surf(sift, img):
    surf = cv2.SURF(400)
    kp, des = surf.detectAndCompute(img[:,:,0],None)
    return des

def getDescriptors_hog(hog1, img):
    #hog = cv2.HOGDescriptor()
    #des = hog.compute(img)
    
    
    kp, des = hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(1, 1), visualize=True, multichannel=True)
    #(img, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True, multichannel=True)
    
    return des

def readImage(img_path):
    img = cv2.imread(img_path,1)
    #print(img.shape)
    return img
    #cv2.resize(img,(150,150))

def vstackDescriptors(descriptor_list):
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor)) 

    return descriptors

def clusterDescriptors(descriptors, no_clusters):
    kmeans = KMeans(n_clusters = no_clusters).fit(descriptors)
    return kmeans

def extractFeatures(kmeans, descriptor_list, image_count, no_clusters):
    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, 128)
            #feature.array.reshape(-1, 1)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1

    return im_features

def normalizeFeatures(scale, features):
    return scale.transform(features)

def plotHistogram(im_features, no_clusters):
    x_scalar = np.arange(no_clusters)
    y_scalar = np.array([abs(np.sum(im_features[:,h], dtype=np.int32)) for h in range(no_clusters)])

    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Complete Vocabulary Generated")
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.show()

def svcParamSelection(X, y, kernel, nfolds):
    Cs = [0.5, 0.1, 0.15, 0.2, 0.3]
    gammas = [0.1, 0.11, 0.095, 0.105]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel=kernel), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

def findSVM(im_features, train_labels, kernel):
    features = im_features
    if(kernel == "precomputed"):
      features = np.dot(im_features, im_features.T)
    
    params = svcParamSelection(features, train_labels, kernel, 2)
    C_param, gamma_param = params.get("C"), params.get("gamma")
    print(C_param, gamma_param)
  
    svm = SVC(kernel = kernel, C =  C_param, probability=True, gamma = gamma_param)
    print('features: ',features.shape)
    svm.fit(features, train_labels)
    return svm



def trainModel(path, no_clusters, kernel,SIFT_type):
    images = getFiles(True, path)
    #print("Train images path detected.")
    sift = cv2.SIFT_create()
    descriptor_list = []
    train_labels = np.array([])
    label_count = 7
    image_count = len(images)

    for img_path in images:
        if("airplanes" in img_path):
            class_index = 0
        elif("birds" in img_path):
            class_index = 1
        elif("cars" in img_path):
            class_index = 2
        elif("horses" in img_path):
            class_index = 3
        elif("ships" in img_path):
            class_index = 4
        else:
          class_index = 5

        train_labels = np.append(train_labels, class_index)
        #print('class_index: ',class_index)
        img = readImage(img_path)
        
        #choose SIFT or RGB-SIFT
        if (SIFT_type=='grayscale SIFT'):
            des = getDescriptors_gray(sift, img)
        elif (SIFT_type=='RGBscale SIFT'):
            des = getDescriptors_rgb(sift, img)
        elif (SIFT_type=='SURF'):
            des = getDescriptors_surf(sift, img)
        elif (SIFT_type=='HOG'):
            des = getDescriptors_hog(sift, img)
            
        descriptor_list.append(des)

    descriptors = vstackDescriptors(descriptor_list)
    print("Descriptors vstacked.")

    kmeans = clusterDescriptors(descriptors, no_clusters)
    print("Descriptors clustered.")

    im_features = extractFeatures(kmeans, descriptor_list, image_count, no_clusters)
    print("Images features extracted.")

    scale = StandardScaler().fit(im_features)        
    im_features = scale.transform(im_features)
    print("Train images normalized.")
    
    
    #print plotHistogram here
    plotHistogram(im_features, no_clusters)
    print("Features histogram plotted.")

    svm = findSVM(im_features, train_labels, kernel)
    print("SVM fitted.")
    print("Training completed.")

    return kmeans, scale, svm, im_features
    
def mAP(kernel_test,classs):
    counter = 0
    total_sum = 0
    for idx, val in enumerate(kernel_test):
        
        if ( val == classs):
            counter = counter + 1
            total_sum = total_sum + (counter / (idx+1))
    mAP = total_sum/10
    print("mAP of ",classs," : ",mAP)
    return mAP
    
    
def map_image(mAP_value1,mAP_value2):

    label_list = ['Airplanes', 'Birds', 'cars', 'Horses', 'Ships']    # show x label value
    num_list1 = mAP_value1      # show y label value 
    num_list2 = mAP_value2      # show y label value 
    x = range(len(num_list1))
    
    rects1 = plt.bar(x, height=num_list1, width=0.4, alpha=0.8, color='blue', label="SIFT_gray")
    rects1 = plt.bar([i + 0.4 for i in x], height=num_list2, width=0.4, alpha=0.8, color='red', label="SIFT_rgb")
    plt.xlabel('Differnt class')
    plt.xticks([index + 0.2 for index in x], label_list)
    #plt.ylim(0, 50)     
    plt.ylabel("mAP")
    plt.title("Classification comparsion between SIFT_RGB and SIFT_gray")
    #plt.xticks([index + 0.2 for index in x], label_list)
    plt.legend()
    plt.show()
    
def map_image3(mAP_value1,mAP_value2,mAP_value3):

    label_list = ['Airplanes', 'Birds', 'cars', 'Horses', 'Ships']    # show x label value
    num_list1 = mAP_value1      # show y label value 
    num_list2 = mAP_value2      # show y label value 
    num_list3 = mAP_value3
    x = range(len(num_list1))
    
    rects1 = plt.bar(x, height=num_list1, width=0.2, alpha=0.8, color='blue', label="Vocabulary size 400")
    rects1 = plt.bar([i + 0.2 for i in x], height=num_list2, width=0.2, alpha=0.8, color='red', label="Vocabulary size 1000")
    rects1 = plt.bar([i + 0.4 for i in x], height=num_list3, width=0.2, alpha=0.8, color='green', label="Vocabulary size 4000")
    plt.xticks([index + 0.2 for index in x], label_list)
    plt.xlabel('Differnt class')
    
    plt.ylabel("mAP")
    plt.title("Classification comparsion between different vocabulary size")
    #plt.xticks([index + 0.2 for index in x], label_list)
    plt.legend()
    plt.show()


def testModel(path, kmeans, scale, svm, im_features, no_clusters, kernel, SIFT_type):
    test_images = getFiles(False, path)
    print("Test images path detected.")

    count = 0
    true = []
    descriptor_list = []

    name_dict =	{
        "0": "airplanes",
        "1": "birds",
        "2": "cars",
        "3": "horses",
        "4": "ships",
    }

    sift = cv2.SIFT_create()

    for img_path in test_images:
        img = readImage(img_path)
        
        
        #choose SIFT or RGB-SIFT
        if (SIFT_type=='grayscale SIFT'):
            des = getDescriptors_gray(sift, img)
        elif (SIFT_type=='RGBscale SIFT'):
            des = getDescriptors_rgb(sift, img)
        elif (SIFT_type=='SURF'):
            des = getDescriptors_surf(sift, img)
        elif (SIFT_type=='HOG'):
            des = getDescriptors_hog(sift, img)


        if(des is not None):
            count += 1
            descriptor_list.append(des)

            if("airplanes" in img_path):
                true.append("airplanes")
            elif("birds" in img_path):
                true.append("birds")
            elif("cars" in img_path):
                true.append("cars")
            elif("horses" in img_path):
                true.append("horses")
            elif("ships" in img_path):
                true.append("ships")
            else:
                true.append("else")

    descriptors = vstackDescriptors(descriptor_list)

    test_features = extractFeatures(kmeans, descriptor_list, count, no_clusters)

    test_features = scale.transform(test_features)
    
    kernel_test = test_features
    if(kernel == "precomputed"):
        kernel_test = np.dot(test_features, im_features.T)
    
    predictions = [name_dict[str(int(i))] for i in svm.predict(kernel_test)]
    print('predictions is:',predictions)

    
    # Here do the calculation of mAP
    #for i in svm.predict_proba(kernel_test):
    #    print('predict_proba is:',i)
     #   count +=1
    #For mAP of first class
    mAP_airplanes = mAP(predictions,'airplanes')
    mAP_birds = mAP(predictions,'birds')
    mAP_cars = mAP(predictions,'cars')
    mAP_horses = mAP(predictions,'horses')
    mAP_ships = mAP(predictions,'ships')
    
    #Draw the result image
    map_value = [mAP_airplanes,mAP_birds,mAP_cars,mAP_horses,mAP_ships]
    print('map_value is :',map_value)

        
        
    
    print("Total images is:",count)
    print("predictions is",predictions)
    print("Test images classified.")

    #plotConfusions(true, predictions)
    print("Confusion matrixes plotted.")

    #findAccuracy(true, predictions)
    print("Accuracy calculated.")
    print("Execution done.")
    return map_value

def execute(train_path, test_path, no_clusters, kernel,SIFT_type):
    kmeans, scale, svm, im_features = trainModel(train_path, no_clusters, kernel,SIFT_type)
    map_value= testModel(test_path, kmeans, scale, svm, im_features, no_clusters, kernel,SIFT_type)

    return map_value

if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', action="store", dest="train_path", required=True)
    parser.add_argument('--test_path', action="store", dest="test_path", required=True)
    parser.add_argument('--no_clusters', action="store", dest="no_clusters", default=500)
    parser.add_argument('--kernel_type', action="store", dest="kernel_type", default="linear")

    args =  vars(parser.parse_args())
    if(not(args['kernel_type'] == "linear" or args['kernel_type'] == "precomputed")):
        print("Kernel type must be either linear or precomputed")
        exit(0)
        
    #Comparsin between SIFT_rgb and SIFT_gray
    
    
    SIFT_type=['grayscale SIFT','RGBscale SIFT']
    mapvalue1 = execute(args['train_path'], args['test_path'], int(args['no_clusters']), args['kernel_type'],'grayscale SIFT')
    mapvalue2 = execute(args['train_path'], args['test_path'], int(args['no_clusters']), args['kernel_type'],'RGBscale SIFT')
    map_image(mapvalue1,mapvalue2)
    '''
    mapvalue1 = execute(args['train_path'], args['test_path'], 400, args['kernel_type'],'grayscale SIFT')
    mapvalue2 = execute(args['train_path'], args['test_path'], 1000, args['kernel_type'],'grayscale SIFT')
    mapvalue3 = execute(args['train_path'], args['test_path'], 4000, args['kernel_type'],'grayscale SIFT')
    map_image3(mapvalue1,mapvalue2,mapvalue3)
    '''
    '''
    Comparsin between SURF and SIFT_gray
    mapvalue1 = execute(args['train_path'], args['test_path'], int(args['no_clusters']), args['kernel_type'],'SURF')
    mapvalue2 = execute(args['train_path'], args['test_path'], int(args['no_clusters']), args['kernel_type'],'HOG')
    map_image(mapvalue1,mapvalue2)
    '''