import numpy as np
import matplotlib.pyplot as plt
import cv2




def randomCentroids(k):
    centroids = []
    for i in range(k):
        centroids.append([np.random.randint(0,255), np.random.randint(0,256), np.random.randint(0,255)])
    return centroids

def shouldStop(oldCentroids, centroids, iterations):
    if len(oldCentroids) == 0: return False
    MAX_ITERATIONS = 75
    if iterations > MAX_ITERATIONS: return True
    return (oldCentroids == centroids).all()

def createLabels(dataSet, centroids):
    distanceSet = np.zeros((len(centroids), len(dataSet)))
    for c in range(len(centroids)):
        distanceSet[c] = ((dataSet-centroids[c])**2).sum(axis=1)
        # creates array of distances between points and each centroid

    # find the lowest distance centroid and return idx(centroid number)
    labels = np.argmin(distanceSet, axis=0)
    return labels

def newCentroids(dataSet, labels, k):
    centroids = np.zeros((k, 3))
    for i in range(k):
        # get all points sharing a label
        mask = np.where(labels == i, True, False)
        points = dataSet[mask]
        if len(points) == 0:
            centroids[i] = randomCentroids(1)[0]
            continue


        newCent = points.sum(axis=0) #add columns
        newCent = np.rint(np.divide(newCent, len(points))) # divide for average
        centroids[i] = newCent
    return centroids

def kMeans(dataset, k):
    centroids = randomCentroids(k)

    iterations = 0
    oldCentroids = []

    while not shouldStop(oldCentroids, centroids, iterations):
        #display3DScatter(dataset, centroids)
        oldCentroids = centroids

        iterations +=1

        labels = createLabels(dataset, centroids)

        centroids = newCentroids(dataset, labels, k)

    #display3DScatter(dataset, centroids)
    print("iterations:", iterations)
    print("k:", k)
    return centroids


def recolor(dataset, centroids):
    labels = createLabels(dataset, centroids)

    for i in range(len(centroids)):
        mask = np.where(labels == i, True, False)
        dataset[mask] = centroids[i]

    return dataset

def swapRecolor(dataset1, dataset2, centroids1, centroids2):
    labels1 = createLabels(dataset1, centroids1)
    labels2 = createLabels(dataset2, centroids2)

    priorityCentroids1 = []
    priorityCentroids2= []

    # sort centroids by how many labels they have
    for c in range(len(centroids1)):
        priorityCentroids1.append((np.count_nonzero(labels1 == c), c, centroids1[c]))
        priorityCentroids2.append((np.count_nonzero(labels2 == c), c, centroids2[c]))

    priorityCentroids1.sort(key=lambda x: -x[0])
    priorityCentroids2.sort(key=lambda x: -x[0])



    for i in range(len(centroids1)):
        mask = np.where(labels1 == priorityCentroids1[i][1], True, False)
        dataset1[mask] = priorityCentroids2[i][2]

    for i in range(len(centroids2)):
        mask = np.where(labels2 == priorityCentroids2[i][1], True, False)
        dataset2[mask] = priorityCentroids1[i][2]
    return dataset1, dataset2

def display3DScatter(dataset, centroids):
    fig = plt.figure()
    ax= fig.add_subplot(projection='3d')
    colors = np.multiply(dataset, 1/255)
    cents = np.array(centroids)
    ax.scatter(cents[:,0], cents[:,1], cents[:,2], c="blue", marker='X')
    ax.scatter(dataset[:,0], dataset[:,1], dataset[:,2], c=colors)

    outImage(dataset, "test.png", centroids, (21,19,3))
    plt.show()

def outImage(RGB, name, centroids, original_shape):
    new_RGB = recolor(RGB.copy(), centroids)
    reduced_image = new_RGB.reshape(original_shape)
    output_image = reduced_image[:,:,::-1]
    cv2.imwrite(name, output_image)

# # Test
# dataSet = np.array([[1,2,3], [1,1,1], [0,1,4], [6,7,8]])
# weights = np.array([1,1,1,1])
#
# print(kMeans(dataSet, 2))