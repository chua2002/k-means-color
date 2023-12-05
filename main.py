import numpy as np
import cv2
import matplotlib.pyplot as plt
import kmeans


def colorDistance(original, reduced):
    test = np.sum(np.linalg.norm(reduced-original, axis=1))
    return test

def avrgNearestCentroidDist(centroids):
    centroids = centroids.tolist()
    sum = 0
    for c in centroids:
        temp_centroids = centroids.copy()
        temp_centroids.remove(c)
        temp_centroids = np.array(temp_centroids)
        deltas = temp_centroids - c
        dist = np.linalg.norm(deltas, axis=1)
        sum += min(dist)

    return sum / len(centroids)


def testImage(RGB, k):
    # old_RGB = RGB.copy()
    centroids = kmeans.kMeans(RGB.copy(),k)
    # new_RGB = kmeans.recolor(RGB.copy(), centroids)
    return (avrgNearestCentroidDist(centroids), centroids)


def bestImage(RGB, k, n):
    values = []
    for t in range(n):
        values.append(testImage(RGB, k))

    return max(values, key = lambda x: x[0])



def multiTest(input_image, r, n):
    # get alpha channel
    input_image = input_image[:, :, :3]

    # convert BGR -> RGB
    input_image = input_image[:, :, ::-1]
    # print('input_image', input_image.shape, input_image.dtype, input_image.min(), input_image.max())

    RGB = np.array([input_image[:, :, 0].ravel(), input_image[:, :, 1].ravel(), input_image[:, :, 2].ravel()]).T
    # RGB, counts = np.unique(RGB, return_counts=True, axis=0)

    values = []
    kRange = []
    for k in r:
        kRange.append(k)
        values.append(bestImage(RGB, k, n))

    for im in values:
        filename = "output_k=" + str(len(im[1])) + ".png"
        print(filename)
        outImage(RGB, filename, im[1], input_image.shape)

    showTrend(kRange, [element[0] for element in values])
    plt.scatter(kRange, [element[0] for element in values])
    plt.show()



def outImage(RGB, name, centroids, original_shape):
    new_RGB = kmeans.recolor(RGB.copy(), centroids)
    reduced_image = new_RGB.reshape(original_shape)
    output_image = reduced_image[:,:,::-1]
    cv2.imwrite(name, output_image)

def showTrend(x, y):
    coef = np.polyfit(x, y, 3)
    plt.plot(x, np.polyval(coef, x))

def colorSwap(input1, input2, k, n):
    input1 = input1[:, :, :3]
    input1 = input1[:, :, ::-1]
    RGB1 = np.array([input1[:, :, 0].ravel(), input1[:, :, 1].ravel(), input1[:, :, 2].ravel()]).T

    input2 = input2[:, :, :3]
    input2 = input2[:, :, ::-1]
    RGB2 = np.array([input2[:, :, 0].ravel(), input2[:, :, 1].ravel(), input2[:, :, 2].ravel()]).T

    centroids1 = bestImage(RGB1, k, n)[1]
    centroids2 = bestImage(RGB2, k, n)[1]


    new_RGB1 , new_RGB2 = kmeans.swapRecolor(RGB1, RGB2, centroids1, centroids2)
    reduced_image1 = new_RGB1.reshape(input1.shape)
    output_image1 = reduced_image1[:, :, ::-1]
    cv2.imwrite("output_swap1.png", output_image1)

    reduced_image2 = new_RGB2.reshape(input2.shape)
    output_image2 = reduced_image2[:, :, ::-1]
    cv2.imwrite("output_swap2.png", output_image2)

def singleReduce(input_image, k, n):
    # get alpha channel
    input_image = input_image[:, :, :3]

    # convert BGR -> RGB
    input_image = input_image[:, :, ::-1]
    # print('input_image', input_image.shape, input_image.dtype, input_image.min(), input_image.max())

    RGB = np.array([input_image[:, :, 0].ravel(), input_image[:, :, 1].ravel(), input_image[:, :, 2].ravel()]).T
    # RGB, counts = np.unique(RGB, return_counts=True, axis=0)

    centroids = bestImage(RGB, k, n)[1]

    outImage(RGB, "output_single.png", centroids, input_image.shape)


def getFileName():
    while (1):
        fileName = input("Enter file name:\t")
        try:
            f = open(fileName)
        except:
            print("Invalid file name. Try again")
        else:
            f.close()
            input_image = cv2.imread(fileName)
            return input_image


while 1:
    print("select an option:\n"
          "1. reduce image colors once\n"
          "2. reduce image colors multiple\n"
          "3. swap image colors\n"
          "4. exit")
    userin = int(input())

    match userin:
        case 1:
            input_image = getFileName()
            while 1:
                k = int(input("Enter number of colors(>1):\t"))
                if k > 1: break

            while 1:
                n = int(input("Enter number of samples(>0)\t"))
                if n > 0: break

            print("Running... ")
            singleReduce(input_image, k, n)
            print("Finished!")
            print("Result saved as output_single.png")
            exit(0)

        case 2:
            input_image = getFileName()
            while 1:
                k = int(input("Enter starting number of colors(>1):\t"))
                if k > 1: break

            while 1:
                a = int(input("Enter number of additional colors(>0):\t"))
                if a > 0: break

            while 1:
                n = int(input("Enter number of samples(>0)\t"))
                if n > 0: break

            print("Running... ")
            multiTest(input_image, range(k, k+a+1), n)
            print("Finished!")
            print("Output saved as output_k=*.png")
            exit(0)

        case 3:
            input_image1 = getFileName()
            input_image2 = getFileName()
            while 1:
                k = int(input("Enter number of colors(>1):\t"))
                if k > 1: break

            while 1:
                n = int(input("Enter number of samples(>0)\t"))
                if n > 0: break

            print("Running... ")
            colorSwap(input_image1, input_image2, k, n)
            print("Finished!")
            print("Output saved as output_swap1.png and output_swap2.png")
            exit(0)

        case 4:
            print("Goobye")
            exit(0)

        case _:
            print("Invalid Input. Try again\n")
