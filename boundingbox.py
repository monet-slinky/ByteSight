# -*- coding: utf-8 -*-
"""boundingBox.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1d2TizaCAgv8b-Wy4wR0sulrnGGJWEEkM
"""

from thresholdingfunction import otsuthresholding
#from classify import load_model, load_image_fromnumpy, predict_single
from torchvision import datasets, models, transforms
import torch
import os
import cv2
from blackandwhiteratios import blackandwhiteratio

def removeBorder(image):
  '''
  Takes in an image, makes it grayscale, and then removes black "widescreen" style border from the image
  Note we momentarily remove the bottom of the image to get rid of the words
  '''
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # Mask of non-black pixels (assuming image has a single channel).
  mask = gray_image[0:680, 0:1280] > 1

  # Coordinates of non-black pixels.
  coords = np.argwhere(mask)

  # Bounding box of non-black pixels.
  x0, y0 = coords.min(axis=0)
  x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top

  # Get the contents of the bounding box.
  cropped = gray_image[x0:x1, y0:y1]
  return cropped

class MyImage:
    def __init__(self, file):
        self.image = cv2.imread(file)
        self.__name = file

    def __str__(self):
        return self.__name

def cropImage(image_file, modality, labelsfile, steps,scale):
    '''
    image_file is the image to crop
    modality can be m, p, d, or pm (miscope, phone, DSLR, phone miscope)
    '''
    
    if modality=='pm':
      # just crop out the black
      objectimage = cv2.imread(image_file)
      objectimage=removeBorder(objectimage)
      return objectimage
    # for convenience atm, not putting an else statement but this is effectively "else"
    objectdictionary = {}
    
    blurredimage = cv2.medianBlur(image_file ,5)
    objectimage2 = otsuthresholding(blurredimage)
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(objectimage2, kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
    objectimage = img_dilation
    dimensions = np.shape(objectimage)
    if image_file[-4] == 'm':
        steps = 1
        skip = True
        windowSize = int(2*scale * min(dimensions))
        looping = [dimensions[0],dimensions[1]]
    else:
        skip = False
        windowSize = int(scale * min(dimensions))
        looping = [int(dimensions[0]/steps), int(dimensions[1]/steps)]
    number = 0
    pixelvalues = np.unique(objectimage)
    #print(pixelvalues)
    maximumpixelvalue = max(pixelvalues)
    bwimage = np.zeros(dimensions,np.uint8) + objectimage
    ################################################################################
    # Now we have defined our windowSize and looping. Our steps hyperparameter is an
    # integer n. We use this to only look at a pixel every n steps, i.e. if it was 30,
    # we would look at pixel 1, then pixel 31. This speeds up the computation greatly,
    # and doesn't affect the accuracy that much, at least qualitatively.
    # Here it gets confusing, at least to me. So imagine we get to our first 0 value pixel.
    # We define a 'searchzone' around this pixel, the size of which is determined by the
    # windowSize hyperparameter. We obtain a list of all pixel values within this searchzone,
    # and since this is our first 0 value pixel, we would expect to obtain a list containing
    # only 255 and 0, again because we binarized the image previously. Next we obtain all coordinates
    # within the search-zone that have a pixel value of 0, including our OG center pixel by looping
    # through the search-zone(there is definitely a better way to do this, like maybe
    # [i for i, x in enumerate(searchzoneproper)]
    # but honestly, this might not work and I'm not sure whether this will be faster at all.
    # Anywho, now we have an image with mostly 255 and 0 values, and some 1 values. We continue to loop
    # through the image and find another 0 value pixel. We generate our searchzone, obtain a list
    # of all unique values, and remove our trivial values 0 and 255. There are three main cases:
    # 1. Our list contains no other values. This suggests that an entirely new object has been discovered
    # In our case, we would then define all 0 values in this searchzone as '2' or whatever object we're up to.
    # Side note: We are limited to 255 objects, but who wants to deal with that large/cluttered
    # of an image that needs more than 255 objects defined?
    # 2. Our list contains one other value. Up to this point, the only possibility is '1'.
    # The pixel we are centered on, and all surrounding 0 pixels are then considered to be part of object '1'.
    # 3. Our list contains more than one other value. Not possible up till now, but imagine further in the loop
    # when we have let's say 5 objects defined, and maybe '1' and '4' pixel values are found. Then perhaps, 1 and 4
    # are not actual separate objects, but one large object. We compress these objects into one single object(in a messy
    # way).
    # Side note: This encourages zoomed-out crops, since a disembodied leg or
    # artifact that is close enough to the mosquito will cause it to be recognized
    # as a part of the object. To fix this, it might be worth it to rerun the
    # function on the resulting cropped image. The caveat here will be that the
    # bounding box coordinates are no longer correct, and thus will need to be
    # corrected by the coordinates obtained by the first run.
    ###############################################################################
    for j in range(looping[0]):
        for i in range(looping[1]):
            coordinates = [j*steps,i*steps]
            # changed != maximumpixelvalue to == 0, to see if this is bit faster
            # faulty logic is the reason,
        
            if objectimage[coordinates[0], coordinates[1]] == 0:
                y = coordinates[0] - windowSize
                x = coordinates[1] - windowSize
                if y < 0 and x < 0:
                    x = 0
                    y = 0
                elif y < 0:
                    y = 0
                elif x < 0:
                    x = 0
                # Define an area of search zone
                searchzoneproper = objectimage[y:j*steps + windowSize, x:i*steps + windowSize]
                objectsnearpixel = list(np.unique(searchzoneproper))
                try:
                    objectsnearpixel.remove(255)
                except ValueError:
                    pass
                try:
                    objectsnearpixel.remove(0)
                except ValueError:
                    pass
                if len(objectsnearpixel) == 0:
                    number += 1
                    #objects.append(number)
                    objectimage[coordinates[0], coordinates[1]] = number
                    objectdictionary[number] = [(coordinates[0], coordinates[1])]
                    newdimensions = np.shape(searchzoneproper)
                    for l in range(newdimensions[0]):
                        for m in range(newdimensions[1]):
                            if searchzoneproper[l, m] == 0:
                                calculatedcoordinates = (y+l,x+m)
                                objectimage[calculatedcoordinates[0],calculatedcoordinates[1]] = number
                                objectdictionary[number].append(calculatedcoordinates)
                    continue
                elif len(objectsnearpixel) == 1:
                    newnumber = objectsnearpixel[0]
                    objectimage[coordinates[0],coordinates[1]] = newnumber
                    objectdictionary[newnumber].append((coordinates[0],coordinates[1]))
                    continue
                elif len(objectsnearpixel) > 1:
                    newnumber = min(objectsnearpixel)
                    #print(newnumber)
                    objectimage[coordinates[0],coordinates[1]] = newnumber
                    objectdictionary[newnumber].append((coordinates[0],coordinates[1]))
                    #print(objectsnearpixel)
                    for k in objectsnearpixel:
                        #print(k)
                        if k == newnumber:
                            continue
                        else:
                            secondaryobject = objectdictionary[k]
                            print(secondaryobject)
                            #secondaryobject = list(secondaryobject)
                            for z in range(len(secondaryobject)):
                                #print(z)
                                #print(secondaryobject[z])
                                objectdictionary[newnumber].append(secondaryobject[z])
                                changedcoordinates = secondaryobject[z]
                                objectimage[changedcoordinates[0],changedcoordinates[1]] = newnumber
                            # fix this in future
                            del objectdictionary[k]
                            continue
            else:
                continue
    #print(str(len(objectdictionary)) + ' objects in image: Checking for mosquitos now')
    ###############################################################################
    # Hooray, we now have a dictionary of all objects found in image, and the pixels
    # that comprise each object. We load in our transfer learning model to have it
    # predict the nature of each object detected. We determine a thresholdarea(magic
    # numbers again) to filter our objects, and remove those that are most likely
    # noise. Finally, we generate a list of coordinates to crop our image based on
    # the pixels comprising the object. We use the network to predict whether
    # the cropped image contains a mosquito. There is also a block of commented
    # code, in case you want to see the objects highlighted on the original image
    # in real time. It might not work, but it did at some point, in a different
    # environment.
   
    colorimage = image_file
    coordinates = {}
    coordinates['miny'] = []
    coordinates['maxy'] = []
    coordinates['minx'] = []
    coordinates['maxx'] = []
    scores = []

    if modality== 'm':
        thresholdarea = 50
    elif modality== 'p':
        thresholdarea = 100    
    else:
        thresholdarea = 20
    for k in objectdictionary:
        objectcoordinates = np.array(objectdictionary[k])
        if len(objectcoordinates) < thresholdarea:
            continue
        minx = dimensions[1]
        maxx = 0
        maxy = 0
        miny = dimensions[0]
        for (yy, xx) in objectdictionary[k]:
            if xx <= minx:
                minx = xx
            if xx >= maxx:
                maxx = xx
            if yy <= miny:
                miny = yy
            if yy >= maxy:
                maxy = yy
        if minx == maxx:
            continue
        if maxy == miny:
            continue
        croppedimage = colorimage[miny:maxy+1,minx:maxx+1]
   
        bands = 10
        miny = miny - bands
        maxy = maxy + bands
        minx = minx - bands
        maxx = maxx + bands
        
        coordinates['miny'].append(miny)
        coordinates['maxy'].append(maxy)
        coordinates['minx'].append(minx)
        coordinates['maxx'].append(maxx)

        if len(coordinates['miny']) == 0:
          print('this method failed')
          newcoordinates = False
          status = 'False'
        elif len(coordinates['miny']) == 1:
            print('mosquito found')
            newcoordinates = {}
            newcoordinates['miny'] = min(coordinates['miny'])
            newcoordinates['maxy'] = max(coordinates['maxy'])
            newcoordinates['minx'] = min(coordinates['minx'])
            newcoordinates['maxx'] = max(coordinates['maxx'])
            status = 'Good'
        else:
            print('too many candidates')
            bestobject = np.argmax(scores)

            newcoordinates = {}
            newcoordinates['miny'] = coordinates['miny'][bestobject]
            newcoordinates['maxy'] = coordinates['maxy'][bestobject]
            newcoordinates['minx'] = coordinates['minx'][bestobject]
            newcoordinates['maxx'] = coordinates['maxx'][bestobject]
            status = 'Fixed'
        if newcoordinates == False:
            croppedimage = colorimage
        else:
            width = newcoordinates['maxx']-newcoordinates['minx']
            height = newcoordinates['maxy']-newcoordinates['miny']
            someroom = int(max(height,width)*.1)
            newcoordinates['miny'] -= someroom
            newcoordinates['maxy'] += someroom
            newcoordinates['minx'] -= someroom
            newcoordinates['maxx'] += someroom
            for key , value in newcoordinates.items():
                if value <= 0:
                    newcoordinates[key] = 0
                else:
                    continue
            croppedimage = colorimage[newcoordinates['miny']:newcoordinates['maxy'],newcoordinates['minx']:newcoordinates['maxx']]

        return newcoordinates, croppedimage, status


def main():
    #test_file="/content/drive/MyDrive/Colab Notebooks/VT/vectech-classification/data/cropped/cropped/aedes/aegypti/JHU-001849_04m.jpg"
    #test_file = "/content/201207_160327.jpg"
    #labelsfile = 'labels.txt'
    #cropImage(test_file, 'm', labelsfile,5, 0.08)
if __name__ == '__main__':
    main()
