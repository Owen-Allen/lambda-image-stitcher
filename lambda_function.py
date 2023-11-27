import json
import cv2 as cv
import numpy as np
import boto3
import uuid

 # for memory checking
import sys
import gc

s3 = boto3.client('s3')

def padder(image):
    # input: numpy array image
    # returns: a numpy array image (canvas), where the image is projected onto a black canvas
    # the image ontop of the canvas is centered vertically, and shifted to the left 1/4

    h, w = image.shape # for colour, needs 3rd arg on left

    canvas = np.zeros((round(h * 1.5), round(w * 1.5)), dtype=np.uint8) # add ,3 to np.zeros args for colour

    x_offset = 0 #round(w/8)
    y_offset = round(h / 3)

    canvas[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image
    return canvas


def stitch(img1, img2):
    # img2 (left image) will act as our 'destination' or 'canvas' that img1 will be warped to fit
    img2 = padder(img2)
    
    # find the keypoints and descriptors of both images with SIFT
    detector = cv.AKAZE_create()
    kp1, des1 = detector.detectAndCompute(img1, None)
    # use a mask to only look at the left half of the right image, since it is unlikely that the intersecting features are on the right side of the right image
    mask = np.zeros_like(img2)
    height, width = img2.shape
    roi = (0, 0, width // 2, height)
    cv.rectangle(mask, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (255), -1)
    kp2, des2 = detector.detectAndCompute(img2, mask)

    # brute force check for matches between the images, then sort
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    src_pts = np.float32([ kp1[matches[m].queryIdx].pt for m in range(0, 20) ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[matches[m].trainIdx].pt for m in range(0, 20) ]).reshape(-1,1,2)

    h, status = cv.findHomography(src_pts, dst_pts)
    
    # warp the right image to fit with the left
    img = cv.warpPerspective(img1, h, (img2.shape[1], img2.shape[0])) # mapping to DESTINATION domain

    # finally, stitch them together
    img = cv.bitwise_or(img2, img) #https://docs.opencv.org/3.4/d2/de8/group__core__array.html#gab85523db362a4e26ff0c703793a719b4
    return img


def lambda_handler(event, context):

    # grab filenames from request parameters
    left_filename = event['queryStringParameters']['left']
    right_filename = event['queryStringParameters']['right']
    
    # read left
    left_file_object = s3.get_object(Bucket='opencv-data', Key=left_filename)
    left_file_content = left_file_object["Body"].read()
    left_np_array = np.fromstring(left_file_content, np.uint8)
    left_image = cv.imdecode(left_np_array, cv.IMREAD_COLOR)
    
    # read right
    right_file_object = s3.get_object(Bucket='opencv-data', Key=right_filename)
    right_file_content = right_file_object["Body"].read()
    right_np_array = np.fromstring(right_file_content, np.uint8)
    right_image = cv.imdecode(right_np_array, cv.IMREAD_COLOR)
    
    # converting images to gray saves a ton of memory. still very close to running out
    left_image_gray = cv.cvtColor(left_image, cv.COLOR_BGR2GRAY)
    right_image_gray = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)
    
    # cleanup
    # from testing this saves about ~100mb of memory
    del right_file_object
    del right_file_content
    del right_np_array
    del right_image
    del left_file_object
    del left_file_content
    del left_np_array
    del left_image
    gc.collect()

    # process images
    result = stitch(right_image_gray, left_image_gray)
    result_filename = str(uuid.uuid4()) + '.png'
    
    # save to s3
    cv.imwrite(f'/tmp/{result_filename}', result)
    s3.put_object(Bucket='opencv-data', Key=result_filename, Body=open(f'/tmp/{result_filename}', 'rb').read())
    
    # return name of file to pull from s3
    res_data = {'result_filename': result_filename}
    
    return {
        'statusCode': 200,
        'body': json.dumps(res_data)
    }
