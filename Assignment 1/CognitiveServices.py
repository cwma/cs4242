import time

import requests

class ComputerVision():
    """Computer Vision API"""

    def __init__(self, key):
        """Initialize the ComputerVision
        Keyword arguments:
        key -- the API key for Computer Vision API
        """

        self._url = 'https://westus.api.cognitive.microsoft.com/vision/v1.0/analyze'
        self._key = key

    def analyze_image(self, image_data, features, language='en', max_retries=3):
        """Analyze the impage
        Keyword arguments:
        image_data -- the binary data of the image
        features -- the features what to analyze, sepatated by ',',
                    e.g. 'Categories,Tags,Description,Faces,ImageType,Color,Adult'
        language -- the language for response, only support 'en' and 'zh' (default 'en')
        max_tetries -- the number of retries when the request failed (default 3)
        """

        params = {'visualFeatures': features, 'language': language}

        headers = dict()
        headers['Ocp-Apim-Subscription-Key'] = self._key
        headers['Content-Type'] = 'application/octet-stream'

        retries = 0
        result = None

        while True:
            response = requests.request('post', self._url, data=image_data, headers=headers, params=params)

            if response.status_code == 429:
                print("Message: %s" % (response.json()['error']['message']))

                if retries <= max_retries:
                    time.sleep(1)
                    retries += 1
                    continue
                else:
                    print('Error: failed after retrying!')
                    break
            elif response.status_code == 200 or response.status_code == 201:
                if 'content-length' in response.headers \
                        and int(response.headers['content-length']) == 0:
                    result = None
                elif 'content-type' in response.headers \
                        and isinstance(response.headers['content-type'], str):
                    if 'application/json' in response.headers['content-type'].lower():
                        result = response.json() if response.content else None
                    elif 'image' in response.headers['content-type'].lower():
                        result = response.content
            else:
                print("Error code: %d" % (response.status_code))
                print("Message: %s" % (response.json()))

            break

        return result

class Face():
    """Face API"""

    def __init__(self, key):
        """Initialize Face
        Keyword arguments:
        key -- the API key for Face API
        """

        self._url = 'https://westus.api.cognitive.microsoft.com/face/v1.0/detect'
        self._key = key

    def detect_image(self, image_data, face_attributes):
        """Analyze the impage
        Keyword arguments:
        image_data -- the binary data of the image
        face_attributes -- the face attributes what to be returned, sepatated by ',',
                    e.g. 'age,gender,headPose,smile,facialHair,glasses'
        returnFaceAttributes
        """

        params = {'returnFaceAttributes' : face_attributes}

        headers = dict()
        headers['Ocp-Apim-Subscription-Key'] = self._key
        headers['Content-Type'] = 'application/octet-stream'
        response = requests.request('post', self._url, \
            data=image_data, headers=headers, params=params)

        return response.json()


class Emotion():
    """Emotion API"""

    def __init__(self, key):
        """Initialize Emotion
        Keyword arguments:
        key -- the API key for Emotion API
        """

        self._url = 'https://westus.api.cognitive.microsoft.com/emotion/v1.0/recognize'
        self._key = key

    def recognize_image(self, image_data):
        """Analyze the impage
        Keyword arguments:
        image_data -- the binary data of the image
        """

        headers = dict()
        headers['Ocp-Apim-Subscription-Key'] = self._key
        headers['Content-Type'] = 'application/octet-stream'
        response = requests.request('post', self._url, data=image_data, headers=headers)

        return response.json()