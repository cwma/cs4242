import os
import json
import time

from CognitiveServices import ComputerVision, Emotion


class ImageAnalysis():

    def save_file(self, result, file_name):
        _file = open(file_name, "w")
        _file.write(json.dumps(result))

    def run_computer_vision(self):
        computer_vision_key = '288662f5a12f4e2cb81270bcc9ecf35f'
        captions_list = []
        image_label = []

        computer_vision = ComputerVision(computer_vision_key)

        listing = os.listdir(r'./dataset/photos/')
        count = len(listing)
        for i, infile in enumerate(listing):
            if not os.path.exists('dataset/vision/{0}.json'.format(infile)):
                print(r'./dataset/photos/' + infile + " {0} of {1}".format(i, count))
                with open(r'./dataset/photos/' + infile, 'rb') as image_file:
                    data = image_file.read()

                result = computer_vision.analyze_image(data, 'Tags, Description')

                #image_label.append(r'./dataset/photos/' + infile)
                #captions_list.append(result['description']['captions'][0]['text'])
                try:
                    _result = { r'photos/' + infile: { "description": result['description']['captions'][0]['text'], "tags": result['description']['tags']}}
                except TypeError as e:
                    _result = { r'photos/' + infile: { "description": "", "tags": []}}
                except KeyError as e:
                    _result = { r'photos/' + infile: { "description": "", "tags": []}}

                self.save_file(_result, 'dataset/vision/{0}.json'.format(infile))

        return captions_list, image_label

    def run_emotion(self):
        """ Emotion API"""
        emotion_key = 'c8c290727dcd4e8ca75dc2c92fa4fa94'
        emotion = Emotion(emotion_key)
        emotion_list = []
        emotion_label = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

        listing = os.listdir(r'./dataset/photos/')
        count = len(listing)
        for i, infile in enumerate(listing):
            if not os.path.exists('dataset/emotion/{0}.json'.format(infile)):
                print(r'./dataset/photos/' + infile + " {0} of {1}".format(i, count))
                with open(r'./dataset/photos/' + infile, 'rb') as image_file:
                    data = image_file.read()

                while 1:
                    result = emotion.recognize_image(data)
                    print(result)
                    if 'error' in result and 'RateLimitExceeded' == result['error']['code']:
                        time.sleep(3)
                    else:
                        break


                for face in result:
                    if 'error' not in face:
                        emotion_list.append(
                            [face['scores']['anger'], face['scores']['contempt'], face['scores']['disgust'],
                             face['scores']['fear'],
                             face['scores']['happiness'], face['scores']['neutral'], face['scores']['sadness'],
                             face['scores']['surprise']])
                        # print(face['scores'])
                        # print(emotion_list)

                try:
                    _result = { r'photos/' + infile: [{key: value for key, value in zip(emotion_label, emotion_sublist)} for emotion_sublist in emotion_list]}
                except TypeError as e:
                    _result = { r'photos/' + infile: []}
                except KeyError as e:
                    _result = { r'photos/' + infile: []}
                self.save_file(_result, 'dataset/emotion/{0}.json'.format(infile))

        return emotion_list, emotion_label

if __name__ == '__main__':

    ia = ImageAnalysis()
    ia.run_computer_vision()
    ia.run_emotion()
