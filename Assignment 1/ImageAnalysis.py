import os

from CognitiveServices import ComputerVision, Emotion


class ImageAnalysis():
    def run_computer_vision(self):
        computer_vision_key = '288662f5a12f4e2cb81270bcc9ecf35f'
        captions_list = []
        image_label = []

        computer_vision = ComputerVision(computer_vision_key)

        listing = os.listdir(r'./dataset/photos/')
        for infile in listing:
            print(r'./dataset/photos/' + infile)
            with open(r'./dataset/photos/' + infile, 'rb') as image_file:
                data = image_file.read()

            result = computer_vision.analyze_image(data, 'Tags, Description')

            image_label.append(r'./dataset/photos/' + infile)
            captions_list.append(result['description']['captions'][0]['text'])

        return captions_list, image_label

    def run_emotion(self):
        """ Emotion API"""
        emotion_key = 'c8c290727dcd4e8ca75dc2c92fa4fa94'
        emotion = Emotion(emotion_key)
        emotion_list = []
        emotion_label = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

        listing = os.listdir(r'./dataset/photos/')
        for infile in listing:
            print(r'./dataset/photos/' + infile)
            with open(r'./dataset/photos/' + infile, 'rb') as image_file:
                data = image_file.read()

            result = emotion.recognize_image(data)
            print(result)

            for face in result:
                emotion_list.append(
                    [face['scores']['anger'], face['scores']['contempt'], face['scores']['disgust'],
                     face['scores']['fear'],
                     face['scores']['happiness'], face['scores']['neutral'], face['scores']['sadness'],
                     face['scores']['surprise']])
                # print(face['scores'])
                print(emotion_list)

        return emotion_list, emotion_label

    if __name__ == '__main__':
        run_computer_vision()
        run_emotion()
