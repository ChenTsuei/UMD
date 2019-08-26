class Utterance():
    def __init__(self, speaker, text, images, false_images):
        self.speaker = speaker
        self.text = text
        self.images = images
        self.false_images = false_images

    def __str__(self):
        # for debug
        return str((self.speaker, self.text, self.images, self.false_images))
