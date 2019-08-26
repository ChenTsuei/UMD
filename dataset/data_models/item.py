class Item:
    def __init__(self, texts, text_lengths, true_images, true_prods, false_images, false_prods):
        self.texts = texts
        self.text_lengths = text_lengths
        self.true_images = true_images
        self.true_prods = true_prods
        self.false_images = false_images
        self.false_prods = false_prods

    def __str__(self):
        return str(self.__dict__)
