class ProductCollection:
    def __init__(self):
        self.products = []
        self.url2prod = {}

    def insert(self, prod):
        # insert a new product
        self.products.append(prod)

        # set the map relation
        if prod.image_filename:
            self.url2prod[prod.image_filename] = prod
        if prod.image_filename_all:
            for images in prod.image_filename_all.values():
                for image in images:
                    self.url2prod[image] = prod

    def prod_from_url(self, img):
        # get product object according to the url
        return self.url2prod.get(img, None)

    def empty(self):
        # test if the collection is empty
        return not self.products
