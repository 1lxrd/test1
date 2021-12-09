import cv2
import numpy


class Image:
    """
    Base class for image to be processed.
    Should be inherited and extended.
    """

    def __init__(self, path: str):
        self.origin = path
        self.cv_image = cv2.imread(path)

    def save(self, path: str) -> None:
        cv2.imwrite(path, self.cv_image)


class TemplateNotFound(Exception):

    def __init__(self, deviation=0, message="Template hasn't been found in given image."):
        self.deviation = deviation
        self.message = ''.join([message, ' Deviation: ', str(deviation)])
        super().__init__(self.message)


class TemplateFinder:
    """
    This class holds template image and mask for more reliable processing
    """

    def __init__(self, path: str):
        self.template = Image(path)
        self.mask = cv2.imread(self.template.origin, cv2.IMREAD_UNCHANGED)
        self.mask[self.mask[:, :, 3] == 0] = [0, 0, 0, 0]
        self.mask[self.mask[:, :, 3] == 1] = [255, 255, 255, 0]
        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)

    def find(self, image: Image, threshold: int = 0.2):
        """
        This function takes image to be processed and deviation threshold.
        Returns tuple:
        (left_upper_point, right_lower_point, deviation)
        """
        matches = cv2.matchTemplate(image.cv_image,
                                    self.template.cv_image,
                                    method=cv2.TM_SQDIFF_NORMED,
                                    mask=self.mask)
        trows, tcols = self.template.cv_image.shape[:2]
        deviation, _, mn_loc, _ = cv2.minMaxLoc(matches)

        if deviation > threshold:
            raise TemplateNotFound(deviation=deviation)

        px, py = mn_loc

        return (px, py), (px + tcols, py + trows), float(deviation)


def show(text, img):
    cv2.imshow(text, img)
    cv2.waitKey(2)


class NewImage(Image):
    def __init__(self, path: str):
        super().__init__(path)

    def grayscale(self, path):
        self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        show('gray', self.cv_image)
        self.save(path)

    def outline(self, path):
        self.grayscale(path)
        grayscale = self.cv_image

        # grayscale = cv2.GaussianBlur(self.cv_image, (5, 5), 0)
        # grayscale = cv2.Canny(self.cv_image, 100, 140)

        ret, thresh = cv2.threshold(grayscale, 127, 255, 0)

        con, hir = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(self.cv_image, con, -1, (0, 255, 0), 2)

        show('outline', self.cv_image)
        self.save(path)
