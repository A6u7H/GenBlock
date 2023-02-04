import os
import cv2


class Preprocessor:
    def __init__(
        self,
        transformer
    ) -> None:
        self.transformer = transformer

    def preprocess_data(
        self,
        src_data_path: str,
        tgt_data_path: str
    ) -> None:
        for image_name in os.listdir(src_data_path):
            image_path_src = os.path.join(src_data_path, image_name)
            image_path_tgt = os.path.join(tgt_data_path, image_name)
            image = cv2.imread(
                image_path_src,
                cv2.IMREAD_UNCHANGED
            )[1:-1, 1:-1, [0, 1, 2]]
            image = self.transformer(image)
            cv2.imwrite(image_path_tgt, image)
