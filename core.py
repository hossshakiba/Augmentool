import collections
import os
import io
import base64

import numpy as np
import albumentations as A
import cv2
from PIL import Image


class AugmentCore:
    def __init__(self):
        self._image_ext = None
        self.augmented_image = None
        self.augmented_polygons = []
        self.augmented_bboxes = []

    def __set_image_ext(self, image_path):
        self._image_ext = os.path.splitext(image_path)[1][1:]
        if self._image_ext == 'jpg':
            self._image_ext = 'jpeg'

    def __transform(self, image, augmentation_settings, polygons=None, bboxes=None):
        transforms_list = []
        if polygons is None:
            polygons = []
        if bboxes is None:
            bboxes = []
        for bbox in bboxes:
            if bbox[-1] != "fake_label":
                bbox.append("fake_label")

        polygon_descriptor = {}
        keypoints = []

        for index, item in enumerate(polygons):
            polygon_descriptor[index] = len(item)
            keypoints += item
        it = iter(keypoints)
        keypoints = [*zip(it, it)]

        # Pixel-level transforms
        if 'hue' in augmentation_settings:
            if augmentation_settings['hue'] is not None:
                transforms_list.append(
                    A.HueSaturationValue(
                        sat_shift_limit=augmentation_settings['hue'].get('sat_shift_limit', 0),
                        hue_shift_limit=augmentation_settings['hue'].get('hue_shift_limit', 0),
                        val_shift_limit=augmentation_settings['hue'].get('val_shift_limit', 0),
                        p=1
                    )
                )
        if 'blur' in augmentation_settings:
            if augmentation_settings['blur'] is not None:
                transforms_list.append(
                    A.Blur(
                        blur_limit=augmentation_settings['blur'].get('blur_limit', 1),
                        p=1 if augmentation_settings['blur']['blur_limit'] else 0
                    )
                )
        if 'noise' in augmentation_settings:
            if augmentation_settings['noise'] is not None:
                transforms_list.append(
                    A.GaussNoise(
                        var_limit=augmentation_settings['noise'].get('var_limit', 0),
                        p=1
                    )
                )
        if 'contrast' in augmentation_settings:
            if augmentation_settings['contrast'] is not None:
                limit = augmentation_settings['contrast'].get('limit', 0)
                transforms_list.append(
                    A.RandomContrast(
                        limit=(limit, limit),
                        p=1
                    )
                )
        if 'rain' in augmentation_settings:
            if augmentation_settings['rain'] is not None:
                transforms_list.append(
                    A.RandomRain(
                        drop_length=int(augmentation_settings['rain'].get('drop_length', 20)),  # 0 - 100
                        drop_width=int(augmentation_settings['rain'].get('drop_width', 1)),  # 1 - 5
                        blur_value=int(augmentation_settings['rain'].get('blur_value', 1)),
                        rain_type=augmentation_settings['rain'].get('rain_type', None),
                        # [None, "drizzle", "heavy", "torrential"]
                        p=1
                    )
                )
        if 'snow' in augmentation_settings:
            if augmentation_settings['snow'] is not None:
                transforms_list.append(
                    A.RandomSnow(
                        snow_point_lower=augmentation_settings['snow'].get('snow_point_lower', 0.1),  # 0 - 1
                        snow_point_upper=augmentation_settings['snow'].get('snow_point_upper', 0.3),  # 0 - 1
                        brightness_coeff=augmentation_settings['snow'].get('brightness_coeff', 2.0),  # >= 0
                        p=1
                    )
                )
        if 'sun' in augmentation_settings:
            if isinstance(augmentation_settings['sun'], collections.OrderedDict):
                if augmentation_settings['sun'].get('sun', False) is True:
                    transforms_list.append(
                        A.RandomSunFlare(
                            src_radius=100,
                            p=1
                        )
                    )
            elif isinstance(augmentation_settings['sun'], bool):
                if augmentation_settings['sun'] is True:
                    transforms_list.append(
                        A.RandomSunFlare(
                            src_radius=100,
                            p=1
                        )
                    )
        if 'fog' in augmentation_settings:
            if augmentation_settings['fog'] is not None:
                transforms_list.append(
                    A.RandomFog(
                        fog_coef_lower=augmentation_settings['fog'].get('fog_coef_lower', 0.25),  # 0 - 1
                        fog_coef_upper=augmentation_settings['fog'].get('fog_coef_upper', 0.3),  # 0 - 1
                        alpha_coef=augmentation_settings['fog'].get('alpha_coef', 0.4),  # 0 - 1
                        p=1
                    )
                )
        # Spatial-level transforms
        if 'vertical_flip' in augmentation_settings:
            if isinstance(augmentation_settings['vertical_flip'], collections.OrderedDict):
                if augmentation_settings['vertical_flip'].get('vertical_flip', False) is True:
                    transforms_list.append(
                        A.VerticalFlip(
                            p=1
                        )
                    )
            elif isinstance(augmentation_settings['vertical_flip'], bool):
                if augmentation_settings['vertical_flip'] is True:
                    transforms_list.append(
                        A.VerticalFlip(
                            p=1
                        )
                    )
        if 'horizontal_flip' in augmentation_settings:
            if isinstance(augmentation_settings['horizontal_flip'], collections.OrderedDict):
                if augmentation_settings['horizontal_flip'].get('horizontal_flip', False) is True:
                    transforms_list.append(
                        A.HorizontalFlip(
                            p=1
                        )
                    )
            elif isinstance(augmentation_settings['horizontal_flip'], bool):
                if augmentation_settings['horizontal_flip'] is True:
                    transforms_list.append(
                        A.HorizontalFlip(
                            p=1
                        )
                    )
        if 'rotation' in augmentation_settings:
            if augmentation_settings['rotation'] is not None:
                transforms_list.append(
                    A.SafeRotate(
                        limit=augmentation_settings['rotation'].get('limit', 0),
                        p=1
                    )
                )

        transform = A.Compose(
            transforms=transforms_list,
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=True),
            bbox_params=A.BboxParams(format='coco')
        )
        transformed = transform(image=image, keypoints=keypoints, bboxes=bboxes)

        self.augmented_bboxes = [list(item[:4]) for item in transformed['bboxes']]
        self.augmented_polygons = []

        step = 0
        for key, size in polygon_descriptor.items():
            self.augmented_polygons.append(
                [item for sublist in transformed['keypoints'] for item in sublist][step: step + size])
            step += size

        self.augmented_polygons = [self.augmented_polygons]
        self.augmented_image = transformed['image']

    def __get_image_from_local(self, image_path):
        try:
            return np.array(Image.open(image_path))
        except FileNotFoundError as fnf:
            raise fnf
        except Exception as e:
            raise e

    def __get_image_from_s3(self, image_path, client, bucket_name):
        try:
            response = client.get_object(bucket_name, image_path)
            image = io.BytesIO(response.read())
            return np.array(Image.open(image))
        except ConnectionError:
            raise ConnectionError("Ooops! There seems to be a connection issue. Please Try again in a while.")
        except Exception as e:
            raise e

    def save_to_file(self, destination):
        image_save_path = os.path.splitext(destination)[0]
        cv2.imwrite(
            f'{image_save_path}.jpg',
            self.augmented_image
        )
        return self

    def save_to_s3(self, destination, client, bucket_name):
        image = Image.fromarray(self.augmented_image)
        in_mem_file = io.BytesIO()
        image.save(in_mem_file, format=self._image_ext)
        image = in_mem_file.getvalue()
        image = io.BytesIO(image)
        image.seek(0)

        image_save_path = os.path.splitext(destination)[0]
        client.put_object(
            bucket_name=bucket_name,
            object_name=f'{image_save_path}.{self._image_ext}',
            data=image,
            length=len(image.getvalue())
        )
        return self

    def augment_from_file(self, image_path, augmentation_settings, **kwargs):
        self.__set_image_ext(image_path)
        image = self.__get_image_from_local(image_path)
        self.__transform(image, augmentation_settings, **kwargs)

        return self

    def augment_from_s3(self, image_path, augmentation_settings, client, bucket_name, **kwargs):
        self.__set_image_ext(image_path)
        image = self.__get_image_from_s3(image_path, client, bucket_name)
        self.__transform(image, augmentation_settings, **kwargs)

        return self

    def augment_from_bytes(self, image_bytes, augmentation_settings, **kwargs):
        self._image_ext = 'jpeg'
        image = np.array(Image.open(io.BytesIO(image_bytes)))
        self.__transform(image, augmentation_settings, **kwargs)

        return self

    def to_base64(self, with_mime_type=True):
        image = Image.fromarray(self.augmented_image)
        in_mem_file = io.BytesIO()
        image.save(in_mem_file, format=self._image_ext if self._image_ext is not None else 'jpeg')
        image_bytes = in_mem_file.getvalue()
        base64_image = base64.b64encode(image_bytes).decode()

        if with_mime_type:
            return f"data:image/{self._image_ext};base64,{base64_image}"

        return base64_image
