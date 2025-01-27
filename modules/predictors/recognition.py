import os
from PIL import Image

import numpy as np
import cv2

from paddleocr import PaddleOCR
import paddle
from paddle import inference
from paddle.base.libpaddle import PaddleInferPredictor
from ppocr.postprocess import build_post_process
import tools.infer.utility as utility

from .abstract_predictor import Predictor
from .prediction_result import RecognitionPredictionResult


class SVTR_RecognitionPredictor(Predictor):
    def __init__(self,
                 model_path: str) -> None:
        self.model_path = model_path
        self.char_path = os.path.join(model_path, "charset.txt")

        self.use_gpu = paddle.device.is_compiled_with_cuda()
        self.__model = None

        self.rec_img_shape = [64, 256, 3]
        self.rec_img_height = self.rec_img_shape[0]
        self.rec_img_width = self.rec_img_shape[1]

        self.batch_num = 12

        self.postprocess_params = {"name": "CTCLabelDecode",
                                   "character_dict_path": self.char_path,
                                   "use_space_char": True}
        self.postprocess_op = build_post_process(self.postprocess_params)

    def __create_config(self) -> inference.Config:
        params_file_path = f"{self.model_path}/inference.pdiparams"
        model_file_path = f"{self.model_path}/inference.pdmodel"

        config = inference.Config(model_file_path, params_file_path)

        if self.use_gpu:
            gpu_id = utility.get_infer_gpuid()
            config.enable_use_gpu(500, gpu_id)

        else:
            config.disable_gpu()

        config.enable_memory_optim()
        config.disable_glog_info()
        config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        config.delete_pass("matmul_transpose_reshape_fuse_pass")
        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        return config

    @property
    def model(self) -> PaddleInferPredictor:
        if self.__model is None:
            config = self.__create_config()
            self.__model = inference.create_predictor(config)

        return self.__model

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        img = np.array(image)
        img = cv2.resize(
            img, (self.rec_img_width, self.rec_img_height), interpolation=cv2.INTER_LINEAR)
        img = img.astype("float32")
        img = img.transpose((2, 0, 1)) / 255
        img -= 0.5
        img /= 0.5
        img = img[np.newaxis, :]

        return img

    def __get_tensors(self) -> tuple[list, list]:
        # Get input tensor
        input_names = self.model.get_input_names()

        for name in input_names:
            input_tensor = self.model.get_input_handle(name)

        # Get output tensor
        output_names = self.model.get_output_names()
        output_tensors = []

        for output_name in output_names:
            output_tensor = self.model.get_output_handle(output_name)
            output_tensors.append(output_tensor)

        return input_tensor, output_tensors

    def __create_preds(self, img: np.ndarray):
        input_tensor, output_tensors = self.__get_tensors()

        img_list = [img]

        img_num = len(img_list)
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))

        indices = np.argsort(np.array(width_list))
        preds = [["", 0.0]] * img_num

        for beg_img_no in range(0, img_num, self.batch_num):
            end_img_no = min(img_num, beg_img_no + self.batch_num)
            img_batch = []

            max_wh_ratio = self.rec_img_width / self.rec_img_height
            wh_ratio_list = []

            for ino in range(beg_img_no, end_img_no):
                height, width = img_list[indices[ino]].shape[0:2]
                wh_ratio = width * 1.0 / height
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
                wh_ratio_list.append(wh_ratio)

                img_batch.append(img_list[indices[ino]])

            img_batch = np.concatenate(img_batch)
            img_batch = img_batch.copy()
            input_tensor.copy_from_cpu(img_batch)

            self.model.run()
            outputs = []
            for output_tensor in output_tensors:
                output = output_tensor.copy_to_cpu()
                outputs.append(output)

            result = self.postprocess_op(outputs,
                                         return_word_box=False,
                                         wh_ratio_list=wh_ratio_list,
                                         max_wh_ratio=max_wh_ratio)

            for rno in range(len(result)):
                preds[indices[beg_img_no + rno]] = result[rno]

        return preds

    def create_result(self, preds) -> RecognitionPredictionResult:
        result = RecognitionPredictionResult(text=preds[0][0],
                                             probability=preds[0][1],
                                             id2label={0: "text"})

        return result

    def predict(self, image) -> RecognitionPredictionResult:
        img = self.preprocess_image(image)
        preds = self.__create_preds(img)
        result = self.create_result(preds)

        return result
