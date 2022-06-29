# YOLOv5 :로켓: by Ultralytics, GPL-3.0 license
"""
PyTorch Hub models https://pytorch.org/hub/ultralytics_yolov5/
Usage:
    import torch
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model = torch.hub.load('ultralytics/yolov5:master', 'custom', 'path/to/yolov5s.onnx')  # file from branch
"""
import torch
def _create(name, pretrained=True, channels=3, classes=1, autoshape=True, verbose=True, device=str):
    """Creates or loads a YOLOv5 model
    Arguments:
        name (str): model name 'yolov5s' or path 'path/to/best.pt'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes
        autoshape (bool): apply YOLOv5 .autoshape() wrapper to model
        verbose (bool): print all information to screen
        device (str, torch.device, None): device to use for model parameters
    Returns:
        YOLOv5 model
    """
    from pathlib import Path
    from models.common import AutoShape, DetectMultiBackend
    from models.yolo import Model
    from utils.downloads import attempt_download
    from utils.general import LOGGER, check_requirements, intersect_dicts, logging
    from utils.torch_utils import select_device
    if not verbose:
        LOGGER.setLevel(logging.WARNING)
    check_requirements(exclude=('tensorboard', 'thop', 'opencv-python'))
    name = Path(name)
    path = 'daebardaebar/tellosibar/best.pt'  # checkpoint path
    try:
        device = select_device(device)
        if pretrained and channels == 3 and classes == 1:
            model = DetectMultiBackend(path, device='')  # download/load FP32 model
            # model = models.experimental.attempt_load(path, map_location=device)  # download/load FP32 model
        else:
            cfg = list((Path(__file__).parent / 'models').rglob(f'{path.stem}.yaml'))[0]  # model.yaml path
            model = Model(cfg, channels, classes)  # create model
            if pretrained:
                ckpt = torch.load(attempt_download(path), map_location=device)  # load
                csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
                csd = intersect_dicts(csd, model.state_dict(), exclude=['anchors'])  # intersect
                model.load_state_dict(csd, strict=False)  # load
                if len(ckpt['model'].names) == classes:
                    model.names = ckpt['model'].names  # set class names attribute
        if autoshape:
            model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS
        return model.to(device)
    except Exception as e:
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'
        s = f'{e}. Cache may be out of date, try `force_reload=True` or see {help_url} for help.'
        raise Exception(s) from e
def custom(path='daebardaebar/tellosibar/best.pt', autoshape=True, _verbose=True, device=str):
    # YOLOv5 custom or local model
    return _create(path, autoshape=autoshape, verbose=_verbose, device=str)

if __name__ == '__main__':
    #model = _create(name='yolov5s', pretrained=True, channels=3, classes=80, autoshape=True, verbose=True)
    model = custom(path='daebardaebar/tellosibar/best.pt')  # custom
    # Verify inference
    from pathlib import Path
    import numpy as np
    from PIL import Image
    from utils.general import cv2
    #imgs = [
    #    'data/images/zidane.jpg',  # filename
    #    Path('data/images/zidane.jpg'),  # Path
    #    'https://ultralytics.com/images/zidane.jpg',  # URI
    #    cv2.imread('data/images/bus.jpg')[:, :, ::-1],  # OpenCV
    #    Image.open('data/images/bus.jpg'),  # PIL
    #    np.zeros((320, 640, 3))]  # numpy
    #results = model(imgs, size=360)  # batched inference
    #results.print()
    #results.save()
