import random
import os
import json
import glob

CREATE_ANNOS = True
CALCULATE_METRICS = True
THS = [0.99, 0.98, 0.95]

dstDir = '/mnt/datos/custom_datasets/simulacro/no_guns/Simulacro_guns_annos'
uris = glob.glob('/media/datos/shared_datasets/guns_granada/Test/*')


# def create_annos(cfg, dstPath, uris):
#
if CREATE_ANNOS:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    from detectron2.utils.logger import setup_logger

    setup_logger()

    import cv2 as cv
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    cfg = get_cfg()
    if True:
        cfg.merge_from_file("../../build/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"  # initialize from model zoo
    else:
        cfg.merge_from_file("../../build/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"  # initialize from model zoo

    cfg.INPUT.MAX_SIZE_TEST = 2500
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.MODEL.WEIGHTS = "/mnt/datos/experiments/guns_detection/faster_rcnn_R_50_FPN_1x[LR=0.002][('guns_granada_train')]FROM[faster_rcnn_R_50_FPN_1x[LR=0.002][('guns_edgecase',)]]/model_final.pth"

    predictor = DefaultPredictor(cfg)

    for uri in uris[:]:
        print(uri)
        cap = cv.VideoCapture(uri)
        frameIdx = 0
        imagesAnnos = []
        while True:
            for i in range(1):
                ret, img = cap.read()
            if not ret or not cap.isOpened():
                break

            preds = predictor(img)["instances"].to("cpu")

            bboxes = [b.numpy().tolist() for b in preds.pred_boxes] if preds.pred_boxes else []
            scores = preds.scores.numpy().tolist()
            for i in range(len(bboxes)):
                bboxes[i].append(scores[i])
            imagesAnnos.append({'frameIdx': frameIdx, 'bboxes': bboxes})

            frameIdx += 1
            if frameIdx % 10 == 0:
                print(frameIdx)

        with open(os.path.join(dstDir, os.path.splitext(os.path.basename(uri))[0]) + '_annos.json', 'w') as f:
            print(imagesAnnos)
            json.dump(imagesAnnos, f)

if CALCULATE_METRICS:
    for th in THS:
        metrics = {}
        for uri in uris[:]:
            basename = os.path.splitext(os.path.basename(uri))[0]
            with open(os.path.join(dstDir, basename) + '_annos.json') as f:
                imagesAnnos = json.load(f)
                numDetections = len([bbox for anno in imagesAnnos for bbox in anno['bboxes'] if bbox[4] >= th])
                metrics[basename] = {'FP': numDetections, 'NUM_FRAMES': len(imagesAnnos)}

        with open(os.path.join(dstDir, f'metrics_summary_th-{th}.json'), 'w') as f:
            json.dump(metrics, f)

        detailList = ', '.join([f"{v['FP']}" for k, v in metrics.items()])
        numFrames = sum([v['NUM_FRAMES'] for k, v in metrics.items()])
        print(f"FP{th}: {sum([m['FP'] for m in metrics.values()])}/{numFrames} ({detailList})")
