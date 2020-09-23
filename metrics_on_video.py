import random
import os
import json
import glob
import cv2 as cv

CREATE_ANNOS = True
CALCULATE_METRICS = True
THS = [0.99, 0.98, 0.95]

dstDir = '/mnt/datos/custom_datasets/simulacro/no_guns/Simulacro_guns_annos'
uris = glob.glob('/mnt/datos/custom_datasets/simulacro/no_guns/Simulacro/*.mp4')

VIS_RESULT = True
confidence = 0.99
def drawAnnos(annoGuns, img, fname):
    for gun in annoGuns:
        (x0, y0, x1, y1, c) = gun
        x1, y1 = int(x1), int(y1)
        x0, y0 = int(x0), int(y0)
        if c >= confidence:
            cv.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), thickness=2)
            cv.putText(img, str(int(c*100)), (x0, y0-20), cv.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)

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

    cfg.MODEL.WEIGHTS = "/mnt/datos/experiments/guns_detection/faster_rcnn_R_50_FPN_1x[LR=0.002][('guns_granada_train', 'guns_simulacro_1', 'guns_simulacro_7')]FROM[faster_rcnn_R_50_FPN_1x[LR=0.002][('unity_synthetic_2500',)]]/model_final.pth"

    predictor = DefaultPredictor(cfg)

    testImgIdx = 0
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
            if VIS_RESULT:
                if any([bbox[4] >= confidence for bbox in bboxes]):
                    drawAnnos(bboxes, img, '')
                    cv.imwrite(os.path.join('fp_no_guns_detections', f'img{testImgIdx}.jpg'), img)
                    testImgIdx+=1
                    # factor = .5
                    # img = cv.resize(img, (0, 0), fx=factor, fy=factor)
                    # cv.imshow('win', img)
                    # k = cv.waitKey(0)
            imagesAnnos.append({'frameIdx': frameIdx, 'bboxes': bboxes})

            frameIdx += 1
            if frameIdx % 10 == 0:
                print(f"frameIdx: {frameIdx}, boxes: {len(bboxes)}")

        # with open(os.path.join(dstDir, os.path.splitext(os.path.basename(uri))[0]) + '_annos.json', 'w') as f:
        #     print(imagesAnnos)
        #     json.dump(imagesAnnos, f)

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
