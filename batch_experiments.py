#import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)
import random
import os
import json
import argparse
import shutil
import re
import openpyxl
import glob

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('id', type=int, action="store")

args = parser.parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.id}"

from detectron2.engine import launch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import importlib.util

from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.evaluation.coco_evaluation import _evaluate_predictions_on_coco

from detectron2.data import build_detection_test_loader
from detectron2.modeling import build_model

spec = importlib.util.spec_from_file_location("custom_datasets", os.path.join(os.path.dirname(__file__), 'custom_datasets.py'))
custom_datasets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_datasets)

DATASETS_REAL = ['guns_granada_train', 'guns_simulacro_1', 'guns_simulacro_7']
DATASETS_SYTH = ['guns_edgecase', 'unity_synthetic_500', 'unity_synthetic_1000', 'unity_synthetic_2500', 'unity_synthetic_5000']


SHOW_DATASET = False
TRAIN = False
SAVE_DATASET_PREDICTIONS = False
SHOW_PREVIEW = False
PREVIEW_ON_DIR = False
MAKE_ANNOS = False
SHOW_EVALUATION = True

class JsonCOCOEvaluator(COCOEvaluator):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        with open(os.path.join(self._output_dir, 'coco_instances_results.json')) as f:
            self._coco_results = json.load(f)
        
    def evaluate(self, theshold):
        task ='bbox'
        coco_eval = (
            _evaluate_predictions_on_coco(
                self._coco_api, self._coco_results, task, kpt_oks_sigmas=self._kpt_oks_sigmas
            )
            if len(self._coco_results) > 0
            else None  # cocoapi does not handle empty results very well
        )
        res = self._derive_coco_results(
            coco_eval, task, class_names=self._metadata.get("thing_classes")
        )

        FP = FN = TP = 0.
        prob_preds = []
        for cocoRes in coco_eval.evalImgs[:int(len(coco_eval.evalImgs)/4)]:
            if cocoRes is None:
                continue
            for det, score in zip(cocoRes['dtMatches'][0], cocoRes['dtScores']):
                # No gt match
                if det == 0.:
                    prob_preds.append([0, score])
                    if score >= theshold:
                        FP += 1
                else:
                    prob_preds.append([1, score])
                    if score >= theshold:
                        TP += 1
                    else:
                        FN += 1

            for gt in cocoRes['gtMatches'][0]:
                if gt == 0.:
                    prob_preds.append([1, 0.])
                    FN += 1

        res.update({'TP': TP, 'FP': FP, 'FN': FN, 'ProbPreds': prob_preds})
        return {task: res}

custom_datasets.loadDatasets()

EXPERIMENTS_OUTPUT_PATH = '/mnt/datos/experiments/guns_detection'

networks = {
    'faster_rcnn_R_50_FPN_1x': {
        'cfg': '../../build/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml',
        'weights': 'detectron2://COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/model_final_b275ba.pkl'
    },
    'faster101': {
        'cfg': '../../build/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml',
        'weights': 'detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl'
    },
}

experiments = [
    {'steps': 40000, 'lr': 0.002, 'net': 'faster_rcnn_R_50_FPN_1x', 'train': ('guns_edgecase',), 'test': ("guns_granada_test",)},
    {'steps': 40000, 'lr': 0.002, 'net': 'faster_rcnn_R_50_FPN_1x', 'train': ('unity_synthetic_500',), 'test': ("guns_granada_test",),},
    {'steps': 40000, 'lr': 0.002, 'net': 'faster_rcnn_R_50_FPN_1x', 'train': ('unity_synthetic_1000',), 'test': ("guns_granada_test",),},
    {'steps': 40000, 'lr': 0.002, 'net': 'faster_rcnn_R_50_FPN_1x', 'train': ('unity_synthetic_2500',), 'test': ("guns_granada_test",),},

    {'steps': 80000, 'lr': 0.002, 'net': 'faster_rcnn_R_50_FPN_1x', 'train': ('guns_granada_train',), 'test': ('guns_granada_test',), 'from': f"{EXPERIMENTS_OUTPUT_PATH}/faster_rcnn_R_50_FPN_1x[LR=0.002][('guns_edgecase',)]/model_0039999.pth"},
    {'steps': 80000, 'lr': 0.002, 'net': 'faster_rcnn_R_50_FPN_1x', 'train': ('guns_granada_train',), 'test': ('guns_granada_test',), 'from': f"{EXPERIMENTS_OUTPUT_PATH}/faster_rcnn_R_50_FPN_1x[LR=0.002][('unity_synthetic_500',)]/model_0039999.pth"},
    {'steps': 80000, 'lr': 0.002, 'net': 'faster_rcnn_R_50_FPN_1x', 'train': ('guns_granada_train',), 'test': ('guns_granada_test',), 'from': f"{EXPERIMENTS_OUTPUT_PATH}/faster_rcnn_R_50_FPN_1x[LR=0.002][('unity_synthetic_1000',)]/model_0039999.pth"},
    {'steps': 80000, 'lr': 0.002, 'net': 'faster_rcnn_R_50_FPN_1x', 'train': ('guns_granada_train',), 'test': ('guns_granada_test',), 'from': f"{EXPERIMENTS_OUTPUT_PATH}/faster_rcnn_R_50_FPN_1x[LR=0.002][('unity_synthetic_2500',)]/model_0039999.pth"},
    {'steps': 80000, 'lr': 0.002, 'net': 'faster_rcnn_R_50_FPN_1x', 'train': ('guns_granada_train', 'guns_simulacro_1', 'guns_simulacro_7',), 'test': ("guns_granada_test",), 'from': f"{EXPERIMENTS_OUTPUT_PATH}/faster_rcnn_R_50_FPN_1x[LR=0.002][('guns_edgecase',)]/model_0039999.pth"},
    {'steps': 80000, 'lr': 0.002, 'net': 'faster_rcnn_R_50_FPN_1x', 'train': ('guns_granada_train', 'guns_simulacro_1', 'guns_simulacro_7',), 'test': ("guns_granada_test",), 'from': f"{EXPERIMENTS_OUTPUT_PATH}/faster_rcnn_R_50_FPN_1x[LR=0.002][('unity_synthetic_500',)]/model_0039999.pth"},
    {'steps': 80000, 'lr': 0.002, 'net': 'faster_rcnn_R_50_FPN_1x', 'train': ('guns_granada_train', 'guns_simulacro_1', 'guns_simulacro_7',), 'test': ("guns_granada_test",), 'from': f"{EXPERIMENTS_OUTPUT_PATH}/faster_rcnn_R_50_FPN_1x[LR=0.002][('unity_synthetic_2500',)]/model_0039999.pth"},
    {'steps': 40000, 'lr': 0.002, 'net': 'faster_rcnn_R_50_FPN_1x', 'train': ('guns_granada_train',), 'test': ("guns_granada_test",)},
    {'steps': 40000, 'lr': 0.002, 'net': 'faster_rcnn_R_50_FPN_1x', 'train': ('guns_granada_train', 'guns_simulacro_1', 'guns_simulacro_7'), 'test': ("guns_granada_test",)},

    {'steps': 40000, 'lr': 0.002, 'net': 'faster_rcnn_R_50_FPN_1x', 'train': ('guns_simulacro_1', 'guns_simulacro_7'), 'test': ("guns_granada_test",)},
    {'steps': 80000, 'lr': 0.002, 'net': 'faster_rcnn_R_50_FPN_1x', 'train': ('guns_granada_train',), 'test': ("guns_granada_test",)},

    {'steps': 40000, 'lr': 0.002, 'net': 'faster101', 'train': ('guns_edgecase',), 'test': ('guns_granada_test',),},
    {'steps': 40000, 'lr': 0.002, 'net': 'faster101', 'train': ('unity_synthetic_500',), 'test': ('guns_granada_test',),},
    {'steps': 40000, 'lr': 0.002, 'net': 'faster101', 'train': ('unity_synthetic_1000',), 'test': ('guns_granada_test',),},
    {'steps': 40000, 'lr': 0.002, 'net': 'faster101', 'train': ('unity_synthetic_2500',), 'test': ('guns_granada_test',),},

    {'steps': 80000, 'lr': 0.002, 'net': 'faster101', 'train': ('guns_granada_train',), 'test': ('guns_granada_test',), 'from': f"{EXPERIMENTS_OUTPUT_PATH}/faster101[LR=0.002][('guns_edgecase',)]/model_0039999.pth"},
    {'steps': 80000, 'lr': 0.002, 'net': 'faster101', 'train': ('guns_granada_train',), 'test': ('guns_granada_test',), 'from': f"{EXPERIMENTS_OUTPUT_PATH}/faster101[LR=0.002][('unity_synthetic_500',)]/model_0039999.pth"},
    {'steps': 80000, 'lr': 0.002, 'net': 'faster101', 'train': ('guns_granada_train',), 'test': ('guns_granada_test',), 'from': f"{EXPERIMENTS_OUTPUT_PATH}/faster101[LR=0.002][('unity_synthetic_1000',)]/model_0039999.pth"},
    {'steps': 80000, 'lr': 0.002, 'net': 'faster101', 'train': ('guns_granada_train',), 'test': ('guns_granada_test',), 'from': f"{EXPERIMENTS_OUTPUT_PATH}/faster101[LR=0.002][('unity_synthetic_2500',)]/model_0039999.pth"},

    {'steps': 20000, 'lr': 0.002, 'net': 'faster_rcnn_R_50_FPN_1x', 'train': ('guns_edgecase',),},
    {'steps': 20000, 'lr': 0.002, 'net': 'faster_rcnn_R_50_FPN_1x', 'train': ('guns_edgecase', 'guns_granada_train', 'guns_simulacro_1', 'guns_simulacro_7'), 'test': ("guns_granada_test",),},
    {'steps': 20000, 'lr': 0.002, 'net': 'faster_rcnn_R_50_FPN_1x', 'train': ('guns_edgecase', 'guns_granada_train', 'guns_simulacro_1', 'guns_simulacro_7', 'alamy'), 'test': ("guns_granada_test",), 'test': ("guns_granada_test",),},
    {'steps': 20000, 'lr': 0.002, 'net': 'faster_rcnn_R_50_FPN_1x', 'train': ('unity_synthetic_1000',), 'test': ("guns_granada_test",),},
    {'steps': 20000, 'lr': 0.002, 'net': 'faster_rcnn_R_50_FPN_1x', 'train': ('unity_synthetic_2500',), 'test': ("guns_granada_test",),},
    {'steps': 20000, 'lr': 0.002, 'net': 'faster_rcnn_R_50_FPN_1x', 'train': ('unity_synthetic_1000', 'guns_granada_train', 'guns_simulacro_1', 'guns_simulacro_7'), 'test': ("guns_granada_test",),},
    {'steps': 20000, 'lr': 0.002, 'net': 'faster_rcnn_R_50_FPN_1x', 'train': ('unity_synthetic_2500', 'guns_granada_train', 'guns_simulacro_1', 'guns_simulacro_7'), 'test': ("guns_granada_test",),},


]

def train(exp):
    print(f'Executing experiment: \n{exp}\n')
    cfg = get_cfg()

    cfg.merge_from_file(networks[exp['net']]['cfg'])

    cfg.MODEL.WEIGHTS = exp.get('from', networks[exp['net']]['weights'])

    cfg.INPUT.MAX_SIZE_TEST = 2500
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.DATASETS.TRAIN = exp['train']
    cfg.DATASETS.TEST = exp['test']
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = exp['lr']
    cfg.SOLVER.MAX_ITER = exp['steps']  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True

    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    cfg.TEST.EVAL_PERIOD = 0
    #cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 640, 672, 704, 736, 768, 800)

    if 'from' in exp:
        trainFromStr = exp['from'].split('/')[-2]
        cfg.OUTPUT_DIR = f"{EXPERIMENTS_OUTPUT_PATH}/{exp['net']}[LR={exp['lr']}][{str(exp['train'])}]FROM[{trainFromStr}]"
        #if not os.path.exists(cfg.OUTPUT_DIR):
        #    print(f"copying {trainFromFolder}\nto {cfg.OUTPUT_DIR}")
        #    shutil.copytree(trainFromFolder, cfg.OUTPUT_DIR)
    else:
        cfg.OUTPUT_DIR = f"{EXPERIMENTS_OUTPUT_PATH}/{exp['net']}[LR={exp['lr']}][{str(exp['train'])}]"

    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    class Trainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg_, dataset_name):
            return COCOEvaluator(dataset_name, cfg_, distributed=False)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()

def test(exp, selected_dataset, theshold):
    print(f'Testing experiment: \n{exp}\n')
    cfg = get_cfg()
    cfg.merge_from_file(networks[exp['net']]['cfg'])
    cfg.DATASETS.TEST = exp['test']
    cfg.INPUT.MAX_SIZE_TEST = 2500
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.DATALOADER.NUM_WORKERS = 2

    if 'from' in exp:
        trainFromStr = exp['from'].split('/')[-2]
        
        cfg.OUTPUT_DIR = f"{EXPERIMENTS_OUTPUT_PATH}/{exp['net']}[LR={exp['lr']}][{str(exp['train'])}]FROM[{trainFromStr}]"
    else:
        cfg.OUTPUT_DIR = f"{EXPERIMENTS_OUTPUT_PATH}/{exp['net']}[LR={exp['lr']}][{str(exp['train'])}]"

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')

    if all([ds in DATASETS_REAL or ds in DATASETS_SYTH for ds in exp['train']]):
        output_dir = f"{cfg.OUTPUT_DIR}/{selected_dataset}"
        model = build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )
        evaluator = COCOEvaluator(selected_dataset, cfg, False, output_dir=output_dir)
        val_loader = build_detection_test_loader(cfg, selected_dataset)
        inference_on_dataset(model, val_loader, evaluator)

if __name__ == '__main__':
    module = args.id - 1
    size = 2

    num_gpus = 2
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14

    for i, exp in enumerate(experiments):
        if i % size != module:
            continue
        train(exp)
        # launch(
        #     train,
        #     num_gpus,
        #     dist_url="tcp://127.0.0.1:{}".format(port),
        #     args=(exp,)
        # )
