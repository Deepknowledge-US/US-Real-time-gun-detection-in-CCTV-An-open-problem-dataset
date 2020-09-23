
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json, register_coco_instances

def loadDatasets():
    classRemapping = {1: 0}
    meta = {'thing_classes': ['gun'], 'thing_dataset_id_to_contiguous_id': classRemapping}
    
    def register_coco_instances(name, metadata, classRemapping, json_file, image_root):
        DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, None, classRemapping=classRemapping))
        MetadataCatalog.get(name).set(
            json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
        )
    
    ROOT_PATH = '/media/datos/shared_datasets/simulacro_guns'
    register_coco_instances("guns_simulacro_1", meta, classRemapping, json_file=f'{ROOT_PATH}/cam1.json', image_root=f'{ROOT_PATH}/images')
    register_coco_instances("guns_simulacro_5", meta, classRemapping, json_file=f'{ROOT_PATH}/cam5.json', image_root=f'{ROOT_PATH}/images')
    register_coco_instances("guns_simulacro_7", meta, classRemapping, json_file=f'{ROOT_PATH}/cam7.json', image_root=f'{ROOT_PATH}/images')
    register_coco_instances("guns_simulacro_all", meta, classRemapping, json_file=f'{ROOT_PATH}/all_cams.json', image_root=f'{ROOT_PATH}/images')
    
    ROOT_PATH = '/media/datos/shared_datasets/guns_edgecase'
    register_coco_instances("guns_edgecase", meta, classRemapping, json_file=f'{ROOT_PATH}/synthetic_train.json', image_root=f'{ROOT_PATH}/images')
    
    ROOT_PATH = '/media/datos/shared_datasets/guns_granada'
    register_coco_instances("guns_granada_train", meta, classRemapping, json_file=f'{ROOT_PATH}/real_train.json', image_root=f'{ROOT_PATH}/images')
    register_coco_instances("guns_granada_test", meta, classRemapping, json_file=f'{ROOT_PATH}/real_val.json', image_root=f'{ROOT_PATH}/images')
    
    ROOT_PATH = '/media/datos/shared_datasets/unity_syntectic_victory/split-500'
    register_coco_instances("unity_synthetic_500", meta, classRemapping, json_file=f'{ROOT_PATH}/split_coco.json', image_root=f'{ROOT_PATH}')
    ROOT_PATH = '/media/datos/shared_datasets/unity_syntectic_victory/split-1000'
    register_coco_instances("unity_synthetic_1000", meta, classRemapping, json_file=f'{ROOT_PATH}/split_coco.json', image_root=f'{ROOT_PATH}')
    ROOT_PATH = '/media/datos/shared_datasets/unity_syntectic_victory/split-2500'
    register_coco_instances("unity_synthetic_2500", meta, classRemapping, json_file=f'{ROOT_PATH}/split_coco.json', image_root=f'{ROOT_PATH}')
    ROOT_PATH = '/media/datos/shared_datasets/unity_syntectic_victory/split-5000'
    register_coco_instances("unity_synthetic_5000", meta, classRemapping, json_file=f'{ROOT_PATH}/split_coco.json', image_root=f'{ROOT_PATH}')