#!/usr/bin/python

# pip install lxml

import sys
import os
import json
import xml.etree.ElementTree as ET
import glob
import pickle

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {'Pistol': 1}

def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def get_categories(xml_files):
    classes_names = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name: i for i, name in enumerate(classes_names)}


def convert(xml_files, json_file):
    json_dict = {"images": [], 'categories': [], "annotations": [], "type": "instances"}

    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(xml_files)
    bnd_id = START_BOUNDING_BOX_ID
    for xml_file in sorted(xml_files):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = get(root, "path")
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, "filename", 1).text
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))

        fname = filename.split('\\')[-1]

        objects = []
        for obj in get(root, "object"):
            category = get_and_check(obj, "name", 1).text
            # TODO: Change to match edgecase dataset
            if category == 'Knife':
                continue
            else:
                category = 'Pistol'

            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(float(get_and_check(bndbox, "xmin", 1).text)) - 1
            ymin = int(float(get_and_check(bndbox, "ymin", 1).text)) - 1
            xmax = int(float(get_and_check(bndbox, "xmax", 1).text))
            ymax = int(float(get_and_check(bndbox, "ymax", 1).text))
            assert xmax > xmin
            assert ymax > ymin
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {
                "area": int(o_width * o_height),
                "iscrowd": 0,
                "image_id": fname,
                "bbox": [xmin, ymin, o_width, o_height],
                "category_id": category_id,
                "ignore": 0,
                "segmentation": [],
                "id": bnd_id,
            }
            json_dict["annotations"].append(ann)
            objects.append(ann)
            bnd_id = bnd_id + 1

        size = get_and_check(root, "size", 1)
        width = int(float(get_and_check(size, "width", 1).text))
        height = int(float(get_and_check(size, "height", 1).text))
        image = {
            "file_name": fname,
            "height": height,
            "width": width,
            #"objects": objects,
            "id": fname,
        }
        json_dict["images"].append(image)

    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, "w") as f:
        json.dump(json_dict, f)


if __name__ == "__main__":
    from glob import glob
    import shutil
    dirs = [
        '/home/datos/Downloads/Images',
    ]

    RAW_DATASET = '/home/datos/Downloads/Images'
    ROOT_DATASET_FOLDER = '/media/datos/shared_datasets/simulacro_guns'

    allFiles = glob(os.path.join(RAW_DATASET, '*.jpg'))
    jpgDir = os.path.join(ROOT_DATASET_FOLDER, 'allCams')
    os.makedirs(jpgDir, exist_ok=True)
    for f in allFiles:
        if not os.path.exists(os.path.join(jpgDir, f)):
            shutil.copy(f, jpgDir)

    matchTuples = [('*.xml', 'all_cams') , ('Cam1*.xml', 'cam1'), ('Cam5*.xml', 'cam5'), ('Cam7*.xml', 'cam7')]

    for matchStr, datasetName in matchTuples:
        allFiles = glob(os.path.join(RAW_DATASET, matchStr))
        fname = os.path.join(ROOT_DATASET_FOLDER, f'{datasetName}.json')
        print("Number of xml files: {}".format(len(allFiles)))
        convert(allFiles, fname)
        print("Success: {}".format(fname))