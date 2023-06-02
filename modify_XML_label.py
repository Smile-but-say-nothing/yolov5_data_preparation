import os
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm
import glob


def stats(opt):
    xml_path = glob.glob(os.path.join(opt.anno_path, '*.xml'))
    object_name_set = {}
    for p in tqdm(xml_path, desc='[INFO] Stats Label'):
        root = ET.parse(p).getroot()
        object_set = root.findall('object')
        for obj in object_set:
            name = obj.find('name').text
            if name not in object_name_set:
                object_name_set[name] = 1
            else:
                object_name_set[name] += 1
    print(f"[INFO] Label stats: {object_name_set}, Done.")


def rename(opt):
    xml_path = glob.glob(os.path.join(opt.anno_path, '*.xml'))
    for p in tqdm(xml_path, desc='[INFO] Rename Label'):
        if not os.path.exists(p):
            continue
        root = ET.parse(p).getroot()
        object_set = root.findall('object')
        for obj in object_set:
            name = obj.find('name').text
            if name in opt.before_classes:
                obj.find('name').text = opt.after_classe
                tree = ET.ElementTree(root)
                tree.write(p)
    print(f"[INFO] Label rename, Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_path', default=None, required=True, help='annotations folder path')
    parser.add_argument('--stats', action='store_true', help='count label numbers')
    parser.add_argument('--rename', action='store_true', help='rename labels')
    parser.add_argument('--before_classes', type=str, nargs='+', help='old classes name')
    parser.add_argument('--after_class', type=str, help='new class name')
    opt = parser.parse_args()
    print(f"[INFO] Options: {opt}")
    if opt.stats:
        stats(opt)
    if opt.rename:
        rename(opt)

