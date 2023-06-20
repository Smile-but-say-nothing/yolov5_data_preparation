import numpy as np
import glob
import os
import argparse
import random
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
from xml.dom import minidom as dom
from tqdm import tqdm
from utils import get_file_prefix
random.seed(42)

class XMLGenerator(object):
    def __init__(self, xml_name: str):
        self.doc = dom.Document()
        self.xml_name = xml_name
    
    def create_append_node(self, node_name, root_node=None):
        """创建一个新node并将node添加到root_node下"""
        new_node = self.doc.createElement(node_name)
        if root_node is not None:
            root_node.appendChild(new_node)
        else:
            self.doc.appendChild(new_node)
        return new_node
    
    def create_text_node(self, node_name, node_value, root_node):
        """
        创建一个新node，然后在该node中添加一个text_node，
        最后将node添加到root_node下
        """
        new_node = self.doc.createElement(node_name)
        node_data = self.doc.createTextNode(node_value)
        new_node.appendChild(node_data)
        root_node.appendChild(new_node)
    
    def create_object_node(self, info_dict: dict = None, root_node: str = None):
        if (info_dict is None) or (root_node is None):
            return
        object_node = self.create_append_node('object', root_node)
        self.create_text_node("name", info_dict.pop("name"), object_node)
        box_node = self.create_append_node('bndbox', object_node)
        self.create_text_node("xmin", str(info_dict.pop("xmin")), box_node)
        self.create_text_node("ymin", str(info_dict.pop("ymin")), box_node)
        self.create_text_node("xmax", str(info_dict.pop("xmax")), box_node)
        self.create_text_node("ymax", str(info_dict.pop("ymax")), box_node)
    
    def save_xml(self):
        if not os.path.exists(os.path.split(self.xml_name)[0]):
            os.mkdir(os.path.split(self.xml_name)[0])
        f = open(self.xml_name, "w")
        self.doc.writexml(f, addindent="\t", newl="\n")
        f.close()


def make_xml(xml_path, object: dict):
    xml_generator = XMLGenerator(xml_path)
    # xml root node
    node_root = xml_generator.create_append_node('annotation')
    xml_generator.create_text_node(node_name='filename', node_value=object['filename'], root_node=node_root)
    # size
    node_size = xml_generator.create_append_node('size', root_node=node_root)
    xml_generator.create_text_node(node_name='height', node_value=str(object['height']), root_node=node_size)
    xml_generator.create_text_node(node_name='width', node_value=str(object['width']), root_node=node_size)
    xml_generator.create_text_node(node_name='depth', node_value=str(object['depth']), root_node=node_size)
    # object
    xml_generator.create_object_node(info_dict=object, root_node=node_root)
    # XML write
    xml_generator.save_xml()


def crop(opt):
    imgs_paths = glob.glob(os.path.join(opt.img_folder_path, '*.jpg'))
    print(f'[INFO] Find {len(imgs_paths)} images.')
    for r_p in tqdm(imgs_paths, desc='[INFO] Cropping'):
        img = Image.open(r_p)
        w, h = img.size
        root = ET.parse(os.path.join(opt.xml_folder_path, os.path.split(r_p)[1].replace('.jpg', '.xml'))).getroot()
        object_set = root.findall('object')
        object_counter = 0
        for obj in object_set:
            bndbox = [int(e.text) for e in obj.find('bndbox')]
            xmin, ymin, xmax, ymax = bndbox
            gt_w, gt_h = xmax - xmin, ymax - ymin
            dxmin, dxmax = random.randint(int(opt.width_scaler[0] * gt_w), int(opt.width_scaler[1] * gt_w)), random.randint(int(opt.width_scaler[0] * gt_w), int(opt.width_scaler[1] * gt_w))
            dymin, dymax = random.randint(int(opt.height_scaler[0] * gt_h), int(opt.height_scaler[1] * gt_h)), random.randint(int(opt.height_scaler[0] * gt_h), int(opt.height_scaler[1] * gt_h))
            xmin_new = xmin - dxmin if xmin - dxmin > 0 else 0
            ymin_new = ymin - dymin if ymin - dymin > 0 else 0
            xmax_new = xmax + dxmax if xmax + dxmax < w else w
            ymax_new = ymax + dymax if ymax + dymax < h else h
            region = img.crop((xmin_new, ymin_new, xmax_new, ymax_new))
            factor = 1
            # while factor * region.size[0] < 460:
            #     factor += 0.1
            # region = region.resize((int(region.size[0] * factor), int(region.size[1] * factor)))
            if 0 not in region.size:
                region.save(os.path.join(opt.dst_img_path, f'{get_file_prefix(r_p)}_{str(object_counter)}.jpg'))
            # new bndbox
            bndbox = [int(dxmin * factor), int(dymin * factor), int((dxmin + gt_w) * factor), int((dymin + gt_h) * factor)]
            object_dict = {
                'filename': str(object_counter).rjust(6, "0") + '.jpg',
                'width': region.size[0], 'height': region.size[1], 'depth': 3,
                'name': str(obj.find('name').text),
                'xmin': bndbox[0], 'ymin': bndbox[1], 'xmax': bndbox[2], 'ymax': bndbox[3]}
            make_xml(os.path.join(opt.dst_anno_path, f'{get_file_prefix(r_p)}_{str(object_counter)}.xml'), object_dict)
            object_counter += 1
    print(f"[INFO] Crop object regions, Done.")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder_path', type=str, required=True, help='images folder path')
    parser.add_argument('--xml_folder_path', type=str, required=True, help='annotations folder path')
    parser.add_argument('--save_dir', type=str, default='./output', help='save dir')
    parser.add_argument('--width_scaler', type=float, default=[0.1, 0.3], nargs='+', help='width scaler for width sample')
    parser.add_argument('--height_scaler', type=float, default=[0.1, 0.3], nargs='+', help='height scaler for width sample')
    opt = parser.parse_args()
    print(f"[INFO] Options: {opt}")
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)
        os.mkdir(os.path.join(opt.save_dir, 'images'))
        os.mkdir(os.path.join(opt.save_dir, 'Annotations'))
        print(f"[INFO] Creating save_dir: {opt.save_dir}, Done.")
    opt.dst_img_path = os.path.join(opt.save_dir, 'images')
    opt.dst_anno_path = os.path.join(opt.save_dir, 'Annotations')
    crop(opt)
    
