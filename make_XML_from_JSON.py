import argparse
import glob
import json
import os.path
from xml.dom import minidom as dom
from PIL import Image
from tqdm import tqdm


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
    xml_generator.create_text_node(node_name='folder', node_value=object['folder'], root_node=node_root)
    xml_generator.create_text_node(node_name='filename', node_value=object['filename'], root_node=node_root)
    xml_generator.create_text_node(node_name='database', node_value=object['database'], root_node=node_root)
    # size
    node_size = xml_generator.create_append_node('size', root_node=node_root)
    xml_generator.create_text_node(node_name='height', node_value=str(object['height']), root_node=node_size)
    xml_generator.create_text_node(node_name='width', node_value=str(object['width']), root_node=node_size)
    xml_generator.create_text_node(node_name='depth', node_value=str(object['depth']), root_node=node_size)
    # object
    xml_generator.create_object_node(info_dict=object, root_node=node_root)
    # XML write
    xml_generator.save_xml()


def parse_json(opt):
    json_paths = glob.glob(opt.json_path + '*.json')
    for j_p in tqdm(json_paths, desc='Converting JSON to XML'):
        xml_path = opt.anno_path + os.path.split(j_p)[1].replace('.json', '.xml')
        img_path = opt.img_path + os.path.split(j_p)[1].replace('.json', '.jpg')
        img = Image.open(img_path)
        w, h = img.size
        minX, minY, maxX, maxY = 1e9, 1e9, 0, 0
        with open(j_p, 'r') as j_f:
            json_content = json.load(j_f)
            object_dict = {'folder': 'VoMont_Sanitation_suit', 'filename': os.path.split(j_p)[1].replace('.json', '.jpg'), 'database': 'DeepFashion2',
                           'width': w, 'height': h, 'depth': 3}
            for i in range(1, 3):
                item_i = 'item' + str(i)
                if item_i in json_content.keys():
                    item = json_content[item_i]
                    # name = item['category_name']
                    bbox = item['bounding_box']
                    if bbox[0] < minX:
                        minX = bbox[0]
                    if bbox[1] < minY:
                        minY = bbox[1]
                    if bbox[2] > maxX:
                        maxX = bbox[2]
                    if bbox[3] > maxY:
                        maxY = bbox[3]
            object_dict['name'] = 'suit'
            object_dict['xmin'] = minX
            object_dict['ymin'] = minY
            object_dict['xmax'] = maxX
            object_dict['ymax'] = maxY
            make_xml(xml_path, object_dict)
            # print(json.dumps(json_content, indent=4))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='./raw_json/', help='source path of jsons')
    parser.add_argument('--anno_path', type=str, default='./raw_annos/', help='dst path of annos')
    parser.add_argument('--img_path', type=str, default='./raw_imgs/', help='dst path of imgs')
    
    opt = parser.parse_args()
    parse_json(opt)