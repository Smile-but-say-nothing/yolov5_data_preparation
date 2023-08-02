import os
import random
import argparse
import shutil
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
import glob
import cv2
from utils import get_file_name, get_file_prefix


def remove(opt):
    print('[INFO] Run Remove Process!')
    xml_file_names = [f[:-4] for f in os.listdir(opt.anno_path)]
    print(f'[INFO] XML file number: {len(xml_file_names)}')
    images_path = glob.glob(os.path.join(opt.img_path, '*.jpg'))
    for f in images_path:
        if os.path.split(f)[1].replace('.jpg', '') not in xml_file_names:
            os.remove(f)
            print(f'[INFO] {os.path.split(f)[1]} is been deleted because of no corresponding XML file!')
    images_paths = glob.glob(opt.img_path + '*.jpg')
    assert len(xml_file_names) == len(images_paths), f'XML file number {len(xml_file_names)} not equal to image number {len(images_paths)}'


def split(opt):
    print('[INFO] Run Split Process!')
    xml_files = os.listdir(opt.anno_path)
    xml_num = len(xml_files)
    val_test = random.sample(range(xml_num), int((opt.val_rate + opt.test_rate) * xml_num))
    train_set = list(set(range(xml_num)) - set(val_test))
    val_set = random.sample(val_test, int(opt.val_rate / (opt.val_rate + opt.test_rate) * len(val_test)))
    test_set = list(set(val_test) - set(val_set))
    assert len(train_set) + len(val_set) + len(test_set) == xml_num
    with open(opt.txt_path + 'train.txt', 'w') as f:
        for idx in train_set:
            f.write(xml_files[idx].replace('.xml', '\n'))
    with open(opt.txt_path + 'val.txt', 'w') as f:
        for idx in val_set:
            f.write(xml_files[idx].replace('.xml', '\n'))
    with open(opt.txt_path + 'test.txt', 'w') as f:
        for idx in test_set:
            f.write(xml_files[idx].replace('.xml', '\n'))
    print(f'[INFO] txt file saved! split is done.')


def compute(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert(name, opt):
    if os.path.exists(f'{opt.anno_path}{name}.xml'):
        xml_file = open(f'{opt.anno_path}{name}.xml', encoding='utf-8')
    else:
        raise Exception
    label_file = open(os.path.join(opt.save_dir, 'labels', name + '.txt'), 'w', encoding='utf-8')
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    # fix bugs
    if not size:
        w, h = Image.open(f'{opt.img_path}{name}.jpg').size
        print(f"[INFO] {name}.jpg get weight and height from PIL: {w, h}")
    else:
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        if (w, h) == (0, 0):
            w, h = Image.open(f'{opt.img_path}{name}.jpg').size
            print(f"[INFO] {name}.jpg get weight and height from PIL: {w, h}")
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in opt.classes:
            continue
        cls_id = opt.classes.index(cls)
        box = obj.find('bndbox')
        # box: (xmin, xmax, ymin, ymax)
        box = (float(box.find('xmin').text), float(box.find('xmax').text), float(box.find('ymin').text), float(box.find('ymax').text))
        bb = compute((w, h), box)
        label_file.write(f'{str(cls_id)} {" ".join([str(a) for a in bb])}\n')


def label(opt):
    print('[INFO] Run Label Process!')
    if not os.path.exists(os.path.join(opt.save_dir, 'labels')):
        os.mkdir(os.path.join(opt.save_dir, 'labels'))
    for set in ['train', 'val', 'test']:
        txt_name = open(os.path.join(opt.txt_path, set + '.txt')).read().strip().split()
        with open(os.path.join(opt.save_dir, set + '.txt'), 'w') as f:
            for t in tqdm(txt_name, desc=f'[INFO] Convertor runs from {set}.txt'):
                f.write(f'{opt.prefix}{t}.jpg\n')
                convert(t, opt)
    print('[INFO] Data convert done!')


def plot(opt):
    image_paths = glob.glob(os.path.join(opt.img_path, '*'))
    for img_file_path in image_paths[:opt.plot_num]:
        img = cv2.imread(img_file_path)
        xml_file_path = opt.anno_path + os.path.split(img_file_path)[1].split('.')[0] + '.xml'
        root = ET.parse(xml_file_path).getroot()
        ObjectSet = root.findall('object')
        for Object in ObjectSet:
            ObjName = Object.find('name').text
            BndBox = Object.find('bndbox')
            try:
                x1 = int(BndBox.find('xmin').text)
                y1 = int(BndBox.find('ymin').text)
                x2 = int(BndBox.find('xmax').text)
                y2 = int(BndBox.find('ymax').text)
            except ValueError:
                print(os.path.split(img_file_path)[1])
                continue
            cv2.putText(img, ObjName, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(opt.plot_path, os.path.split(img_file_path)[1].split('.')[0] + '.jpg'), img)
    print('[INFO] Plot done!')


def check_deep_file(folder_path):
    folder_paths = os.listdir(folder_path)
    for f_n in folder_paths:
        f_p = folder_path + '/' + f_n
        if os.path.isfile(f_p):
            if os.path.splitext(f_p)[-1] == '.jpg':
                deep_file_path_imgs.append(f_p)
            elif os.path.splitext(f_p)[-1] == '.xml':
                deep_file_path_annos.append(f_p)
            else:
                continue
        else:
            check_deep_file(f_p)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder_path', default=None, required=True, help='image folder path.')
    parser.add_argument('--anno_folder_path', default=None, required=True, help='anno folder path.')
    parser.add_argument('--save_dir', type=str, default=None, required=True, help='save dir of dataset.')
    parser.add_argument('--rename', action='store_true', help='rename images to avoid overwrite.')
    parser.add_argument('--split', action='store_true', help='split the dataset for train, dev and test.')
    parser.add_argument('--train_rate', type=float, default=0.9, help='percent of train data.')
    parser.add_argument('--val_rate', type=float, default=0.05, help='percent of val data.')
    parser.add_argument('--test_rate', type=float, default=0.05, help='percent of test data.')
    parser.add_argument('--label', action='store_true', help='convert the XMLs to yolo labels.')
    parser.add_argument('--prefix', type=str, default='/home/example/Data/project/images/', help='prefix of absolute path of image paths.')
    parser.add_argument('--classes', type=str, nargs='+', default=[], help='classes of interest.')
    parser.add_argument('--plot', action='store_true', help='draw box in a few images.')
    parser.add_argument('--plot_num', type=int, default=10, help='images numbers with drawing box.')
    parser.add_argument('--seed', type=int, default=42, help='random seed.')
    opt = parser.parse_args()
    print(f"[INFO] Options: {opt}")
    assert len(opt.classes) if opt.label else True, 'if need convert XML to label txts, please provide classes of interest.'
    assert opt.split if opt.label else True, 'if need convert XML to label txts, please split data first.'
    opt.anno_path = os.path.join(opt.save_dir, 'Annotations/')
    opt.img_path = os.path.join(opt.save_dir, 'images/')
    opt.txt_path = os.path.join(opt.save_dir, 'ImageSets/')
    opt.plot_path = os.path.join(opt.save_dir, 'plot/')
    # if necessary folders don't exist, mkdir them
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)
    if not os.path.exists(opt.anno_path):
        os.mkdir(opt.anno_path)
    if not os.path.exists(opt.img_path):
        os.mkdir(opt.img_path)
    if not os.path.exists(opt.txt_path):
        os.mkdir(opt.txt_path)
    if not os.path.exists(opt.plot_path):
        os.mkdir(opt.plot_path)
    # check deep images and annotations
    deep_file_path_imgs, deep_file_path_annos = [], []
    if opt.img_folder_path is not None and opt.anno_folder_path is not None:
        check_deep_file(opt.img_folder_path)
        check_deep_file(opt.anno_folder_path)
        # print(deep_file_path_annos)
        for idx, path in enumerate(tqdm(deep_file_path_annos, desc=f'[INFO] Match and Copy {opt.img_folder_path}, {opt.anno_folder_path} to {opt.save_dir}')):
            corresponding_img_file_path = path.replace('.xml', '.jpg').replace(opt.anno_folder_path, opt.img_folder_path)
            # Match
            if os.path.exists(corresponding_img_file_path):
                # Rename
                if opt.rename:
                    file_name = str(idx)
                else:
                    file_name = get_file_prefix(path)
                # Copy
                shutil.copy(path, opt.anno_path + file_name + '.xml')
                shutil.copy(corresponding_img_file_path, opt.img_path + file_name + '.jpg')
            else:
                print(f"[INFO] {os.path.split(corresponding_img_file_path)[1]} not exist")
                continue
    assert len(os.listdir(opt.anno_path)) == len(os.listdir(opt.img_path)), f'XML file number {len(os.listdir(opt.anno_path))} not equal to image number {len(os.listdir(opt.img_path))}.'
    random.seed(opt.seed)
    if opt.split:
        split(opt)
    if opt.label:
        label(opt)
    if opt.plot:
        plot(opt)
