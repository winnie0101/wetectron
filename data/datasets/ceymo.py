import os
import pickle

import torch
import torch.utils.data
from PIL import Image
import xml.etree.ElementTree as ET

from utils.bounding_box import BoxList
from utils.boxlist_ops import remove_small_boxes
from .coco import unique_boxes

class CeyMoDataset(torch.utils.data.Dataset):

    CLASSES = (
        # "__background__ ",
        "no jb", # no Junction Box
        "jb" # Junction Box
    )

    def __init__(self, data_dir, split, use_difficult=False, transforms=None, proposal_file=None):
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms

        self._annopath = os.path.join(self.root, "bbox_annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "images", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "%s.txt")

        self.ids = []
        with open(self._imgsetpath % self.image_set) as f:
            file_list = f.readlines()
        for file_name in file_list:
            file_name = file_name.split(".jpg")[0].strip()
            self.ids.append(file_name)
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = CeyMoDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))
        
        # Include proposals from a file
        if proposal_file is not None:
            print('Loading proposals from: {}'.format(proposal_file))
            with open(proposal_file, 'rb') as f:
                self.proposals = pickle.load(f, encoding='latin1')
            # self.id_field = 'indexes' if 'indexes' in self.proposals else 'ids'  # compat fix
            # _sort_proposals(self.proposals, self.id_field)
            self.top_k = -1
        else:
            self.proposals = None

    def get_origin_id(self, index):
        img_id = self.ids[index]
        return img_id

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")

        if not os.path.exists(self._annopath % (img_id.split(".")[0])):
            target = None
        else:
            target = self.get_groundtruth(index)
            target = target.clip_to_image(remove_empty=True)
                
        if self.proposals is not None:
            if '_' in self.ids[index] and self.image_set == "test" and "2012" in self.root :
                img_id = self.ids[index].split('_')[1]
            else:
                img_id = self.ids[index]
            roi_idx = img_id + ".jpg"
            rois = self.proposals[roi_idx]
            # rois = self.proposals['boxes'][roi_idx]

            # remove duplicate, clip, remove small boxes, and take top k
            keep = unique_boxes(rois)
            rois = rois[keep, :]
            # scores = scores[keep]
            rois = BoxList(torch.tensor(rois), img.size, mode="xyxy")
            rois = rois.clip_to_image(remove_empty=True)
            # TODO: deal with scores
            rois = remove_small_boxes(boxlist=rois, min_size=2)
            if self.top_k > 0:
                rois = rois[[range(self.top_k)]]
                # scores = scores[:self.top_k]
        else:
            rois = None

        if self.transforms is not None:
            img, target, rois = self.transforms(img, target, rois)

        return img, target, rois, index

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % (img_id.split(".")[0])).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []

        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue

            name = obj.find("name").text.lower().strip()
            if name == "jb": # Junction Box
                gt_classes.append(self.class_to_ind[name])
                bb = obj.find("bndbox")
                # Make pixel indexes 0-based
                # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
                box = [
                    bb.find("xmin").text,
                    bb.find("ymin").text,
                    bb.find("xmax").text,
                    bb.find("ymax").text,
                ]
                bndbox = tuple(map(int, box))

                boxes.append(bndbox)

            difficult_boxes.append(difficult)

        if len(boxes) == 0: # if this image has no Junction Box
            gt_classes.append(0)
            box=[0, 0, 0, 0]
            bndbox = tuple(map(int, box))
            boxes.append(bndbox)
            difficult_boxes.append(0)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        file_name = "images/%s.jpg" % img_id
        if os.path.exists(self._annopath % img_id):
            anno = ET.parse(self._annopath % img_id).getroot()
            size = anno.find("size")
            im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
            return {"height": im_info[0], "width": im_info[1], "file_name": file_name}
        else:
            name = os.path.join(self.root, file_name)
            img = Image.open(name).convert("RGB")
            return  {"height": img.size[1], "width": img.size[0], "file_name": file_name}
        
    def map_class_id_to_class_name(self, class_id):
        return CeyMoDataset.CLASSES[class_id]
