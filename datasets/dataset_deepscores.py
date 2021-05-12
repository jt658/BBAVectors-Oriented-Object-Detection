from .base import BaseDataset
import os
import cv2
import numpy as np
from obb_anns import OBBAnns

class Deepscores(BaseDataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None):
        super(Deepscores, self).__init__(data_dir, phase, input_h, input_w, down_ratio)
        self.category = ['beam',
                         'stem',
                         'noteheadBlackInSpace',
                         'noteheadBlackOnLine',
                         'noteheadHalfInSpace',
                         'noteheadHalfOnLine',
                         'noteheadWholeInSpace',
                         'noteheadWholeOnLine',
                         'noteheadDoubleWholeInSpace',
                         'noteheadDoubleWholeOnLine',
                         'flag8thUp',
                         'flag8thDown',
                         'flag16thUp',
                         'flag16thDown',
                         'flag32ndUp',
                         'flag32ndDown',
                         'flag64thUp',
                         'flag64thDown',
                         'flag128thUp',
                         'flag128thDown',
                         'augmentationDot'
                         ]
        self.color_pans = [(204,78,210),
                           (0,192,255),
                           (0,131,0),
                           (240,176,0),
                           (254,100,38),
                           (0,0,255),
                           (182,117,46),
                           (185,60,129),
                           (204,153,255),
                           (80,208,146),
                           (0,0,204),
                           (17,90,197),
                           (0,255,255),
                           (102,255,102),
                           (255,255,0),
                           (101, 87, 193),
                           (116, 208, 0),
                           (166, 231, 11),
                           (53, 121, 190),
                           (174, 212, 32),
                           (191, 105, 32)]
        self.num_classes = len(self.category)
        self.cat_ids =  {'beam': 122,
                         'stem': 42,
                         'noteheadBlackInSpace': 27,
                         'noteheadBlackOnLine': 25,
                         'noteheadHalfInSpace': 31,
                         'noteheadHalfOnLine': 29,
                         'noteheadWholeInSpace': 35,
                         'noteheadWholeOnLine': 33,
                         'noteheadDoubleWholeInSpace': 39,
                         'noteheadDoubleWholeOnLine': 37,
                         'flag8thUp': 48,
                         'flag8thDown': 54,
                         'flag16thUp': 50,
                         'flag16thDown': 56,
                         'flag32ndUp': 51,
                         'flag32ndDown': 57,
                         'flag64thUp': 52,
                         'flag64thDown': 58,
                         'flag128thUp': 53,
                         'flag128thDown' : 59,
                         'augmentationDot': 41 
                        }
        self.img_ids, self.anns = self.load_img_ids()
        self.image_path = os.path.join(data_dir, 'images')
        self.label_path = os.path.join(data_dir, 'labelTxt')

    def load_img_ids(self):
        image_lists = []
        if self.phase == 'train':
            #image_set_index_file = os.path.join(self.data_dir, 'trainval.txt')
            train_o = OBBAnns('../ds2_dense/deepscores_train.json')
            train_o.load_annotations('deepscores')
            #img_idxs = [i for i in range(len(o.img_info))]
            img_idxs = [i for i in range(5)]
            imgs, anns = train_o.get_img_ann_pair(idxs=img_idxs, ann_set_filter="deepscores")

            for img in anns:
                img_np = np.array(img)
                filename = train_o.get_imgs(ids=[int(img_np[0][4])])[0]['filename']
                image_lists.append(os.path.splitext(filename)[0])

        else:
            #image_set_index_file = os.path.join(self.data_dir, 'trainval.txt')
            test_o = OBBAnns('../ds2_dense/deepscores_train.json')
            test_o.load_annotations('deepscores')
            #img_idxs = [i for i in range(len(o.img_info))]
            img_idxs = [i for i in range(5)]
            imgs, anns = test_o.get_img_ann_pair(idxs=img_idxs, ann_set_filter="deepscores")

            for img in anns:
                img_np = np.array(img)
                filename = test_o.get_imgs(ids=[int(img_np[0][4])])[0]['filename']
                image_lists.append(os.path.splitext(filename)[0])
        return image_lists, anns

    def load_image(self, index):
        img_id = self.img_ids[index]
        imgFile = os.path.join(self.image_path, img_id+'.png')
        assert os.path.exists(imgFile), 'image {} not existed'.format(imgFile)
        img = cv2.imread(imgFile)
        return img

    def load_annoFolder(self, img_id):
        return os.path.join(self.label_path, img_id+'.txt')

    def load_annotation(self, index):
        image = self.load_image(index)
        h,w,c = image.shape
        valid_pts = []
        valid_cat = []
        valid_dif = []

        img_np = np.array(anns[index])
        for object_instance in img_np:
            if object_instance[2][0] in self.cat_ids.values():
                coord = np.array(object_instance[1], dtype=np.float32)
                x1 = min(max(float(coord[0]), 0), w - 1)
                y1 = min(max(float(coord[1]), 0), h - 1)
                x2 = min(max(float(coord[2]), 0), w - 1)
                y2 = min(max(float(coord[3]), 0), h - 1)
                x3 = min(max(float(coord[4]), 0), w - 1)
                y3 = min(max(float(coord[5]), 0), h - 1)
                x4 = min(max(float(coord[6]), 0), w - 1)
                y4 = min(max(float(coord[7]), 0), h - 1)
                # TODO: filter small instances
                xmin = max(min(x1, x2, x3, x4), 0)
                xmax = max(x1, x2, x3, x4)
                ymin = max(min(y1, y2, y3, y4), 0)
                ymax = max(y1, y2, y3, y4)
                if ((xmax - xmin) > 10) and ((ymax - ymin) > 10):
                    valid_pts.append([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
                    valid_cat.append(object_instance[2][0])

        annotation = {}
        annotation['pts'] = np.asarray(valid_pts, np.float32)
        annotation['cat'] = np.asarray(valid_cat, np.int32)
        # pts0 = np.asarray(valid_pts, np.float32)
        # img = self.load_image(index)
        # for i in range(pts0.shape[0]):
        #     pt = pts0[i, :, :]
        #     tl = pt[0, :]
        #     tr = pt[1, :]
        #     br = pt[2, :]
        #     bl = pt[3, :]
        #     cv2.line(img, (int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])), (0, 0, 255), 1, 1)
        #     cv2.line(img, (int(tr[0]), int(tr[1])), (int(br[0]), int(br[1])), (255, 0, 255), 1, 1)
        #     cv2.line(img, (int(br[0]), int(br[1])), (int(bl[0]), int(bl[1])), (0, 255, 255), 1, 1)
        #     cv2.line(img, (int(bl[0]), int(bl[1])), (int(tl[0]), int(tl[1])), (255, 0, 0), 1, 1)
        #     cv2.putText(img, '{}:{}'.format(valid_dif[i], self.category[valid_cat[i]]), (int(tl[0]), int(tl[1])), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
        #                 (0, 0, 255), 1, 1)
        # cv2.imshow('img', np.uint8(img))
        # k = cv2.waitKey(0) & 0xFF
        # if k == ord('q'):
        #     cv2.destroyAllWindows()
        #     exit()
        return annotation
