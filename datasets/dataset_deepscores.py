from .base import BaseDataset
import os
import cv2
import numpy as np
import random
from obb_anns import OBBAnns
from scipy.spatial import distance as dist

class Deepscores(BaseDataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=None):
        super(Deepscores, self).__init__(data_dir, phase, input_h, input_w, down_ratio)
        self.category = ['beam',
                         #'stem',
                         'noteheadBlackInSpace',
                         'noteheadBlackOnLine',
                         'noteheadHalfInSpace',
                         'noteheadHalfOnLine',
                         'noteheadWholeInSpace',
                         'noteheadWholeOnLine',
                         'noteheadDoubleWholeInSpace',
                         'noteheadDoubleWholeOnLine',
                         #'flag8thUp',
                         #'flag8thDown',
                         #'flag16thUp',
                         #'flag16thDown',
                         #'flag32ndUp',
                         #'flag32ndDown',
                         #'flag64thUp',
                         #'flag64thDown',
                         #'flag128thUp',
                         #'flag128thDown',
                         'slur'
                         #'augmentationDot'
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
                           (80,208,146)
                           #(0,0,204),
                           #(17,90,197),
                           #(0,255,255),
                           #(102,255,102),
                           #(255,255,0),
                           #(101, 87, 193),
                           #(116, 208, 0),
                           #(166, 231, 11),
                           #(53, 121, 190),
                           #(174, 212, 32),
                           #(191, 105, 32)
                           ]
        self.num_classes = len(self.category)
        self.ds_cat_ids =  {'beam': 122,
                         #'stem': 42,
                         'noteheadBlackInSpace': 27,
                         'noteheadBlackOnLine': 25,
                         'noteheadHalfInSpace': 31,
                         'noteheadHalfOnLine': 29,
                         'noteheadWholeInSpace': 35,
                         'noteheadWholeOnLine': 33,
                         'noteheadDoubleWholeInSpace': 39,
                         'noteheadDoubleWholeOnLine': 37,
                         #'flag8thUp': 48,
                         #'flag8thDown': 54,
                         #'flag16thUp': 50,
                         #'flag16thDown': 56,
                         #'flag32ndUp': 51,
                         #'flag32ndDown': 57,
                         #'flag64thUp': 52,
                         #'flag64thDown': 58,
                         #'flag128thUp': 53,
                         #'flag128thDown' : 59,
                         'slur': 121
                         #'augmentationDot': 41 
                        }
        self.cat_ids = {cat:i for i,cat in enumerate(self.category)}
        self.img_ids, self.anns, self.cats, self.img_id_dict = self.load_img_ids()
        self.image_path = os.path.join(data_dir, 'images')
        self.label_path = os.path.join(data_dir, 'labelTxt')

    def load_img_ids(self):
        image_lists = []
        image_id_dict = {}
        if self.phase == 'train' or self.phase == 'val' or self.phase == 'val2' or self.phase == 'train2':
            #image_set_index_file = os.path.join(self.data_dir, 'trainval.txt')
            train_o = OBBAnns('../ds2_dense_resize/deepscores_train.json')
            train_o.load_annotations('deepscores')
            cats = train_o.get_cats()
            img_idxs = [i for i in range(len(train_o.img_info))]
            #img_idxs = [i for i in range(len(train_o.img_info))]
            #random.seed(34)
            #random.shuffle(img_idxs)
            #img_idxs_train = img_idxs[:1212]
            img_idxs_train = img_idxs
            #img_idxs_val = img_idxs[1212:]

            if self.phase == 'train' or self.phase == 'train2':
                imgs, anns = train_o.get_img_ann_pair(idxs=img_idxs_train, ann_set_filter="deepscores")
            elif self.phase == 'val' or self.phase == 'val2':
                imgs, anns = train_o.get_img_ann_pair(idxs=img_idxs_val, ann_set_filter="deepscores")

            for img in anns:
                img_np = np.array(img)
                filename = train_o.get_imgs(ids=[int(img_np[0][4])])[0]['filename']
                image_lists.append(os.path.splitext(filename)[0])
                image_id_dict[os.path.splitext(filename)[0]] = int(img_np[0][4])
                #train_o.visualize(img_idx=0,out_dir='/home/jessicatawade/BBAVectors-Oriented-Object-Detection/', show=False)

        else:
            #test_o = OBBAnns('../ds2_dense_resize/deepscores_test.json')
            test_o = OBBAnns('../ds2_dense_resize/deepscores_train.json')
            test_o.load_annotations('deepscores')
            cats = test_o.get_cats()
            #img_idxs = [i for i in range(len(test_o.img_info))]
            #random.seed(34)
            #random.shuffle(img_idxs)
            #img_idxs = img_idxs[0:16]
            img_idxs = [i for i in range(16)]
            #img_idxs = [i for i in range(len(test_o.img_info))]
            #random.seed(34)
            #random.shuffle(img_idxs)
            #img_idxs = img_idxs[:16]

            imgs, anns = test_o.get_img_ann_pair(idxs=img_idxs, ann_set_filter="deepscores")

            for img in anns:
                img_np = np.array(img)
                filename = test_o.get_imgs(ids=[int(img_np[0][4])])[0]['filename']
                image_lists.append(os.path.splitext(filename)[0])
                image_id_dict[os.path.splitext(filename)[0]] = int(img_np[0][4])
        return image_lists, anns, cats, image_id_dict

    def load_image(self, index):
        img_id = self.img_ids[index]
        imgFile = os.path.join(self.image_path, img_id+'.png')
        assert os.path.exists(imgFile), 'image {} not existed'.format(imgFile)
        img = cv2.imread(imgFile)
        return img

    def load_annoFolder(self, img_id):
        return os.path.join(self.label_path, img_id+'.txt')

    def order_points(self, pts):
        xSorted = pts[np.argsort(pts[:, 0]), :]
        
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]

        return np.array([tl, tr, br, bl], dtype="float32")

    def load_annotation(self, index):
        image = self.load_image(index)
        h,w,c = image.shape
        valid_pts = []
        valid_ds_cat = []
        valid_cat = []
        
        count = 0
        img_np = np.array(self.anns[index])
        for object_instance in img_np:
            if object_instance[2][0] in self.ds_cat_ids.values():
                count += 1
                coord = np.array(object_instance[1], dtype=np.float32)
                coord_reshape = np.reshape(coord, (4,2))
                coord = self.order_points(coord_reshape)

                x1 = min(max(float(coord[0][0]/2), 0), w - 1)
                y1 = min(max(float(coord[0][1]/2), 0), h - 1)
                x2 = min(max(float(coord[1][0]/2), 0), w - 1)
                y2 = min(max(float(coord[1][1]/2), 0), h - 1)
                x3 = min(max(float(coord[2][0]/2), 0), w - 1)
                y3 = min(max(float(coord[2][1]/2), 0), h - 1)
                x4 = min(max(float(coord[3][0]/2), 0), w - 1)
                y4 = min(max(float(coord[3][1]/2), 0), h - 1)

                #x1 = min(max(float(coord[0]/2), 0), w - 1)
                #y1 = min(max(float(coord[1]/2), 0), h - 1)
                #x2 = min(max(float(coord[2]/2), 0), w - 1)
                #y2 = min(max(float(coord[3]/2), 0), h - 1)
                #x3 = min(max(float(coord[4]/2), 0), w - 1)
                #y3 = min(max(float(coord[5]/2), 0), h - 1)
                #x4 = min(max(float(coord[6]/2), 0), w - 1)
                #y4 = min(max(float(coord[7]/2), 0), h - 1)
                # TODO: filter small instances
                xmin = max(min(x1, x2, x3, x4), 0)
                xmax = max(x1, x2, x3, x4)
                ymin = max(min(y1, y2, y3, y4), 0)
                ymax = max(y1, y2, y3, y4)
                #if ((xmax - xmin) > 10) and ((ymax - ymin) > 10):
                valid_pts.append([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
                valid_ds_cat.append(object_instance[2][0])
                valid_cat.append(self.cat_ids[self.cats[object_instance[2][0]]['name']])

        #print('COUNT: ', count)

        annotation = {}
        annotation['pts'] = np.asarray(valid_pts, np.float32)
        annotation['ds_cat'] = np.asarray(valid_ds_cat, np.int32)
        annotation['cat'] = np.asarray(valid_cat, np.int32)
        
        pts0 = np.asarray(valid_pts, np.float32)
        img = self.load_image(index)
        for i in range(pts0.shape[0]):
            pt = pts0[i, :, :]
            tl = pt[0, :]
            tr = pt[1, :]
            br = pt[2, :]
            bl = pt[3, :]
            cv2.line(img, (int(tl[0]), int(tl[1])), (int(tr[0]), int(tr[1])), (0, 0, 255), 1, 1)
            cv2.line(img, (int(tr[0]), int(tr[1])), (int(br[0]), int(br[1])), (255, 0, 255), 1, 1)
            cv2.line(img, (int(br[0]), int(br[1])), (int(bl[0]), int(bl[1])), (0, 255, 255), 1, 1)
            cv2.line(img, (int(bl[0]), int(bl[1])), (int(tl[0]), int(tl[1])), (255, 0, 0), 1, 1)
            cv2.putText(img, '{}'.format(self.category[valid_cat[i]]), (int(tl[0]), int(tl[1])), cv2.FONT_HERSHEY_TRIPLEX, 0.6,
                        (0, 0, 255), 1, 1)
        #cv2.imshow('img', np.uint8(img))
        cv2.imwrite('pretrain-downsized.png', np.uint8(img)) 
        k = cv2.waitKey(0) & 0xFF
        if k == ord('q'):
            cv2.destroyAllWindows()
            exit()

        return annotation

    def dec_evaluation(self, result_path):


        o = OBBAnns('../ds2_dense_resize/deepscores_train.json')
        o.load_annotations("deepscores")

        o.load_proposals(result_path+'/proposals.json')
        metric_results = o.calculate_metrics(classwise=True)

        cat_id_to_name = {y:x for x,y in self.ds_cat_ids.items()}

        classaps = []
        map = 0

        for cls_key, avg_dict in metric_results.items():
            classname = cat_id_to_name[int(cls_key)]
            if classname == 'background':
                continue
            print('classname:', classname)

            rec = avg_dict['recall']
            prec = avg_dict['precision']
            ap = avg_dict['accuracy']

            map = map + ap
            # print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
            print('{}:{} '.format(classname, ap*100))
            classaps.append(ap)
            # umcomment to show p-r curve of each category
            # plt.figure(figsize=(8,4))
            # plt.xlabel('recall')
            # plt.ylabel('precision')
            # plt.plot(rec, prec)
        # plt.show()
        map = map / len(self.category)
        print('map:', map*100)
        # classaps = 100 * np.array(classaps)
        # print('classaps: ', classaps)
        return map

