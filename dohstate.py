import torch, json, os
from PIL import Image
from xml.etree.ElementTree import parse as ET_parse

from torchvision.ops import box_area
from torchvision.datasets import VOCDetection
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

class Pre100DOH(VOCDetection):
    states = ['No', 'Self', 'Another', 'Portable', 'Stationary']
    states2binary = [0, 0, 0, 1, 1]
    binary_states = ['NoOBJ', 'WithOBJ']
    def __getitem__(self, index: int):
        assert self.transforms is None
        targets = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())
        targets = targets['annotation']['object']
        boxes, sides, states = [], [], []
        for t in targets:
            if t['name'] == 'hand':
                boxes.append([
                    float(t['bndbox']['xmin']),
                    float(t['bndbox']['ymin']),
                    float(t['bndbox']['xmax']),
                    float(t['bndbox']['ymax'])
                ])
                sides.append(int(t['handside']))
                states.append(self.states2binary[int(t['contactstate'])])
        image = Image.open(self.images[index])
        return dict(image=self.images[index], boxes=boxes, sides=sides, states=states, height=image.height, width=image.width)

class DOHState(torch.utils.data.Dataset):
    states = [''] + Pre100DOH.binary_states
    def __init__(self, annos, transforms=None) -> None:
        self.annos = json.load(open(annos))
        self.transforms = transforms
    
    def __getitem__(self, index):
        anno = self.annos[index]
        image = Image.open(anno['image']).convert("RGB")
        boxes = torch.tensor(anno['boxes'])
        # labels: 0 - no hand, 1 - left hand, no contact, 2 - left hand, contact, ...
        # when learning, it will transfer to 3 classes with binary ce
        # so clever!
        labels = torch.tensor(anno['sides'], dtype=torch.long) + 1#1 + torch.tensor(anno['sides'], dtype=torch.long) * 2 + torch.tensor(anno['states'], dtype=torch.long)
        area = box_area(boxes)
        select = area > 1 # remove small boxes
        targets = dict(
            boxes=boxes[select], 
            labels=labels[select],
            area=area[select],
            image_id=torch.tensor(index, dtype=torch.long),
        )
        if self.transforms is not None:
            image, targets = self.transforms(image, targets)
        # DOHState.visualize(image, boxes=targets['boxes'], labels=targets['labels'], output_file='v.jpg')
        return image, targets

    # only hand
    # def __getitem__(self, index):
    #     anno = self.annos[index]
    #     image = Image.open(anno['image']).convert("RGB")
    #     boxes = torch.tensor(anno['boxes'])
    #     labels = torch.tensor(anno['labels'], dtype=torch.long) + 1 
    #     labels[:] = 1 # all 1
    #     area = box_area(boxes)
    #     select = area > 1 # remove small boxes
    #     targets = dict(
    #         boxes=boxes[select], 
    #         labels=labels[select],
    #         area=area[select],
    #         image_id=torch.tensor(index, dtype=torch.long),
    #     )
    #     if self.transforms is not None:
    #         image, targets = self.transforms(image, targets)
    #     # DOHState.visualize(image, boxes=targets['boxes'], labels=targets['labels'], output_file='v.jpg')
    #     return image, targets

    def __len__(self,):
        return len(self.annos)

    def get_height_and_width(self, index):
        anno = self.annos[index]
        return anno['height'], anno['width']

    @staticmethod
    def visualize(image, boxes, labels, output_file):
        img = image * 255
        img = img.to(torch.uint8)
        img = draw_bounding_boxes(
            img, boxes=boxes,
            labels=[DOHState.states[l] for l in labels],
            font='times_b.ttf',
            width=4, font_size=30
        )
        img = to_pil_image(img)
        img.save(output_file)

def get_dohstate(root, image_set, transforms, mode="instances"):
    if image_set == "train":
        dataset = DOHState(os.path.join(root, "hi_100doh+ego_train.json"), transforms)
    else:
        assert image_set == "test"
        dataset = DOHState(os.path.join(root, "hi_100doh_test.json"), transforms)
    return dataset

def prepare():
    print('100doh trainval load & states -> binary ...')
    doh_trainval = Pre100DOH('100doh/pascal_voc_format', year='2007', image_set='trainval')
    doh_trainval_annos = [doh_trainval.__getitem__(i) for i in range(len(doh_trainval))]
    print('done!')

    print('ego trainval load & states -> binary ...')
    ego_trainval = Pre100DOH('100doh/pascal_voc_format_ego', year='2007', image_set='trainval')
    ego_trainval_annos = [ego_trainval.__getitem__(i) for i in range(len(ego_trainval))]
    print('done!')

    print('ego test load & states -> binary ...')
    ego_test = Pre100DOH('100doh/pascal_voc_format_ego', year='2007', image_set='test')
    ego_test_annos = [ego_test.__getitem__(i) for i in range(len(ego_test))]
    print('done!')
    
    print('save them as train split...')
    train_annos = doh_trainval_annos + ego_trainval_annos + ego_test_annos
    with open('100doh/hi_100doh+ego_train.json', 'w') as f:
        json.dump(train_annos, f)
    print('done!')

    print('doh test load & states -> binary ...')
    doh_test = Pre100DOH('100doh/pascal_voc_format', year='2007', image_set='test')
    print('done!')

    print('save them as test split...')
    test_annos = [doh_test.__getitem__(i) for i in range(len(doh_test))]
    with open('100doh/hi_100doh_test.json', 'w') as f:
        json.dump(test_annos, f)
    print('done!')

if __name__ == '__main__':
    prepare()