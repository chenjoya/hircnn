import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T
from torchvision.models import detection

class HandStateRCNN(torch.nn.Module):
    def __init__(self, 
        model='fasterrcnn_resnet50_fpn_v2', 
        weights='hsrcnn/outputs/ms_bs2x16_lr1e-2_12e_syncbn_amp/model_11.pth',
        box_score_thresh=0.5,
        box_detections_per_img=2):
        super().__init__()
        model = getattr(detection, model)(
            num_classes=3, box_score_thresh=box_score_thresh,
            box_detections_per_img=box_detections_per_img)
        model.load_state_dict(torch.load(weights, map_location="cpu")["model"])
        self.transforms = T.Compose(
            [
                T.PILToTensor(),
                T.ConvertImageDtype(torch.float),
            ]
        )
        self.model = model

    def __call__(self, bgr_frame):
        x = Image.fromarray(bgr_frame[:,:,::-1])
        x = self.transforms(x)
        with torch.no_grad():
            y = self.model([x.cuda()])[0]
        boxes, states, scores = y['boxes'], y['labels'] - 1, y['scores'] # 0-N, 1-P
        return boxes, states, scores
    
    @staticmethod
    def state2str(state):
        return 'WithOBJ' if state == 1 else 'NoOBJ'

def draw_box_masks(image, boxes, labels, colors, font):
    mask = Image.new('RGBA', (image.width, image.height))
    pmask = ImageDraw.Draw(mask)
    draw = ImageDraw.Draw(image)

    for box, label, color in zip(boxes, labels, colors):
        pmask.rectangle(box, outline=color, width=4, fill=(*color,70))
        extend = 210 if 'With' in label else 180
        draw.rectangle([box[0], max(0, box[1]-30), box[0]+extend, max(0, box[1]-30)+30], fill=(255, 255, 255), outline=color, width=4)
        draw.text((box[0]+6, max(0, box[1]-30)-2), label, font=font, fill=color) # 

    image.paste(mask, (0,0), mask)
    return image

def visualize(image, boxes, states, scores, font):
    labels = [f'{HandStateRCNN.state2str(state)}: {score.item():.2f}'for state, score in zip(states, scores)]
    colors = [(0, 90, 181) if state == 0 else (220, 50, 32) for state in states]
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return draw_box_masks(image, boxes.round().int().tolist(), labels, colors, font)

if __name__ == '__main__':
    hsrcnn = HandStateRCNN(weights='outputs/ms_bs2x16_lr1e-2_12e_syncbn_amp/model_11.pth', box_score_thresh=0.7, box_detections_per_img=10).cuda()
    hsrcnn.eval()
    font = ImageFont.truetype('times_b.ttf', size=30)
    import cv2, json
    import numpy as np
    # test_annos = json.load(open('100doh/hs_100doh_test.json'))
    # for test_anno in test_annos:
    #     image = cv2.imread(test_anno['image'])
    #     boxes, states, scores = hsrcnn(image)
    #     visualize(image, boxes, states, scores, font)
    #     pass
    
    cap = cv2.VideoCapture('microwave.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter("microwave_result.mp4", fourcc, round(cap.get(5)), (int(cap.get(3)), int(cap.get(4))))
    ret, frame = cap.read()
    while ret:
        # print()
        boxes, states, scores = hsrcnn(frame)
        vis = visualize(frame, boxes, states, scores, font)
        writer.write(cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR))
        ret, frame = cap.read() 
    writer.release()
    cap.release()