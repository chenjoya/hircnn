from dataclasses import dataclass
import cv2, torch, os
import numpy as np
from PIL import ImageFont
from hircnn import HandStateRCNN, visualize

# data
inputs = 'inputs/C0450.mp4'
outputs = 'outputs/C0450.mp4'
outjpgs_folder = f'{outputs}_jpgs/'
os.makedirs(outjpgs_folder, exist_ok=True)
# smooth tracker
fps = 5
window_size = 3
interaction_threshold = 0.9
# weights
weights = 'outputs/ms_bs2x16_lr1e-2_12e_syncbn_amp/model_11.pth'
box_score_thresh = 0.5
font = ImageFont.truetype('times_b.ttf', size=30)

# define network
hircnn = HandStateRCNN(
    weights=weights, 
    box_score_thresh=box_score_thresh, 
    box_detections_per_img=2
).cuda()
hircnn.eval()

# define data stream
cap = cv2.VideoCapture(inputs)
writer = cv2.VideoWriter(
    outputs, 
    cv2.VideoWriter_fourcc(*'mp4v'),
    # fps, (w, h) 
    fps, (int(cap.get(3)), int(cap.get(4)))
)
interval = round(cap.get(5) / fps)

# a clipseg pipeline to detect 2-hand interaction in video
class SmoothTracker:
    def __init__(self, window_size, interaction_threshold):
        self.window_size = window_size
        self.interaction_threshold = interaction_threshold
        self._queue = []
        self._smoothed = None

    def _inqueue(self, frame):
        self._queue.append(frame)
        self._queue = self._queue[-self.window_size:]

    def _smooth(self, ):
        # process intermediate frame
        if len(self._queue) < self.window_size:
            return
        frame = self._queue[window_size // 2]

        # states = torch.stack([q.state for q in self._queue])
        # scores = torch.stack([q.score for q in self._queue])
        # weights = torch.ones(window_size, device='cuda')

        # weighted_scores = scores * weights
        # interaction_score = weighted_scores[states].sum() - weighted_scores[~states].sum()

        # if frame.states.any() and frame.scores[frame.states].max() > self.interaction_threshold:
        self._smoothed = frame 
        # else:
        #     self._smoothed = None
        
    def track(self, frame):
        self._inqueue(frame)
        self._smooth()
    
    @property
    def last_smoothed_frame(self):
        return self._smoothed

@dataclass
class Frame:
    image: np.ndarray
    boxes: torch.Tensor
    states: torch.Tensor
    scores: torch.Tensor

@dataclass
class Clip:
    frames: list

fidx = -1
smooth_tracker = SmoothTracker(window_size, interaction_threshold)
ret, image = cap.read()
while ret:
    fidx += 1
    if fidx % interval:
        continue
    print(f'process in {fps}fps, now {fidx}-th frame...')
    # detect
    boxes, states, scores = hircnn(image)
    # smooth frame by tracking state
    frame = Frame(image, boxes, states, scores)
    smooth_tracker.track(frame)
    # save newest smoothed frame 
    if smooth_tracker.last_smoothed_frame is not None:
        frame = smooth_tracker.last_smoothed_frame
        vis = visualize(frame.image, frame.boxes, frame.states, frame.scores, font)
        writer.write(vis)
        cv2.imwrite(f'{outjpgs_folder}/{fidx}.jpg', vis)
    ret, image = cap.read() 
writer.release()
cap.release()