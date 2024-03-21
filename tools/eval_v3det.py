from pycocotools.coco import COCO
from mmdet.datasets.api_wrappers.cocoeval_mp import COCOevalMP


v3det_gt = COCO('data/V3Det/annotations/v3det_2023_v1_val.json')  # gt annotation file
v3det_dt = v3det_gt.loadRes('work_dirs/v3det/val.bbox.json')  # coco-format det results
v3det_eval = COCOevalMP(v3det_gt, v3det_dt, 'bbox', num_proc=8)
v3det_eval.params.maxDets = [300]

v3det_eval.evaluate()
v3det_eval.accumulate()
v3det_eval.summarize()

