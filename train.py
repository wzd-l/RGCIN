from ultralytics import YOLO
import os
os.environ['WANDB_MODE'] = 'disabled'

model = YOLO("cfg/yolov8s.yaml")

results = model.train(data="/data.yaml", 
                      batch=32, 
                      workers=8,
                      device=1,
                      resume=True,
                      epochs=400,
                      imgsz=640,
                      )

