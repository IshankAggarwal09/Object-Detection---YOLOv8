EPOCHS = 50               
MOSAIC = 0.1              
OPTIMIZER = 'AdamW'       
MOMENTUM = 0.2            
LR0 = 0.0005              
LRF = 0.001               
SINGLE_CLS = False        
BATCH_SIZE = 1            
IMG_SIZE = 416            
WORKERS = 2               

import argparse
from ultralytics import YOLO
import os

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation probability')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer')
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum')
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate multiplier')
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS, help='Single class training')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--imgsz', type=int, default=IMG_SIZE, help='Image size for training')
    parser.add_argument('--workers', type=int, default=WORKERS, help='Number of dataloader workers')
    
    args = parser.parse_args()

    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)

    model = YOLO(os.path.join(this_dir, "yolov8n.pt"))

    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"), 
        epochs=args.epochs,
        device=0,                   
        single_cls=args.single_cls, 
        mosaic=args.mosaic,
        optimizer=args.optimizer, 
        lr0=args.lr0, 
        lrf=args.lrf, 
        momentum=args.momentum,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.1,
        fliplr=0.5,
        erasing=0.4,
        copy_paste=0.1,
        patience=20,          
        save_period=-1        
    )
