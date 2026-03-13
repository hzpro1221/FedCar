## Introduce
1. This folder contains dataloader for all datasets (the path are set up with assuming that you are currently in FedCar folder). All datasets share the same label space and index label.
2. To perform make each of these datasets as domain, their label space should be the same. In this work, we fix the label as bellow (other label will be convert as 255)  
    - 0: 'road'.
    - 1: 'sidewalk'.
    - 2: 'building'
    - 3: 'wall'
    - 4: 'fence'
    - 5: 'pole'
    - 6: 'traffic light'
    - 7: 'traffic sign'
    - 8: 'vegetation'
    - 9: 'terrain'
    - 10: 'sky'
    - 11: 'person'
    - 12: 'rider'
    - 13: 'car'
    - 14: 'truck'
    - 15: 'bus'
    - 16: 'train'
    - 17: 'motorcycle'
    - 18: 'bicycle'
    - 255: 'void / ignore' 
