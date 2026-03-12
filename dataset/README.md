## To download GTA5 dataset, do following step:
1. Change directory to `FedCar/dataset` folder.
2. Run `python gta5.py --dest_dir gta5 --urls "https://download.visinf.tu-darmstadt.de/data/from_games/data/01_images.zip,https://download.visinf.tu-darmstadt.de/data/from_games/data/01_labels.zip,https://download.visinf.tu-darmstadt.de/data/from_games/data/02_images.zip,https://download.visinf.tu-darmstadt.de/data/from_games/data/02_labels.zip,https://download.visinf.tu-darmstadt.de/data/from_games/data/03_images.zip,https://download.visinf.tu-darmstadt.de/data/from_games/data/03_labels.zip,https://download.visinf.tu-darmstadt.de/data/from_games/data/04_images.zip,https://download.visinf.tu-darmstadt.de/data/from_games/data/04_labels.zip,https://download.visinf.tu-darmstadt.de/data/from_games/data/05_images.zip,https://download.visinf.tu-darmstadt.de/data/from_games/data/05_labels.zip,https://download.visinf.tu-darmstadt.de/data/from_games/data/06_images.zip,https://download.visinf.tu-darmstadt.de/data/from_games/data/06_labels.zip,https://download.visinf.tu-darmstadt.de/data/from_games/data/07_images.zip,https://download.visinf.tu-darmstadt.de/data/from_games/data/07_labels.zip,https://download.visinf.tu-darmstadt.de/data/from_games/data/08_images.zip,https://download.visinf.tu-darmstadt.de/data/from_games/data/08_labels.zip,https://download.visinf.tu-darmstadt.de/data/from_games/data/09_images.zip,https://download.visinf.tu-darmstadt.de/data/from_games/data/09_labels.zip,https://download.visinf.tu-darmstadt.de/data/from_games/data/10_images.zip,https://download.visinf.tu-darmstadt.de/data/from_games/data/10_labels.zip"`. NOTE: the url order should be: Part1_Img, Part1_Lbl, Part2_Img, Part2_Lbl...

## To download SYNTHIA dataset, run this script on terminal:
```
cd path-to-your-data-folder/
mkdir synthia
cd synthia
wget --no-check-certificate http://synthia-dataset.cvc.uab.cat/SYNTHIA_RAND_CITYSCAPES.rar
unrar x SYNTHIA_RAND_CITYSCAPES.rar
```

## To download cityscape dataset, do following steps:
1. Change directory to `FedCar/dataset` folder.
2. Run `python cityscape.py --dest_dir cityscape --package taret/package` to download data. The two packages you need are `gtFine_trainvaltest.zip` for label, and `leftImg8bit_trainvaltest.zip` for input image data.

DEST_DIR = '/content'
IMAGES_URL = "http://128.32.162.150/bdd100k/bdd100k_images_10k.zip"
LABELS_URL = "http://128.32.162.150/bdd100k/bdd100k_seg_maps.zip"

## To download bdd100 semantic segmentation dataset, do following step:
1. Change directory to `FedCar/dataset` folder. 
2. Run  `python bdd100.py --dest_dir bdd100 --images_url image/url/path --labels_url label/url/path` to download, with `http://128.32.162.150/bdd100k/bdd100k_images_10k.zip` as images_url and `http://128.32.162.150/bdd100k/bdd100k_seg_maps.zip` as labels_url.

## To download mapillary dataset, do following step:
1. Change directory to `FedCar/dataset` folder. 
2. Run `python mapillary.py --dest_dir mapillary --zip_name "mapillary_vistas.zip" --download_url "https://scontent.fhan19-1.fna.fbcdn.net/m1/v/t6/An_o5cmHOsS1VbLdaKx_zfMdi0No5LUpL2htRxMwCjY_bophtOkM0-6yTKB2T2sa0yo1oP086sqiaCjmNEw5d_pofWyaE9LysYJagH8yXw_GZPzK2wfiQ9u4uAKrVcEIrkJiVuTn7JBumrA.zip?_nc_gid=mIOC--ibeXIi-SUIaALR9g&_nc_oc=Adn4siwhw2Aco0wutdQOatd1qPL5veOFSb_52oaGXfPyRvRD_BjO5cmT_46gdHkrXgc&ccb=10-5&oh=00_Afz9d7Iiu6mnz-HjIKdzCD3LGg5mBh38TJ0zpwcqUE9gMw&oe=69D9E368&_nc_sid=6de079"`
