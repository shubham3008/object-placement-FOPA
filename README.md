# object-placement (FOPA)

This is a repository for the IE643 project. The task is object placement. The contents include majority of the code files. The dataset, pretrained encoder weights, and final architecture weights are not pushed due to size restrictions. They will be made available on the drive.

For setting up the environment use the requirements.txt file

The dataset needs to go in the `./data/data/` directory (only required for training). This can be found [here](https://drive.google.com/file/d/1VBTCO3QT1hqzXre1wdWlndJR97SI650d/view).

The pretrained encoder also goes in the `./data/data/` directory. This can be found [here](https://drive.google.com/file/d/1DMCINPzrBsxXj_9fTKnzB7mQcd8WQi3T/view?usp=sharing).

Final architecture weights are to be placed in `./` directory. To be added to the drive.

Run this script for getting object placement and harmonization

```bash
python finalinference.py --bg BACKGROUND_PATH --fg FOREGROUND_PATH --mask FOREGROUND_MASK_PATH
```

The results are displayed in the **results** folder.

For shadow generation, first download the trained SGRNet model from [here](https://drive.google.com/drive/folders/16isd7fPUHW1uaW3oGniCYZqhVve5zCN1?usp=sharing).

Add this to the `./shadow-model/TrainedModels` folder. Now we go to the scripts directory.
```bash
cd ./shadow-model/src/script
```
Update the data related paths appropriately add run
```bash
chmod +x SGRNet_RealComposite_2.sh
./SGRNet_RealComposite_2.sh
```