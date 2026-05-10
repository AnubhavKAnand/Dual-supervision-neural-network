import glob
from train import Trainer # Assuming you saved the previous code as train.py

# 1. Add '/*.png' (or '/*.jpg') to the end of your path!
dataset_folder = "/Users/anubhavkishoreanand/mncproject/data/train/*.jpg" 

# 2. Grab all the file paths
my_dataset_paths = glob.glob(dataset_folder)

print(f"Found {len(my_dataset_paths)} images ready for training!")

# 3. Initialize and run!
if len(my_dataset_paths) > 0:
    trainer = Trainer(dataset_path=my_dataset_paths)
    trainer.train(my_dataset_paths)
else:
    print("Check your folder path, no images were found. Make sure the file extension (.png or .jpg) matches your actual images!")