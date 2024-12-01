# UrbanSound8K Coding Sample
My Private repo for storing code for Urban8k Audio Classification Example. 

# Instructions to run code on Ubuntu 22.04 (Either WSL2 or native Ubuntu)
1. Run `pip install -r requirements.txt` to initialize the environment.
2. Download the UrbanSound8k Dataset (linked here: https://urbansounddataset.weebly.com/urbansound8k.html)
3. Update data paths in main.py to reflect location of UrbanSound8k dataset
4. Run `python3 main.py`
5. To monitor training, run in a separate terminal, navigate to the parent directory of this project, and run `tensorboard --logdir ./logs`
