from flowNet.data_utils import MatlabBasedCosegDataset
from torch.utils.data import dataloader
from flowNet import trainerClass
from tensorboardX import SummaryWriter
import os
log_dir = '/media/fastData/coSegTraining/firstShot'
writer = SummaryWriter(log_dir=log_dir)


dataset = MatlabBasedCosegDataset(root_dir='/media/erez/PassportRD1/SintelCleanPasses/framesAndOldPasses',
                                  example_format='*/rectifiedPatchesSIFT_128X128_withField_*.mat')

train_loader = dataloader.DataLoader(dataset, batch_size=1, shuffle=True)
train_iter = iter(train_loader)
trainer = trainerClass.Trainer()

ctr = 0
while True:
    for k,sample in enumerate(dataset):
        loss = trainer.train_sample(sample)
        ctr += 1
        writer.add_scalar('data/loss', loss, ctr)
        if not (ctr % 10):
            print(loss.item())
writer.export_scalars_to_json(os.path.join(log_dir,"all_scalars.json"))
writer.close()

# for k, sample in enumerate(train_iter):
#     loss = trainer.train_sample(sample)
#     if not(k % 100):
#         print(loss.item())