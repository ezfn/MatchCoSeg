from AB.data_utils import FileBasedEllipseImageDataset
from functools import partial
from AB import TrainerClass
from torch.utils.data import dataloader

class StringReplacer:
    def __init__(self, old='.jpg', new='_mask.png'):
        self.old = old
        self.new = new
    def __call__(self, s):
        return s.replace(self.old, self.new)

batch_size = 8

label_string_replacer = StringReplacer()
label_string_replacer = partial(str.replace, old='*.jpg', new='_mask.png')
dataset = FileBasedEllipseImageDataset(root_dir='/home/rd/Downloads',
                                       train_labels_file='/home/rd/Downloads/images/train_data.txt',
                                       label_string_replacer=StringReplacer(), do_normalize_image=True)


train_loader = dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
train_iter = iter(train_loader)
mod = 100
loss_sum = 0
trainer = TrainerClass.Trainer(train_dir = '/home/rd/Downloads/images/train_dir',
                             mobilenet_pre_trained_weights='/Umedia/SWDEV/CoSegMatching/AB/mobilenet/pretrained/mobilenetv2_1.0-0c6065bc.pth')
global_ctr = 0
while True:
    ctr = 0
    loss_sum = 0
    train_iter = iter(train_loader)
    for k, sample in enumerate(train_iter):
        sample
        loss = trainer.train_sample(sample)
        ctr += 1
        loss_sum += loss.item()
        print(loss_sum/ctr)
    global_ctr += ctr
    # trainer.save_model(global_ctr)
    # if not (ctr % mod):
    #     writer.add_scalar('data/loss', loss_sum / mod, ctr)
    #     print(loss_sum / mod)
    #     loss_sum = 0
    # # if not (ctr % (10 * mod)):
    # #     trainer.save_model(ctr)
    # writer.export_scalars_to_json(os.path.join(log_dir, "all_scalars.json"))

