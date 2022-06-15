import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import tqdm

opt = TestOptions().parse()
opt.RED = False
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
if opt.split != "":
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s_%s' % (opt.phase, opt.split, opt.which_epoch))
else:
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test

count = 0
for i, data in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    # print('%04d: process image... %s' % (i, img_path))
    visualizer.save_images(webpage, visuals, img_path)
    count+=1

print(f'Spent time: ', model.total_spent / count)
print('MACs: %.3fG' % (model.macs / 1e9))

webpage.save()
