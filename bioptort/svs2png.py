import glob
import openslide
from tqdm import tqdm
import os

im_path = '/home/jackson/research/data/tortuosity_study/USCAP_prelim_dataset/*.svs'
out_path = '/home/jackson/research/data/tortuosity_study/USCAP_prelim_dataset/pngs/'

if not os.path.exists(out_path):
    os.makedirs(out_path)

files = sorted(glob.glob(im_path))

for fn_i, fn in tqdm(enumerate(files)):
    fname = fn.split('/')[-1]
    osh = openslide.OpenSlide(fn)
    print(f'Working on file: {fn}')
    print(f'Level dimensions: {osh.level_dimensions}')

    img = osh.read_region((0, 0), 2, osh.level_dimensions[2])
    basewidth = 1000
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth, hsize))

    img.save(out_path + fname.split('.')[0] + '.png')


