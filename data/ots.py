import os
import glob
from data import srdata

# Outdoor Training Set (OTS) of RESIDE
class OTS(srdata.SRData):
    def __init__(self, args, train=True):
        super(OTS, self).__init__(args, train)
        self.args = args
        self.repeat = 1
        self.idx_scale = 0

    def _scan(self):
        lq_apath = os.path.join(self.apath, 'hazy')

        lq_filelist = sorted(
            glob.glob(os.path.join(lq_apath, '*' + self.ext))
        )
        lq_filelist = [f for f in lq_filelist if os.path.getsize(f) > 0]

        return lq_filelist

    def _set_filesystem(self, dir_data):
        self.apath = dir_data + '/' + self.args.data_train  + '/train_outdoor'
        self.ext = '.jpg'


    def __len__(self):
        if self.train: 
            return len(self.images_lq) * self.repeat
        else:
            return len(self.images_lq)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_lq)
        else:
            return idx

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

