import model
import offlinedata

import train



ai = model.load_model('klingon_v1')


batch_size = 64
valid_batch_size = 64
train_gen = offlinedata.get_data_generator(offlinedata.df, offlinedata.train_idx, for_training=True, batch_size=batch_size)
valid_gen = offlinedata.get_data_generator(offlinedata.df, offlinedata.valid_idx, for_training=True, batch_size=valid_batch_size)

train.train_ai(ai, train_gen, len(offlinedata.train_idx)//batch_size, valid_gen, len(offlinedata.valid_idx)//valid_batch_size)

