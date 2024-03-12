
import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
import config


#--- read options ---#
opt = config.read_arguments(train=False)

#--- create dataloader ---#
_, dataloader_val = dataloaders.get_dataloaders(opt)

#--- create utils ---#
image_saver = utils.results_saver(opt)

#--- create models ---#
model = models.OASIS_model(opt)
model = models.put_on_multi_gpus(model, opt)
model.eval()

#--- iterate over validation set ---#
for i, data_i in enumerate(dataloader_val):
    image, rendered, label = models.preprocess_input(opt, data_i)
    generated = model(None, rendered, label, "generate", None)
    image_saver(label, generated, data_i["rendered_name"])
