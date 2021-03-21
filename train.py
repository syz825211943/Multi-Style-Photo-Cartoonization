import time
from options.train_options import TrainOptions
from data.data_loader import DataLoader
from models.cartoongan_model import CartoonGANModel
from util.visualizer import Visualizer


opt = TrainOptions().parse()  # parse the training options
dataset = DataLoader(opt)  # load the dataset
print('# training images = %d' % len(dataset))

model = CartoonGANModel(opt)
visualizer = Visualizer(opt)
total_steps = 0  # remember the total number of steps
   
# the process of training
for epoch in range(opt.which_epoch + 1, opt.niter + opt.niter_decay + 1):
    # update learning rate of initialization or training phase
    model.update_hyperparams(epoch)

    epoch_start_time = time.time()
    epoch_iter = 0
    total_time = 0

    if opt.init and epoch < 20: # flag of inititialization phase
        init = opt.init
    else:
        init = False

    for i, data in enumerate(dataset):  # get the data as input {'A':image, 'B":image}
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters(init=init)
        total_time = total_time + time.time() - iter_start_time
    
        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

