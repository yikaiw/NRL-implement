import numpy as np
batch_size_dis = 64  # batch size for discriminator
batch_size_gen = 64  # batch size for generator
lambda_dis = 1e-5  # l2 loss regulation factor for discriminator
lambda_gen = 1e-5  # l2 loss regulation factor for generator
n_sample_dis = 20  # sample num for generator
n_sample_gen = 20  # sample num for discriminator
update_ratio = 1    # updating ratio when choose the trees
save_steps = 10

lr_dis = 1e-4  # learning rate for discriminator
lr_gen = 1e-3  # learning rate for discriminator

max_epochs = 20  # outer loop number
max_epochs_gen = 30  # loop number for generator
max_epochs_dis = 30  # loop number for discriminator

gen_for_d_iters = 10  # iteration numbers for generate new data for discriminator
model_log = '../log/iteration/'

load_model = False  # if load the model for continual training
gen_update_iter = 200
window_size = 3
random_state = np.random.randint(0, 100000)
train_filename = 'data/tencent/train_edges.npy'
test_filename = 'data/tencent/test_edges.npy'
test_neg_filename = 'data/tencent/test_edges_false.npy'
emb_filenames = ['results/tencent/GraphGAN/embeds_gen.csv', 'results/tencent/GraphGAN/embeds_dis.csv']
result_filename = 'results/tencent/GraphGAN/result.csv'
embed_dim = 50
node_num = 5242
modes = ['dis', 'gen']