import os.path
import scipy.io as sio
from skimage.segmentation import slic
from Preprocessing import Processor
from calcu_graph import *
import opt
from train_SDST import train
from utils import setup_seed

p = Processor()
torch.cuda.empty_cache()
OA_ALL = []
AA_ALL = []
KPP_ALL = []
AVG_ALL = []
Train_Time_ALL = []
Test_Time_ALL = []
seed = 3030
dataset = {
    0: "indian",
    1: "paviau",
    2: "salinas",
    3: "Botswana",
    4: "HoustonU",
}
DATASET = dataset[3]
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
if DATASET == 'Botswana':
    seed = 472
    k = 17
    n_seg = 3000
    data_mat = sio.loadmat('./Dataset/Botswana.mat')
    data = data_mat['Botswana']
    gt_mat = sio.loadmat('./Dataset/Botswana_gt.mat')
    gt = gt_mat['Botswana_gt']
    dataset_name = "Botswana"  # data name
    class_count = 14  # class_num
elif DATASET == 'indian':
    k = 12
    n_seg = 1000
    seed = 346
    data_mat = sio.loadmat(
        './Dataset/Indian_pines_corrected.mat')
    data = data_mat['indian_pines_corrected']
    gt_mat = sio.loadmat('./Dataset/Indian_pines_gt.mat')
    gt = gt_mat['indian_pines_gt']
    dataset_name = "indian"  # data name
    class_count = 16  # class_num
elif DATASET == 'salinas':
    k = 7
    n_seg = 1000
    seed = 452
    data_mat = sio.loadmat(
        '.Dataset/Salinas.mat')
    data = data_mat['salinas_corrected']
    gt = data_mat['salinas_gt']
    dataset_name = "salinas"  # data_name
    class_count = 16  # calss_num
elif DATASET == 'HoustonU':
    k = 12
    n_seg = 1000
    seed = 458
    data_mat = sio.loadmat(
        '.Dataset/HoustonU.mat')
    data = data_mat['HoustonU']
    gt = data_mat['HoustonU_GT']
    dataset_name = "HoustonU"  # data_name
    class_count = 15  # calss_num
load_path = ".Dataset/" + dataset_name
ori_gt = gt
img = p.std_norm(data)

gt = p.label_transform(gt)
pca_bands = 3
# Three dimensions of raw hyperspectral data
height, width, bands = data.shape
gt_reshape = np.reshape(gt, [-1])

if DATASET == 'Botswana':
    read_data = img[:, :, [35, 15, 6]]  # indian [30, 33, 15]
elif DATASET == 'indian':
    read_data = img[:, :, [29, 19, 9]]
elif DATASET == 'salinas':
    read_data = img[:, :, [32, 21, 11]]
elif DATASET == 'PaviaC':
    read_data = img[:, :, [55, 41, 12]]
elif DATASET == 'HoustonU':
    read_data = img[:, :, [60, 28, 13]]
# plt.figure()
# plt.imshow(read_data)
# plt.show()

segments = slic(read_data, n_segments=20, compactness=1,
                convert2lab=True, sigma=1,
                min_size_factor=0.1, max_size_factor=2,
                slic_zero=False, start_label=1)
print("number of superpixel:", segments.max())

trans, prt_img0, bias = p.get_spectral_superpixel(img, segments)
if os.path.exists(load_path + "{}_{}_adj.npy".format(segments.max(), k)):
    print('Created Graph: True')
    Adj = np.load(load_path + "{}_{}_adj.npy".format(segments.max(), k), allow_pickle=True)
else:
    print('Creating Graph...')
    Adj = construct_graph(prt_img0, k)
    np.save(load_path + "{}_{}_adj.npy".format(segments.max(), k), Adj)

Q = trans.astype(np.float32)
opt.args.n_clusters = class_count
opt.args.n_input = img.shape[2]
opt.args.name = dataset_name
opt.args.n_samples = segments.max()
opt.args.data_dim = data.shape[-1]
opt.args.test = True
opt.args.height = img.shape[0]
opt.args.width = img.shape[1]

for curr_seed in range(3150, 3310):
    setup_seed(curr_seed)
    train(Q, gt, prt_img0, img, Adj, bias, opt.args.k, opt)
