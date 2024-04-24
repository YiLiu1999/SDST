import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import adjusted_mutual_info_score as ami_score
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import fowlkes_mallows_score as fmi_score
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import confusion_matrix
import opt
from scipy.optimize import linear_sum_assignment
import os
from torch.nn import functional as F
import matplotlib.colors as mcolors

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
device = torch.device("cuda:{}".format(opt.args.k) if torch.cuda.is_available() else "cpu")


class KMEANS:
    def __init__(self, n_clusters=10, max_iter=None, verbose=True, device=torch.device("cpu")):
        # self.n_cluster = n_clusters
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).to(device)
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit(self, x):
        # Randomly select the initial centroid, for faster convergence you can borrow the kmeans++ initialisation method from sklearn
        init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
        # print(init_row.shape)    # shape 10
        init_points = x[init_row]
        # print(init_points.shape) # shape (10, 2048)
        self.centers = init_points
        while True:
            # 聚类标记
            self.nearest_center(x)
            # 更新中心点
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, (0)))
            if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break

            self.count += 1

        return self.representative_sample()

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        # print(labels.shape)  # shape (250000)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        # print(dists.shape)   # shape (0, 10)
        for i, sample in enumerate(x):
            # print(self.centers.shape) # shape(10, 2048)
            # print(sample.shape)       # shape 2048
            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), (1))
            # print(dist.shape)         # shape 10
            labels[i] = torch.argmin(dist)
            # print(labels.shape)       # shape 250000
            # print(labels[:10])
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
            # print(dists.shape)        # shape (1,10)
            # print('*')
        self.labels = labels  # shape 250000
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists  # 250000, 10
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device)  # shape (0, 250000)
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]
            centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))  # 10, 2048
        self.centers = centers  # shape (10, 2048)

    def representative_sample(self):
        # It is more intuitive to find the sample closest to the centroid as a representative sample for clustering
        # print(self.dists.shape)
        self.representative_samples = torch.argmin(self.dists, 1)
        # print(self.representative_samples.shape)  # shape 250000
        # print('*')
        return self.representative_samples


def get_predict_label(gt, pre, k):
    h, w = gt.shape
    pred = pre.reshape(h, w)
    # Select tabs for evaluation
    # gt != -1，No prediction of context
    ind = torch.where(gt != k)
    true = []
    pre = []
    true.append(gt[ind])
    pre.append(pred[ind])

    return true[0], pre[0]


def kl(z, model):
    q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - model.cluster_centers, 2), 2) / 1.0)
    q = q.pow((1.0 + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()

    weight = q ** 2 / q.sum(0)
    p = (weight.t() / weight.sum(1)).t()

    return F.kl_div(q.log(), p, reduction='batchmean')


def klloss(cross, CNN_out, GCN_out, model):
    c_kl = kl(cross, model)
    C_kl = kl(CNN_out, model)
    G_kl = kl(GCN_out, model)

    kl_loss = c_kl + C_kl + G_kl

    return kl_loss


def squared_distance(X, Y=None):
    '''
    Calculates the pairwise distance between points in X and Y

    X:          n x d matrix
    Y:          m x d matrix
    W:          affinity -- if provided, we normalize the distance

    returns:    n x m matrix of all pairwise squared Euclidean distances
    '''
    if Y is None:
        Y = X
    X1 = torch.reshape(X, (1, X.shape[0], X.shape[-1]))
    Y1 = torch.reshape(Y, (Y.shape[0], 1, Y.shape[-1]))
    DXY = torch.sum(torch.square(X1 - Y1), dim=-1)
    return DXY


def Draw_Classification_Map1(acc, label, mapping, name: str, scale: float = 4.0, dpi: int = 400):
    colors = ["black", "yellow", "lightgreen", "indigo", "orange", "pink", "peru", "crimson", "aqua", "dodgerblue",
              "slategrey", "b", "red", "darkcyan", "grey", "olive", "green", "gold"]

    # Create a blank RGB image with the same shape as the label
    rgb_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)

    # 将颜色值赋给每个类别对应的像素
    for i, color in enumerate(colors):
        rgb = np.array(mcolors.to_rgb(color)) * 255
        rgb_label[label == i] = rgb.astype(np.uint8)

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.imshow(rgb_label)  # Display colour images
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)

    foo_fig = plt.gcf()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig('./view_Label/{}'.format(name) + '_{}'.format(acc) + '.png', format='png',
                    transparent=True, dpi=dpi,
                    pad_inches=0)


def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def gaussian_noised_feature(X, img):
    """
    add gaussian noise to the attribute matrix X
    Args:
        X: the attribute matrix
    Returns: the noised attribute matrix X_tilde
    """
    N = torch.Tensor(np.random.normal(1, 0.1, X.shape)).to(opt.args.device)
    X_tilde = X * N
    return X_tilde


def diffusion_adj(adj, mode="ppr", transport_rate=0.2):  # yong
    """
    graph diffusion
    :param adj: input adj matrix
    :param mode: the mode of graph diffusion
    :param transport_rate: the transport rate
    - personalized page rank
    -
    :return: the graph diffusion
    """
    # add the self_loop
    adj_tmp = adj + torch.eye(adj.shape[0])

    # calculate degree matrix and it's inverse matrix
    d = torch.diag(adj_tmp.sum(0))
    d_inv = torch.linalg.inv(d)
    sqrt_d_inv = torch.sqrt(d_inv)

    # calculate norm adj
    norm_adj = torch.matmul(torch.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)

    # calculate graph diffusion
    if mode == "ppr":
        diff_adj = transport_rate * torch.linalg.inv((torch.eye(d.shape[0]) - (1 - transport_rate) * norm_adj))

    return diff_adj


def load_pretrain_parameter(model):
    """
    load pretrained parameters
    Args:
        model: Dual Correlation Reduction Network
    Returns: models
    """
    pretrained_dict = torch.load('name.pth',
                                 map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def model_init(model, Q, X, y, A_norm):
    """
    load the pre-train models and calculate similarity and cluster centers
    Args:
        model: Dual Correlation Reduction Network
        X: input feature matrix
        y: input label
        A_norm: normalized adj
    Returns: embedding similarity matrix
    """
    # # load pre-train models
    # models = load_pretrain_parameter(models)

    # calculate embedding similarity
    with torch.no_grad():
        cross, CNN_out, GCN_out, DZ_Conv1, DZ_Conv2 = model(X, A_norm, X, A_norm)

    # calculate cluster centers
    acc, nmi, ari, ami, fmi, kappa, purity, centers, cluster_id, y_outbg, true_outbg, y_pred, mapping = clustering(
        cross, Q, y)
    print("Initial clustering results")
    print("PREDICT  ACC: {:.4f},".format(acc), "NMI: {:.4f},".format(nmi), "AMI: {:.4f},".format(ami),
          "ARI: {:.4f},".format(ari), "FMI: {:.4f}".format(fmi))
    return centers, cluster_id, true_outbg, y_pred, mapping


def target_distribution(Q):
    """
    calculate the target distribution (student-t distribution)
    Args:
        Q: the soft assignment distribution
    Returns: target distribution P
    """
    weight = Q ** 2 / Q.sum(0)
    P = (weight.t() / weight.sum(1)).t()
    return P


def clustering(Z, Q, gt):
    """
    clustering based on embedding
    Args:
        Z: the input embedding
        y: the ground truth

    Returns: acc, nmi, ari, f1, clustering centers
    """
    model0 = KMEANS(opt.args.n_clusters, max_iter=20, verbose=False, device=device)
    cluster_id0 = model0.fit(Z)

    cluster_id0 = cluster_id0.float().unsqueeze(1)  # 将其转换为一个列向量
    y_pred = torch.matmul(Q, cluster_id0)

    true_outbg, pre = get_predict_label(torch.from_numpy(gt), y_pred, -1)
    acc, nmi, ari, ami, fmi, y_outbg, kappa, purity, mapping = eva(true_outbg, pre,
                                                                   show_details=opt.args.show_training_details)

    return acc, nmi, ari, ami, fmi, kappa, purity, model0.centers, cluster_id0, y_outbg, true_outbg, y_pred, mapping


def cluster_acc(y_true, y_pred, num_classes):
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth
        y_pred: the clustering id
        num_classes: total number of classes in your dataset

    Returns: acc and f1-score
    """
    y_true = torch.tensor(y_true) - torch.min(torch.tensor(y_true))
    l1 = list(set(y_true.tolist()))
    num_class1 = len(l1)
    y_pred = torch.tensor(y_pred)
    l2 = list(set(y_pred.tolist()))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred.tolist()))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return

    cost = torch.zeros((num_class1, numclass2), dtype=torch.int32)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # 使用 SciPy 的 linear_sum_assignment 执行 Munkres 算法
    cost_np = cost.numpy()
    row_ind, col_ind = linear_sum_assignment(-cost_np)
    new_predict = torch.zeros(len(y_pred))

    mapping = {}  # 用于建立真实标签到预测标签的映射关系
    for i, c in enumerate(l1):
        c2 = l2[col_ind[i]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
        mapping[c2] = c

    acc = metrics.accuracy_score(y_true, new_predict)

    matrix = confusion_matrix(y_true, new_predict)
    # 选择每个簇中最大的数值
    max_cluster_values = np.max(matrix, axis=0)
    # 计算purity
    purity = np.sum(max_cluster_values) / np.sum(matrix)
    ka = kappa(y_true.cpu().numpy(), new_predict.cpu().numpy())
    nmi = nmi_score(y_true.cpu().numpy(), new_predict.cpu().numpy())
    ami = ami_score(y_true.cpu().numpy(), new_predict.cpu().numpy())
    ari = ari_score(y_true.cpu().numpy(), new_predict.cpu().numpy())
    fmi = fmi_score(y_true.cpu().numpy(), new_predict.cpu().numpy())
    return acc, new_predict, mapping, purity, ka, nmi, ari, ami, fmi


def eva(y_true, y_pred, show_details=True):
    """
    evaluate the clustering performance
    Args:
        y_true: the ground truth
        y_pred: the predicted label
        show_details: if print the details
    Returns: None
    """
    acc, y, mapping, purity, kappa, nmi, ari, ami, fmi = cluster_acc(y_true, y_pred, opt.args.n_clusters)

    if show_details:
        print(':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ami {:.4f}'.format(ami),
              ', ari {:.4f}'.format(ari),
              ', fmi {:.4f}'.format(fmi))
    return acc, nmi, ari, ami, fmi, y, kappa, purity, mapping


def squared_distance(X, Y=None):
    '''
    Calculates the pairwise distance between points in X and Y

    X:          n x d matrix
    Y:          m x d matrix
    W:          affinity -- if provided, we normalize the distance

    returns:    n x m matrix of all pairwise squared Euclidean distances
    '''
    if Y is None:
        Y = X
    X1 = torch.reshape(X, (1, X.shape[0], X.shape[-1]))
    Y1 = torch.reshape(Y, (Y.shape[0], 1, Y.shape[-1]))
    DXY = torch.sum(torch.square(X1 - Y1), dim=-1)
    # DXY = tf.norm(X1-Y1, ord='euclidean', axis=-1)

    return DXY


def produce_graph(x, y1, y2):
    fig, ax = plt.subplots()
    ax.plot(x, y1, color='blue', label='ACC')
    ax2 = ax.twinx()
    ax2.plot(x, y2, color='red', label='LOSS')
    ax.set_ylabel('ACC', color='blue')
    ax2.set_ylabel('LOSS', color='red')
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    y2_max = np.max(y2)
    y2_min = np.min(y2)
    ax2.set_ylim(y2_min, y2_max)
    plt.show()


def paua(predicted, gt, ori_gt):
    class_counts = np.bincount(gt, minlength=opt.args.n_clusters)
    true_positive = np.bincount(predicted[predicted == gt], minlength=opt.args.n_clusters)
    pa = true_positive / class_counts
    ua = true_positive / np.bincount(predicted, minlength=opt.args.n_clusters)
    if opt.args.name == 'indian':
        print('PA: Alfalfa:{:.5f} || Corn-notill:{:.5f} || Corn-mintill:{:.5f} || Corn:{:.5f} || Grass-pasture:{:.5f}'
              ' || Grass-trees:{:.5f} || Grass-pasture-mowed:{:.5f} || Hay-windrowed:{:.5f} || Oats:{:.5f} || '
              'Soybean-notill:{:.5f} || Soybean-mintill:{:.5f} || Soybean-clean:{:.5f} || Wheat:{:.5f} || Woods:{:.5f}'
              ' ||Buildings-Grass-Trees-Drivers:{:.5f} || Stone-Steel-Towers:{:.5f}'
              .format(pa[0], pa[1], pa[2], pa[3], pa[4], pa[5], pa[6], pa[7], pa[8], pa[9], pa[10], pa[11], pa[12],
                      pa[13], pa[14], pa[15]))
        print('UA: Alfalfa:{:.5f} || Corn-notill:{:.5f} || Corn-mintill:{:.5f} || Corn:{:.5f} || Grass-pasture:{:.5f}'
              ' || Grass-trees:{:.5f} || Grass-pasture-mowed:{:.5f} || Hay-windrowed:{:.5f} || Oats:{:.5f} || '
              'Soybean-notill:{:.5f} || Soybean-mintill:{:.5f} || Soybean-clean:{:.5f} || Wheat:{:.5f} || Woods:{:.5f}'
              ' ||Buildings-Grass-Trees-Drivers:{:.5f} || Stone-Steel-Towers:{:.5f}'
              .format(ua[0], ua[1], ua[2], ua[3], ua[4], ua[5], ua[6], ua[7], ua[8], ua[9], ua[10], ua[11], ua[12],
                      ua[13], ua[14], ua[15]))

    elif opt.args.name == 'Botswana':
        print('PA: Water:{:.5f} || Hippo grass:{:.5f} || FloodPlain grasses 1:{:.5f} || FloodPlain grasses 2:{:.5f} || '
              'Reeds:{:.5f} || Riparian:{:.5f} || Firescar:{:.5f} || Island interior:{:.5f} || Acacia woodlands:{:.5f}'
              ' || Acacia shrublands:{:.5f} || Acacia grasslands:{:.5f} || Short mopane:{:.5f} || Mixed mopane:{:.5f}'
              ' || Chalcedony:{:.5f}'
              .format(pa[0], pa[1], pa[2], pa[3], pa[4], pa[5], pa[6], pa[7], pa[8], pa[9], pa[10], pa[11], pa[12],
                      pa[13]))
        print('UA: Water:{:.5f} || Hippo grass:{:.5f} || FloodPlain grasses 1:{:.5f} || FloodPlain grasses 2:{:.5f} || '
              'Reeds:{:.5f} || Riparian:{:.5f} || Firescar:{:.5f} || Island interior:{:.5f} || Acacia woodlands:{:.5f}'
              ' || Acacia shrublands:{:.5f} || Acacia grasslands:{:.5f} || Short mopane:{:.5f} || Mixed mopane:{:.5f}'
              ' || Chalcedony:{:.5f}'
              .format(ua[0], ua[1], ua[2], ua[3], ua[4], ua[5], ua[6], ua[7], ua[8], ua[9], ua[10], ua[11], ua[12],
                      ua[13]))

    elif opt.args.name == 'salinas':
        print('PA: Brocoli_green_weeds_1:{:.5f} || Brocoli_green_weeds_2:{:.5f} || Fallow:{:.5f} || Fallow_rough_plow:'
              '{:.5f} || Fallow_smooth:{:.5f} || Stubble:{:.5f} || Celery:{:.5f} || Grapes_untrained:{:.5f} || '
              'Soil_vinyard_develop:{:.5f} || Corn_senesced_green_weeds:{:.5f} || Lettuce_romaine_4wk:{:.5f} || '
              'Lettuce_romaine_5wk:{:.5f} || Lettuce_romaine_6wk:{:.5f} || Lettuce_romaine_7wk:{:.5f} || '
              'Vinyard_untrained:{:.5f} || Vinyard_vertical_trellis:{:.5f}'
              .format(pa[0], pa[1], pa[2], pa[3], pa[4], pa[5], pa[6], pa[7], pa[8], pa[9], pa[10], pa[11], pa[12],
                      pa[13], pa[14], pa[15]))
        print('UA: Brocoli_green_weeds_1:{:.5f} || Brocoli_green_weeds_2:{:.5f} || Fallow:{:.5f} || Fallow_rough_plow:'
              '{:.5f} || Fallow_smooth:{:.5f} || Stubble:{:.5f} || Celery:{:.5f} || Grapes_untrained:{:.5f} || '
              'Soil_vinyard_develop:{:.5f} || Corn_senesced_green_weeds:{:.5f} || Lettuce_romaine_4wk:{:.5f} || '
              'Lettuce_romaine_5wk:{:.5f} || Lettuce_romaine_6wk:{:.5f} || Lettuce_romaine_7wk:{:.5f} || '
              'Vinyard_untrained:{:.5f} || Vinyard_vertical_trellis:{:.5f}'
              .format(ua[0], ua[1], ua[2], ua[3], ua[4], ua[5], ua[6], ua[7], ua[8], ua[9], ua[10], ua[11], ua[12],
                      ua[13], ua[14], ua[15]))
    elif opt.args.name == 'HoustonU':
        print('PA: Healthy grass:{:.5f} || Stressed grass:{:.5f} || Synthetic grass:{:.5f} || Tress:'
              '{:.5f} || Soil:{:.5f} || Water:{:.5f} || Residential:{:.5f} || Commercial:{:.5f} || '
              'Road:{:.5f} || Highway:{:.5f} || Railway:{:.5f} || '
              'Parking Lot1:{:.5f} || Parking Lot2:{:.5f} || Tennis Court:{:.5f} || '
              'Running Track:{:.5f}'
              .format(pa[0], pa[1], pa[2], pa[3], pa[4], pa[5], pa[6], pa[7], pa[8], pa[9], pa[10], pa[11], pa[12],
                      pa[13], pa[14]))
        print('UA: Healthy grass:{:.5f} || Stressed grass:{:.5f} || Synthetic grass:{:.5f} || Tress:'
              '{:.5f} || Soil:{:.5f} || Water:{:.5f} || Residential:{:.5f} || Commercial:{:.5f} || '
              'Road:{:.5f} || Highway:{:.5f} || Railway:{:.5f} || '
              'Parking Lot1:{:.5f} || Parking Lot2:{:.5f} || Tennis Court:{:.5f} || '
              'Running Track:{:.5f}'
              .format(ua[0], ua[1], ua[2], ua[3], ua[4], ua[5], ua[6], ua[7], ua[8], ua[9], ua[10], ua[11], ua[12],
                      ua[13], ua[14]))
