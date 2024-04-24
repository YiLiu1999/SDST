import numpy as np
import torch
from sklearn import preprocessing
import spectral as spy
from sklearn.decomposition import PCA
import scipy.io as sio
from torchvision import transforms


class Processor:
    def __init__(self):
        pass

    def prepare_data(self, data_path, img_name, gt_name):
        data = sio.loadmat(data_path)
        data_keys = data.keys()
        img = data[img_name]
        gt = data[gt_name]
        return img, gt

    def Stand(self, data: np.array):
        """
        :param data:
        :return: Stand data
        """
        height, width, bands = data.shape
        data = np.reshape(data, [height * width, bands])
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        data = np.reshape(data, [height, width, bands])
        return data

    def std_norm(self, image):  # input tensor image size with CxHxW
        image = image.astype(np.float32)
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(torch.tensor(image).mean(dim=[0, 1]), torch.tensor(image).std(dim=[0, 1]))
        ])  # (x - mean(x))/std(x) normalize to mean: 0, std: 1
        img = trans(image)
        img = img.permute(1, 2, 0)

        return img.numpy()

    def one_zero_norm(self, image):
        image = torch.DoubleTensor(image.astype(np.int64)).permute(2, 0, 1)# input tensor image size with CxHxW
        channel, height, width = image.shape
        data = image.reshape(channel, -1)
        data_max = data.max(dim=1)[0]
        data_min = data.min(dim=1)[0]

        data = (data - data_min.unsqueeze(1)) / (data_max.unsqueeze(1) - data_min.unsqueeze(1))
        # (x - min(x))/(max(x) - min(x))  normalize to (0, 1) for each channel
        data = data.reshape(channel, height, width)

        return data.permute(1, 2, 0).numpy()

    def Apply_PCA(self, data, fraction, choice):
        """
        choice=1
        fraction：保留特征百分比
        choice=2
        fraction：降维后的个数
        """
        # PCA主成分分析
        if choice == 1:
            pc = spy.principal_components(data)
            pc_98 = pc.reduce(fraction=fraction)
            print("PCA_DIM", len(pc_98.eigenvalues))
            num_components = len(pc_98.eigenvalues)
            # spy.imshow(data=pc.cov, title="pc_cov")
            img_pc = pc_98.transform(data)
            # spy.imshow(img_pc[:, :, :3], stretch_all=True)
            return img_pc, num_components
        if choice == 2:
            new_data = np.reshape(data, (-1, data.shape[2]))
            pca = PCA(fraction)
            new_data = pca.fit_transform(new_data)
            print("PCA_DIM", new_data.shape[-1])
            n_components = new_data.shape[-1]
            new_data = np.reshape(new_data, (data.shape[0], data.shape[1], n_components))
            return new_data, n_components

    def get_spectral_superpixel(self, img_, segments):
        # 得到超像素块数量
        n_liantong = segments.max()
        # 计算每一个超像素块中的像素数量
        area = np.bincount(segments.flat)
        row, column, band = img_.shape
        img = np.zeros((n_liantong, band))
        trans = np.zeros([row * column, n_liantong], dtype=np.float32)
        for i in range(row):
            for j in range(column):
                img[segments[i][j]-1] += img_[i][j]

        for k in range(n_liantong):
            idx = np.where(segments.reshape(-1) == k + 1)[0]
            trans[idx, k] = 1
            img[k] /= area[k+1]

        bias = np.zeros_like(img_.reshape(-1, band))
        data = img_.reshape(-1, band)
        ind = segments.reshape(-1)
        for i in range(1, segments.max()):
            indics = np.where(ind == i)
            bias[indics] = data[indics] - img[i-1]

        return trans, img, bias

    def label_transform(self, gt):
        '''
            function：tensor label to 0-n for training
            input: gt
            output：gt
            tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
            -> tensor([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])
        '''
        gt = torch.DoubleTensor(gt)
        label = torch.unique(gt)  # tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
        gt_new = torch.zeros_like(gt)  # torch.size(610, 340)
        for each in range(len(label)):  # each 0~9
            indices = torch.where(gt == label[each])  # 2 tuple 两组索引数组来表示值的位置

            if label[0] == 0:
                gt_new[indices] = each - 1
            else:
                gt_new[indices] = each
        label_new = torch.unique(gt_new)  # tensor([-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.])

        return gt_new.numpy()

    def get_remove_background(self, prt_img, gt, segments):
        n_liantong = segments.max()
        area = np.bincount(segments.flat)
        # (num_nodes, n_features)
        non_img = []
        non_inds = []
        for i in range(n_liantong):
            ind = np.where(segments == i+1)
            count = len(np.where(gt[ind] != -1)[0])
            # count = 0
            # for j in range(len(ind[0])):
            #     if gt[ind[0][j]][ind[1][j]] != -1:
            #         count += 1
            if count/area[i+1] > 0.1:
                non_img.append(prt_img[i])
            else:
                non_inds.append(i)

        return np.array(non_img), non_inds


