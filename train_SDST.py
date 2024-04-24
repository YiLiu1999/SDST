import torch.optim
from torch.nn import Module, Parameter
from utils import *
from torch import nn
from torch.nn import functional as F
import warnings

warnings.filterwarnings("ignore")


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x).squeeze()

        # assert x.size() == orig_q_size
        return x


class fc(nn.Module):
    def __init__(self, in_dim, out_dim, f=True):
        super(fc, self).__init__()
        if f:
            self.bn = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ELU()
            )
        else:
            self.bn = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )

    def forward(self, x):
        h = self.bn(x)
        return h


class transformer(nn.Module):
    def __init__(self, hidden_size, ffn_size, out_size, dropout_rate, attention_dropout_rate, num_heads):
        super(transformer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        h = self.fc(x)

        return h


class newtransformer(nn.Module):
    # hidden_size=input_size
    def __init__(self, hidden_size, ffn_size, out_size, dropout_rate, attention_dropout_rate, num_heads):
        super(newtransformer, self).__init__()

        self.self_attention_norm = nn.BatchNorm1d(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.BatchNorm1d(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, q, k, v, attn_bias=None):
        q = self.self_attention_norm(q)
        k = self.self_attention_norm(k)
        v = self.self_attention_norm(v)
        y = self.self_attention(q, k, v, attn_bias)
        y = self.self_attention_dropout(y)
        x = (q + k + v) / 6 + y / 2

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        h = self.fc(x)

        return h


class Autoformer(nn.Module):  # 0.78
    def __init__(self, in_dim, out_dim):
        super(Autoformer, self).__init__()
        self.en1 = fc(in_dim, 64)
        self.en2 = fc(64, 256)
        self.en3 = fc(256, 512)

        self.de1 = fc(768, 256)
        self.de2 = fc(320, 64)
        self.de3 = fc(64, in_dim)

        self.cq = nn.Linear(64, out_dim)
        self.ck = nn.Linear(256, out_dim)
        self.cv = nn.Linear(512, out_dim)
        self.newTransformer = newtransformer(out_dim, 4 * out_dim, out_dim, 0, 0, 3)
        self.Transformer1 = transformer(out_dim, 4 * out_dim, out_dim, 0, 0, 6)
        self.Transformer2 = transformer(out_dim, 4 * out_dim, out_dim, 0, 0, 6)
        self.Transformer3 = transformer(out_dim, 4 * out_dim, out_dim, 0, 0, 6)

    def forward(self, x):
        he1 = self.en1(x)
        he2 = self.en2(he1)
        he3 = self.en3(he2)

        hd1 = self.de1(torch.cat([he3, he2], dim=-1))
        hd2 = self.de2(torch.cat([hd1, he1], dim=-1))
        x_ = self.de3(hd2)
        alpha = opt.args.alpha
        new_x = alpha * x_ + (1 - alpha) * x

        hq = self.en1(new_x)
        hk = self.en2(hq)
        hv = self.en3(hk)

        q = self.cq(hq)
        k = self.ck(hk)
        v = self.cv(hv)

        h1 = self.newTransformer(q, k, v)
        h2 = self.Transformer1(h1)
        h3 = self.Transformer2(h2)

        return h3, x_


class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = nn.Tanh()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

        self.s = nn.Sigmoid()

    def forward(self, features, adj, att=False, active=False):
        if active:
            support = self.act(torch.mm(features, self.weight))
        else:
            support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)

        if att:
            a = self.s(torch.mm(output, output.t()))
            return output, a
        else:
            return output


class new_MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(new_MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, A, q, k, v1, v2, attn_bias=None):
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v1 = self.linear_v(v1).view(batch_size, -1, self.num_heads, d_v)
        v2 = self.linear_v(v2).view(batch_size, -1, self.num_heads, d_v)
        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v1 = v1.transpose(1, 2)
        v2 = v2.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        x = A.matmul(x.squeeze()).unsqueeze(-1).unsqueeze(-1)

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x1 = x.matmul(v1)  # [b, h, q_len, attn]
        x2 = x.matmul(v2)
        x = x1 + x2
        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x).squeeze()
        return x


class new_former(nn.Module):
    # hidden_size=input_size
    def __init__(self, hidden_size, ffn_size, out_size, dropout_rate, attention_dropout_rate, num_heads):
        super(new_former, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = new_MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, A, new_q, new_k, v1, v2, attn_bias=None):
        h = self.self_attention(A, new_q, new_k, v1, v2, attn_bias)
        h = self.self_attention_dropout(h)
        h = h + (new_q + new_k + v1 + v2) / 4

        y = self.ffn_norm(h)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = h + y
        h = self.fc(x)

        return h


class graphormer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(graphormer, self).__init__()
        self.GNN_q = GNNLayer(in_dim, hidden_dim)
        self.GNN_k = GNNLayer(in_dim, hidden_dim)
        self.GNN_v = GNNLayer(in_dim, hidden_dim)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.grapher1 = newtransformer(hidden_dim, 4 * hidden_dim, out_dim, 0, 0, 3)
        self.grapher2 = newtransformer(hidden_dim, 4 * hidden_dim, out_dim, 0, 0, 3)

        self.new_GNN_q = GNNLayer(out_dim, hidden_dim)
        self.new_GNN_k = GNNLayer(out_dim, hidden_dim)

        self.new_former = new_former(hidden_dim, 4 * hidden_dim, out_dim, 0, 0, 3)

    def forward(self, x, x_noise2, A, Ad):
        q1 = self.GNN_q(x, A)
        q1 = self.norm(q1)
        k1 = self.GNN_k(x, A)
        k1 = self.norm(k1)
        v1 = self.GNN_v(x, A)
        v1 = self.norm(v1)
        q2 = self.GNN_q(x_noise2, Ad)
        q2 = self.norm(q2)
        k2 = self.GNN_k(x_noise2, Ad)
        k2 = self.norm(k2)
        v2 = self.GNN_v(x_noise2, Ad)
        v2 = self.norm(v2)

        out1 = self.grapher1(q1, k1, v1)
        out2 = self.grapher2(q2, k2, v2)

        new_q = self.new_GNN_q(out1, A)
        new_k = self.new_GNN_k(out2, Ad)

        out = self.new_former(A, new_q, new_k, v1, v2)

        return out


class SDST(nn.Module):
    def __init__(self, n_node=None):
        super(SDST, self).__init__()

        # Auto Encoder
        self.Autoformer = Autoformer(opt.args.n_input, opt.args.embedding)

        # Graphormer
        self.SDFGM = graphormer(opt.args.n_input, 64, opt.args.embedding)

        # feature fusion
        self.f = nn.Linear(2 * opt.args.embedding, opt.args.embedding)

        # cluster layer (clustering assignment matrix)
        self.cluster_centers = Parameter(torch.Tensor(opt.args.n_clusters, opt.args.embedding), requires_grad=True)

    def forward(self, X, A, X_tilde, Ad):
        z1, DZ_Conv1 = self.Autoformer(X_tilde)
        z2, DZ_Conv2 = self.Autoformer(X)
        FCNout0 = (z1 + z2) / 2

        # node embedding encoded by IGAE
        graphor = self.SDFGM(X, X, A, Ad)

        # Cross = self.AFM(graphor3, FCNout0)
        out = torch.cat([graphor, FCNout0], dim=-1)
        Cross = self.f(out)

        # out
        y = F.softmax(Cross, dim=-1)
        FCN_out = F.softmax(FCNout0, dim=-1)
        GCN_out = F.softmax(graphor, dim=-1)

        return y, FCN_out, GCN_out, DZ_Conv1, DZ_Conv2


def training(model, Trans, X, img, y, A, Ad, b, k, opt):
    print("Model_Init...")
    # calculate embedding similarity and cluster centers
    centers, cluster_id, true_outbg, y_pred, mapping = model_init(model, Trans, X, y, A)

    # initialize cluster centers
    model.cluster_centers.data = torch.tensor(centers).to(device)
    print("Training…")
    # add gaussian noise
    X_tilde = gaussian_noised_feature(X, img)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.args.lr, weight_decay=1e-4)
    cerloss = nn.MSELoss()
    traincd_Z_acc_save = []
    traincd_Z_loss_save = []
    for epoch in range(opt.args.epochs):
        # input & output
        cross, FCN_out, GCN_out, DZ_Conv1, DZ_Conv2 = model(X, A, X_tilde, Ad)
        DYY = squared_distance(cross)
        a = torch.tensor(opt.args.n_samples, dtype=torch.float32)
        DZ_Conv1, DZ_Conv2 = DZ_Conv1 + b, DZ_Conv2 + b

        # dist_loss = _loss(cross, cluster_id, centers.cpu())
        re_loss = cerloss(DZ_Conv1, X) + cerloss(DZ_Conv2, X)  # 加上偏差
        kl_loss = klloss(cross, FCN_out, GCN_out, model) * 1e4
        sc_loss = (torch.sum(A * DYY) / a) * 1e2

        loss = sc_loss + 1 * kl_loss + 10 * (re_loss)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # clustering & evaluation
        acc, nmi, ari, ami, fmi, kappa, purity, centers, cluster_id, y_outbg, true_outbg, y_pred, mapping = clustering(
            GCN_out, Trans, y)
        if acc > opt.args.acc:
            opt.args.acc = acc
            opt.args.nmi = nmi
            opt.args.ami = ami
            opt.args.ari = ari
            opt.args.fmi = fmi
            opt.args.kappa = kappa
            opt.args.purity = purity
            print('epoch{}:   loss: {:.5f} || nmi: {:.5f} || ari: {:.5f}|| ami: {:.5f} '
                  '|| fmi: {:.5f} || kappa: {:.5f} || purity: {:.5f} || acc: {:.5f}'
                  .format(epoch, loss, nmi, ari, ami, fmi, kappa, purity, acc)
                  )
            torch.save(model.state_dict(),
                       'SDST/weight_parameters/{}_{}_{}_{:.5f}.pth'
                       .format(k, opt.args.name, opt.args.n_samples, acc))
            paua(y_outbg, true_outbg, y)
            transformed_arr = np.array([mapping[element] for element in y_pred.reshape(-1).cpu().numpy()]).reshape(
                opt.args.height,
                opt.args.width)
            y_pred = transformed_arr + 1
            Draw_Classification_Map1(opt.args.acc, y_pred, mapping, name=opt.args.name + '_bg')
            # no bg
            y_pred = y_pred.reshape((opt.args.height, opt.args.width))
            ind = np.where(y == -1)
            y_pred[ind] = 0
            Draw_Classification_Map1(opt.args.acc, y_pred, mapping, name=opt.args.name + '_no_bg')
            # Draw_Classification(y_pred, y, opt.args.name, acc)
        traincd_Z_acc_save.append(acc)
        # traincd_dist_loss_save.append(dist_loss.detach().cpu().numpy())
        traincd_Z_loss_save.append(loss.detach().cpu().numpy())

    return opt.args.acc, opt.args.nmi, opt.args.ari, opt.args.ami, opt.args.fmi, opt.args.kappa, opt.args.purity


def train(Q, gt, prt_img0, img, A, bias, k, opt):
    bias = torch.from_numpy(bias).to(device)
    A = torch.from_numpy(A.astype(np.float32))
    Ad = diffusion_adj(A, mode="ppr", transport_rate=opt.args.ppr).to(device)
    S0 = torch.from_numpy(prt_img0.astype(np.float32)).to(device)
    img = torch.from_numpy(img.astype(np.float32)).to(device)
    Q = torch.from_numpy(Q.astype(np.float32)).to(device)
    A = A.to(device)
    b = Q.T.matmul(bias)

    opt.args.device = device
    model = SDST(n_node=S0.shape[0]).to(device)
    print(model)
    acc, nmi, ari, ami, fmi, kappa, purity = training(model, Q, S0, img, gt, A, Ad, b, k, opt)
    print("ACC: {:.4f},".format(acc), "NMI: {:.4f},".format(nmi), "AMI: {:.4f},".format(ami), "ARI: {:.4f},".format(ari)
          , "FMI: {:.4f}".format(fmi), "KAPPA: {:.4f},".format(kappa), "PURITY: {:.4f}".format(purity))
    opt.args.acc, opt.args.nmi, opt.args.ari, opt.args.f1, opt.args.kappa, opt.args.purity = 0, 0, 0, 0, 0, 0
