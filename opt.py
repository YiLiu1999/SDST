import argparse

parser = argparse.ArgumentParser(description='DCRN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--k', type=int, default=1)
# setting
parser.add_argument('--name', type=str, default="dataset")
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--pca', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--show_training_details', type=bool, default=False)

# IGAE structure parameter from DFCN
parser.add_argument('--gae_n_enc_1', type=int, default=16)
# GAT
parser.add_argument('--alpha', type=float, default=0.4)
parser.add_argument('--ppr', type=float, default=0.4)
parser.add_argument('--n_clusters', type=int, default=9)
parser.add_argument('--embedding', type=int, default=20)
parser.add_argument('--n_input', type=int, default=200)
parser.add_argument('--height', type=int, default=512)
parser.add_argument('--width', type=int, default=217)
parser.add_argument('--n_samples', type=int, default=1161)
parser.add_argument('--acc', type=float, default=0)
# clustering performance: acc, nmi, ari, f1
parser.add_argument('--nmi', type=float, default=0)
parser.add_argument('--ami', type=float, default=0)
parser.add_argument('--ari', type=float, default=0)
parser.add_argument('--fmi', type=float, default=0)
parser.add_argument('--kappa', type=float, default=0)
parser.add_argument('--purity', type=float, default=0)

args = parser.parse_args()
