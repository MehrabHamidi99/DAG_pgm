import torch
from generate_data import *
from torch.utils.data import Dataset, DataLoader


class SyntheticDataset(Dataset):
    def __init__(self, n_points, g_path1,g_path2=None,g_path3=None, sem_type='non_linear_gaussian', sigma_min=0.4, sigma_max=0.8,
                 load_data=True,data_path=None,mode='train',device='cuda'):
        if device == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.n_points = n_points
        self.sem_type = sem_type
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.G, self.A = load_graphs(g_path1)
        if g_path2 is not None:
            self.G2, self.A2 = load_graphs(g_path2)
        if not load_data:
            X1, self.sigmas1 = gaussian_anm(self.G, self.A, n_points, sem_type, sigma_min, sigma_max)
            if g_path2 is not None:
                X2,self.sigmas2 = gaussian_anm(self.G2, self.A2, n_points, sem_type, sigma_min, sigma_max)
            #concatenate X1 and X2 in the data with labels from which dataset they are from
            X1 = np.concatenate((X1, np.zeros((n_points, 1))), axis=1)
            if g_path2 is not None:
                X2 = np.concatenate((X2, np.ones((n_points, 1))), axis=1)
                self.X = np.concatenate((X1,X2), axis=0)
            else:
                self.X = X1

        else:
            if g_path3 is not None:
                data_path += '10_nodes_'
            if mode == 'train':
                data_path = data_path + "train_"
            else:
                data_path = data_path + "test_"
            X1 = np.load(data_path + "graph1.npy")
            if g_path2 is not None:
                X2 = np.load(data_path + "graph2.npy")
                self.X = np.concatenate((X1, X2), axis=0)
                self.labels = np.concatenate((np.zeros(n_points), np.ones(n_points)), axis=0)
            if g_path3 is not None:
                X3 = np.load(data_path + "graph3.npy")
                self.X = np.concatenate((X1, X2, X3), axis=0)
                self.labels = np.concatenate((np.zeros(n_points), np.ones(n_points), 2*np.ones(n_points)), axis=0)
            else:
                self.X = X1
                self.labels = np.zeros(n_points)


    def __len__(self):
        return self.n_points

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32).to(self.device),torch.tensor(self.labels[idx], dtype=int).to(self.device)