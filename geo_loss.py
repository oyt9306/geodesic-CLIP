# Modified from https://github.com/chrysts/geodesic_continual_learning/blob/main/algorithms/GFK_distill_cosine.py
import torch.nn.functional as  F
import torch

class GeoCLIP:
    def __init__(self, dim=256, source_dim=128, target_dim=128):
        self.dim = dim
        self.eps = 1e-4
        self.source_dim=source_dim
        self.target_dim=target_dim

    def nullspace(self, At, rcond=None):
        # https://discuss.pytorch.org/t/nullspace-of-a-tensor/69980/4
        ut, st, vht = torch.Tensor.svd(At, some=False, compute_uv=True)
        vht=vht.T        
        Mt, Nt = ut.shape[0], vht.shape[1] 
        if rcond is None:
            rcondt = torch.finfo(st.dtype).eps * max(Mt, Nt)
        tolt = torch.max(st) * rcondt
        numt = torch.sum(st > tolt, dtype=int)
        nullspace = vht[numt:,:].T.conj()
        return nullspace

    def train_pca_tall(self, data,  subspace_dim):
        '''
        Modified PCA function, different from the one in sklearn
        :param data: data matrix
        :param mu_data: mu
        :param std_data: std
        :param subspace_dim: dim
        :return: a wrapped machine object
        '''
        data2 = data - data.mean(0)
        uu, _, _ = torch.svd(data2.double())
        uu = uu.float()
        subspace = uu[:, :subspace_dim]
        return subspace
    
    def sqrt_newton_schulz_minus(self, A, numIters=1):
        A = A.double()
        dim = A.data.shape[0]
        normA = A.mul(A).sum(dim=0).sum(dim=0).sqrt()
        Y = A.div(normA.view(1, 1).expand_as(A))
        I = torch.eye(dim,dim).double()
        Z = torch.eye(dim,dim).double()

        #A.register_hook(print)
        for i in range(numIters):
            T = 0.5*(3.0*I - Z.mm(Y))
            Y = Y.mm(T)
            Z = T.mm(Z)
        sZ = Z * 1./torch.sqrt(normA).expand_as(A) ### diabgi karena ini minus power
        return sZ
    
    def fit(self, input1, input2):
        '''
        Obtain the kernel G
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :param norm_inputs: normalize the inputs or not
        :return:  kernel G
        '''
        input1 = input1/ input1.norm(dim=-1, keepdim=True)
        input2 = input2/ input2.norm(dim=-1, keepdim=True)

        if input1.size(0) <= 2:
            source_dim = input1.size(0) 
            target_dim = input1.size(0) 
        else:
            source_dim = min(self.source_dim, input1.size(0)-2)
            target_dim = min(self.target_dim, input1.size(0)-2)
        num_nullspacedim = self.dim

        Ps = self.train_pca_tall(input1.t(), subspace_dim=source_dim)#.detach()
        Rs = self.nullspace(Ps.t().detach())[:, :num_nullspacedim]
        
        Ps = torch.cat([Ps, Rs], dim=1) ### adding columns
        N  = Ps.shape[1]  # L = NxK shot - 1
        Pt =  self.train_pca_tall(input2.t(), subspace_dim=target_dim)
        G  = self.gfk_G(Ps, Pt, N, source_dim, target_dim).detach()#.detach()

        qq1, qq2  = input1, input2
        nominator =  self.mahalanobis_dist(qq1, qq2, G)
        denom_q1  = torch.sqrt(self.mahalanobis_dist(qq1, qq1, G) )
        denom_q2  = torch.sqrt(self.mahalanobis_dist(qq2, qq2, G) )
        loss = nominator/(denom_q1*denom_q2) ### loss.mean()#
        return loss

    def mahalanobis_dist(self, x1, x2, G):
        x2_proj = G.float().mm(x2.t().float()).t()
        dist = torch.sum(x1 * x2_proj, dim=-1)
        return dist

    def gfk_G(self, Ps, Pt, N, source_dim, target_dim):
        A = Ps[:, :source_dim].t().mm(Pt)
        B = Ps[:, source_dim:].t().mm(Pt)
        device = A.device
        
        UU, SS, VV = self.HOGSVD_fit([A, B])
        V1, V2, V, Gam, Sig = UU[0], UU[1], VV, SS[0], SS[1]
        V2 = -V2

        Gam   = Gam.clamp(min=-1., max=1.)
        theta = torch.acos(Gam) 
        B1 = torch.diag(0.5* (1 + (torch.sin(2 * theta) / (2. * theta + 1e-12))))
        B2 = torch.diag(0.5* (torch.cos(2 * theta) - 1) / (2 * theta + 1e-12))
        B3 = B2
        B4 = torch.diag(0.5*  (1. - (torch.sin(2. * theta) / (2. * theta + 1e-12))))
        # delta_1 
        delta1_1 = torch.cat((V1, torch.zeros((N - source_dim, target_dim)).to(device)), dim=0) 
        delta1_2 = torch.cat((torch.zeros((source_dim, target_dim)).to(device), V2), dim=0)  
        delta1 = torch.cat((delta1_1, delta1_2), dim=1)
        # delta_2 
        delta2_1 = torch.cat((B1, B3), dim=0)  
        delta2_2 = torch.cat((B2, B4), dim=0)  
        delta2 = torch.cat((delta2_1, delta2_2), dim=1)
        # delta_3
        delta3_1 = torch.cat((V1.t(), torch.zeros((target_dim, source_dim)).to(device)), dim=0)  
        delta3_2 = torch.cat((torch.zeros((target_dim,   N-source_dim)).to(device), V2.t()), dim=0)  
        delta3 = torch.cat((delta3_1, delta3_2), dim=1)
        # matmul
        mm_delta = torch.matmul(delta1.double(), delta2.double())
        delta    = torch.matmul(mm_delta, delta3.double())
        G        = torch.matmul(torch.matmul(Ps.double(), delta), Ps.t().double()).float()
        return G

    ############################## HOGSVD #########################
    def inverse(self, X, meta_step=True):
        eye = torch.diag(torch.randn(X.shape[0])) * self.eps
        X = X.double() + eye.double().to(X.device)
        if meta_step:
            Z = self.sqrt_newton_schulz_minus(X.double(), numIters=1).float()
            A = Z.mm(Z) 
        else:
            A = torch.inverse(X)
        return A.float()
        
    def HOGSVD_fit_S(self, X):
        N = len(X)
        data_shape = X[0].shape
        A = [torch.matmul(x.transpose(0, 1), x).float() for x in X]
        A_inv = [self.inverse(a.double()).float() for a in A]
        S = torch.zeros((data_shape[1], data_shape[1])).float().to(X[0].device)
        for i in range(N):
            for j in range(i + 1, N):
                S = S + (torch.matmul(A[i], A_inv[j]) + torch.matmul(A[j], A_inv[i]))
        S = S / (N * (N - 1))
        return S

    def _eigen_decompostion(self, X, subspace_dim):
        eye = torch.diag(torch.ones(X.shape[0])) * self.eps
        X = X.double() + eye.double().to(X.device)
        V, _, _ = torch.svd(X.double())
        return  V.float()

    def HOGSVD_fit_B(self, X, V):
        X = [x.float() for x in X]
        B = [torch.matmul(V.t(), x.transpose(0, 1)).transpose(0, 1) for x in X]
        return B

    def HOGSVD_fit_U_Sigma(self, B):
        B = [b for b in B]
        sigmas = torch.stack([torch.norm(b, dim=0) for b in B])
        U = [b / (sigma  ) for b, sigma in zip(B, sigmas)]
        return sigmas, U

    def HOGSVD_fit(self, X):
        """A reference implementation of a fitting function for a transformer.
        Parameters
        ----------
        X : array-like, shape (n_samples, (n_rows_i, n_cols)
            List of training input samples. Eah input element has
            the same numbe of columns but can have unequal number of rows.
        Returns
        -------
        self : object
            Returns self.
        """
        X = [x for x in X]
        S = self.HOGSVD_fit_S(X).float()
        V = self._eigen_decompostion(S, S.size(0))
        B = self.HOGSVD_fit_B(X, V)
        sigmas, U = self.HOGSVD_fit_U_Sigma(B)
        return U, sigmas, V
