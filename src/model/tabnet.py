import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import TimeEmbedding

class _Sparsemax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        Args:
            ctx: autograd context
            input (torch.Tensor): 2-D tensor, (N, C).
        Returns:
            torch.Tensor: (N, C).
        """
        dim = 1
        # translate input by max for numerical stability.
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)
        z_sorted = torch.sort(input, dim=dim, descending=True)[0]
        input_size = input.size()[dim]
        range_values = torch.arange(1, input_size + 1).to(input.device)
        range_values = range_values.expand_as(z_sorted)

        # Determine sparsity of projection
        range_ = torch.arange(
            1, input.size(dim) + 1, dtype=input.dtype, device=input.device
        )
        bound = 1.0 + range_ * z_sorted
        cumsum_zs = torch.cumsum(z_sorted, dim)
        is_gt = torch.gt(bound, cumsum_zs)
        k = torch.max(is_gt * range_, dim=dim, keepdim=True)[0]

        zs_sparse = is_gt * z_sorted

        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)
        output = (input - taus).clamp(min=0.0)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        dim = 1

        nonzeros = output != 0.0
        sum_grad = torch.sum(grad_output * nonzeros, dim=dim, keepdim=True) / torch.sum(
            nonzeros, dim=dim, keepdim=True
        )
        return nonzeros * (grad_output - sum_grad.expand_as(grad_output))


sparsemax = _Sparsemax.apply

class GLU(nn.Module):
    def forward(self, input):
        return F.glu(input)
class GhostBatchNorm(nn.Module):
    def __init__(self, num_features: int, momentum: float, ghost_batch_size: int):
        super(GhostBatchNorm,self).__init__()
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum)
        self.ghost_batch_size = ghost_batch_size
    def forward(self, input_tensor):
        batch_size = input_tensor.size(0)
        chunks = input_tensor.chunk((batch_size - 1) // self.ghost_batch_size + 1, dim=0)
        normalized_chunks = [self.bn(chunk) for chunk in chunks]
        return torch.cat(normalized_chunks, dim=0)
class SharedFeatureTransformer(nn.Module):
    def __init__(self,
        in_channels: int,
        hidden_size: int,
        bn_momentum: float,
        ghost_batch_size: int,
    ):
        super(SharedFeatureTransformer,self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, hidden_size * 2, bias=False),
            GhostBatchNorm(
                hidden_size * 2, momentum=bn_momentum, ghost_batch_size=ghost_batch_size
            ),
            GLU(),
        )
        self.residual_block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2, bias=False),
            GhostBatchNorm(
                hidden_size * 2, momentum=bn_momentum, ghost_batch_size=ghost_batch_size
            ),
            GLU(),
        )

    def forward(self, input_tensor):
        """
        Args:
            input (torch.Tensor): (N, C)
        Returns:
            torch.Tensor: (N, C)
        """
        x = self.block(input_tensor)
        return (x + self.residual_block(x)) * math.sqrt(0.5)
class FeatureTransformer(nn.Module):
    def __init__(self, in_channels: int, bn_momentum: float, ghost_batch_size: int):
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2, bias=False),
            GhostBatchNorm(
                in_channels * 2, momentum=bn_momentum, ghost_batch_size=ghost_batch_size
            ),
            GLU(),
        )

    def forward(self, input_tensor):
        """
        Args:
            input (torch.Tensor): (N, C)
        Returns:
            torch.Tensor: (N, C)
        """
        return (input_tensor + self.residual_block(input_tensor)) * math.sqrt(0.5)
class TabNet(nn.Module):
    def __init__(self,
        ent_vocab_size:int,
        ent_size: int,
        time_size:int,
        value_size:int,
        ent_dim=80,
        time_embedding_dim=30,
        year_span = 101,
        feature_channels = 10,
        emb_dropout=0.2,
        out_channels: int=1,
        n_decision_steps:int=4,
        bn_momentum: float = 0.1,
        n_d: int = 16,
        n_a: int = 16,
        relaxation_factor: float = 2.0,
        ghost_batch_size: int = 256,
    ):
        """
        Args:
            dense_channels: number of dense features.
            ent_size: entity feature cardinalities.
            out_channels: number of output channels.
            n_decision_steps: number of decision step layers.
            cat_emb_dim: categorical feature embedding size.
            bn_momentum: batch normalization momentum.
            n_d: hidden size of decision output.
            n_a: hidden size of attentive transformer.
            relaxation_factor: relaxation parameter of feature selection regularization.
            ghost_batch_size: ghost batch size for GhostBatchNorm.
        """
        super(TabNet,self).__init__()
        # Embedding 区域
        self.ent_vocab_size = ent_vocab_size
        self.ent_size = ent_size
        self.time_size = time_size
        self.value_size = value_size
        
        self.year_span = year_span
        self.ent_dim = ent_dim
        self.feature_channels = feature_channels
        self.time_embedding_dim = time_embedding_dim

        self.time_embed = TimeEmbedding(years_num=year_span,embedding_dim=time_embedding_dim,
                                        emb_dropout=emb_dropout)
        self.ent_embed = nn.Embedding(self.ent_vocab_size,self.ent_dim)

        nn.init.xavier_uniform_(self.ent_embed.weight, gain=nn.init.calculate_gain('relu'))
        self.n_d = n_d
        self.n_a = n_a
        self.bais_wet = nn.Linear(self.ent_size*self.ent_dim,feature_channels)
        self.bais_wtt = nn.Linear(self.time_size*self.time_embedding_dim,feature_channels)
        self.bais_wvt = nn.Linear(self.value_size*self.feature_channels,feature_channels)
        
        self.n_decision_steps = n_decision_steps
        
        self.relaxation_factor = relaxation_factor
        self.dense_bn = nn.BatchNorm1d(feature_channels, momentum=bn_momentum)

        hidden_size = n_d + n_a

        shared_feature_transformer = SharedFeatureTransformer(
            feature_channels, hidden_size, bn_momentum, ghost_batch_size
        )
        self.feature_transformers = nn.ModuleList(
            [
                nn.Sequential(
                    shared_feature_transformer,
                    FeatureTransformer(hidden_size, bn_momentum, ghost_batch_size),
                    FeatureTransformer(hidden_size, bn_momentum, ghost_batch_size),
                )
                for _ in range(n_decision_steps)
            ]
        )
        self.attentive_transformers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_a, feature_channels, bias=False),
                    GhostBatchNorm(
                        feature_channels,
                        momentum=bn_momentum,
                        ghost_batch_size=ghost_batch_size,
                    ),
                )
                for _ in range(n_decision_steps - 1)
            ]
        )
        self.fc = nn.Linear(n_d, out_channels, bias=False)
        
    def forward(self,ent_tensor,val_tensor,year_tensor,month_tensor,day_tensor):

        batch_size = ent_tensor.shape[0]
        ent_embs = self.ent_embed(ent_tensor)
        time_embs = self.time_embed(year_tensor,month_tensor,day_tensor)
        val_tensor = val_tensor.unsqueeze(1).repeat(1,self.feature_channels,1)
        device = ent_tensor.device
        # 属性对齐
        hid_ent = torch.relu(self.bais_wet(ent_embs.reshape(batch_size,-1)))
        hid_time = torch.relu(self.bais_wtt(time_embs.reshape(batch_size,-1)))
        hid_value = torch.relu(self.bais_wvt(val_tensor.reshape(batch_size,-1)))
        
        feature = hid_ent + hid_time + hid_value

        aggregated_output = torch.zeros(
            batch_size, self.n_d, dtype=torch.float, device=device
        ) # (N,n_d)
        masked_feature = feature # (N,linear_dim)
        prior_scale_term = torch.ones(
            batch_size, feature.size(1), dtype=torch.float, device=device
        ) # (N,linear_dim)
        mask = torch.zeros_like(prior_scale_term) # (N,linear_dim)

        masks: List[torch.Tensor] = []
        
        aggregated_masks = torch.zeros_like(prior_scale_term) # (N,linear_dim)
        sparsity_regularization = torch.tensor(0.0).to(dtype=torch.float, device=device)
        
        for step in range(self.n_decision_steps):
            x = self.feature_transformers[step](masked_feature)  # (N, hidden_size)
            decision_out, coef_out = x.split(self.n_d, dim=1)  # (N, n_d), (N, n_a)

            if step != 0:
                decision_out = F.relu(decision_out)
                aggregated_output += decision_out
                # For visualization and interpretability, aggregate feature mask values for all steps.
                scale = decision_out.sum(1, keepdim=True) / (self.n_decision_steps - 1)
                aggregated_masks += scale * mask

            if step != self.n_decision_steps - 1:
                # Prepare mask values for the next decision step.
                mask = self.attentive_transformers[step](coef_out)
                mask = mask * prior_scale_term
                mask = sparsemax(mask)
                # Update prior scale term to regulate feature selection
                prior_scale_term = prior_scale_term * (self.relaxation_factor - mask)
                # Update sparsity regularization
                sparsity_regularization += (mask * (mask + 1e-5).log()).sum(1).mean(
                    0
                ) / (self.n_decision_steps - 1)
                masked_feature = mask * feature
                masks.append(mask)
        logits = self.fc(aggregated_output).squeeze()
        return torch.sigmoid(logits), masks, sparsity_regularization
