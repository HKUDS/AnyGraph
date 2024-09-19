import torch as t
from torch import nn
import torch.nn.functional as F
from params import args
import numpy as np
from Utils.TimeLogger import log
from torch.nn import MultiheadAttention
from time import time

init = nn.init.xavier_uniform_
# init = nn.init.normal_
uniformInit = nn.init.uniform_

class FeedForwardLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=True, act=None):
        super(FeedForwardLayer, self).__init__()
        self.linear = nn.Linear(in_feat, out_feat, bias=bias)#, dtype=t.bfloat16)
        if act == 'identity' or act is None:
            self.act = None
        elif act == 'leaky':
            self.act = nn.LeakyReLU(negative_slope=args.leaky)
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'relu6':
            self.act = nn.ReLU6()
        else:
            raise Exception('Error')
    
    def forward(self, embeds):
        if self.act is None:
            return self.linear(embeds)
        return self.act(self.linear(embeds))

class TopoEncoder(nn.Module):
    def __init__(self):
        super(TopoEncoder, self).__init__()

        self.layer_norm = nn.LayerNorm(args.latdim, elementwise_affine=False)

    def forward(self, adj, embeds, normed=False):
        with t.no_grad():
            if not normed:
                embeds = self.layer_norm(embeds)
            # embeds_list = []
            final_embeds = 0
            if args.gnn_layer == 0:
                final_embeds = embeds
                # embeds_list.append(embeds)
            for _ in range(args.gnn_layer):
                embeds = t.spmm(adj, embeds)
                final_embeds += embeds
                # embeds_list.append(embeds)
            embeds = final_embeds#sum(embeds_list)
        return embeds

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.dense_layers = nn.Sequential(*[FeedForwardLayer(args.latdim, args.latdim, bias=True, act=args.act) for _ in range(args.fc_layer)])
        self.layer_norms = nn.Sequential(*[nn.LayerNorm(args.latdim, elementwise_affine=True) for _ in range(args.fc_layer)])
        self.dropout = nn.Dropout(p=args.drop_rate)
    
    def forward(self, embeds):
        for i in range(args.fc_layer):
            embeds = self.layer_norms[i](self.dropout(self.dense_layers[i](embeds)) + embeds)
        return embeds

class GTLayer(nn.Module):
    def __init__(self):
        super(GTLayer, self).__init__()
        self.multi_head_attention = MultiheadAttention(args.latdim, args.head, dropout=0.1, bias=False)#, dtype=t.bfloat16)
        self.dense_layers = nn.Sequential(*[FeedForwardLayer(args.latdim, args.latdim, bias=True, act=args.act) for _ in range(2)])# bias=False
        self.layer_norm1 = nn.LayerNorm(args.latdim, elementwise_affine=True)#, dtype=t.bfloat16)
        self.layer_norm2 = nn.LayerNorm(args.latdim, elementwise_affine=True)#, dtype=t.bfloat16)
        self.fc_dropout = nn.Dropout(p=args.drop_rate)
    
    def _pick_anchors(self, embeds):
        perm = t.randperm(embeds.shape[0])
        anchors = perm[:args.anchor]
        return embeds[anchors]
    
    def forward(self, embeds):
        anchor_embeds = self._pick_anchors(embeds)
        _anchor_embeds, _ = self.multi_head_attention(anchor_embeds, embeds, embeds)
        anchor_embeds = _anchor_embeds + anchor_embeds
        _embeds, _ = self.multi_head_attention(embeds, anchor_embeds, anchor_embeds, need_weights=False)
        embeds = self.layer_norm1(_embeds + embeds)
        _embeds = self.fc_dropout(self.dense_layers(embeds))
        embeds = (self.layer_norm2(_embeds + embeds))
        return embeds

class GraphTransformer(nn.Module):
    def __init__(self):
        super(GraphTransformer, self).__init__()
        self.gt_layers = nn.Sequential(*[GTLayer() for i in range(args.gt_layer)])

    def forward(self, embeds):
        for i, layer in enumerate(self.gt_layers):
            embeds = layer(embeds) / args.scale_layer
        return embeds

class Feat_Projector(nn.Module):
    def __init__(self, feats):
        super(Feat_Projector, self).__init__()

        if args.proj_method == 'uniform':
            self.proj_embeds = self.uniform_proj(feats)
        elif args.proj_method == 'svd' or args.proj_method == 'both':
            self.proj_embeds = self.svd_proj(feats)
        elif args.proj_method == 'random':
            self.proj_embeds = self.random_proj(feats)
        elif args.proj_method == 'original':
            self.proj_embeds = feats
        self.proj_embeds = t.flip(self.proj_embeds, dims=[-1])
        self.proj_embeds = self.proj_embeds.detach()
    
    def svd_proj(self, feats):
        if args.latdim > feats.shape[0] or args.latdim > feats.shape[1]:
            dim = min(feats.shape[0], feats.shape[1])
            decom_feats, s, decom_featdim = t.svd_lowrank(feats, q=dim, niter=args.niter)
            decom_feats = t.concat([decom_feats, t.zeros([decom_feats.shape[0], args.latdim-dim]).to(args.devices[0])], dim=1)
            s = t.concat([s, t.zeros(args.latdim - dim).to(args.devices[0])])
        else:
            decom_feats, s, decom_featdim = t.svd_lowrank(feats, q=args.latdim, niter=args.niter)
        decom_feats = decom_feats @ t.diag(t.sqrt(s))
        return decom_feats.cpu()
    
    def uniform_proj(self, feats):
        projection = init(t.empty(args.featdim, args.latdim))
        return feats @ projection
    
    def random_proj(self, feats):
        projection = init(t.empty(feats.shape[0], args.latdim))
        return projection
    
    def forward(self):
        return self.proj_embeds

class Adj_Projector(nn.Module):
    def __init__(self, adj):
        super(Adj_Projector, self).__init__()

        if args.proj_method == 'adj_svd' or args.proj_method == 'both':
            self.proj_embeds = self.svd_proj(adj)
        self.proj_embeds = self.proj_embeds.detach()
    
    def svd_proj(self, adj):
        q = args.latdim
        if args.latdim > adj.shape[0] or args.latdim > adj.shape[1]:
            dim = min(adj.shape[0], adj.shape[1])
            svd_u, s, svd_v = t.svd_lowrank(adj, q=dim, niter=args.niter)
            svd_u = t.concat([svd_u, t.zeros([svd_u.shape[0], args.latdim-dim]).to(args.devices[0])], dim=1)
            svd_v = t.concat([svd_v, t.zeros([svd_v.shape[0], args.latdim-dim]).to(args.devices[0])], dim=1)
            s = t.concat([s, t.zeros(args.latdim-dim).to(args.devices[0])])
        else:
            svd_u, s, svd_v = t.svd_lowrank(adj, q=q, niter=args.niter)
        svd_u = svd_u @ t.diag(t.sqrt(s))
        svd_v = svd_v @ t.diag(t.sqrt(s))
        if adj.shape[0] != adj.shape[1]:
            projection = t.concat([svd_u, svd_v], dim=0)
        else:
            projection = svd_u + svd_v
        return projection.cpu()
    
    def forward(self):
        return self.proj_embeds

class Expert(nn.Module):
    def __init__(self):
        super(Expert, self).__init__()
        
        self.topo_encoder = TopoEncoder().to(args.devices[0])
        if args.nn == 'mlp':
            self.trainable_nn = MLP().to(args.devices[1])
        else:
            self.trainable_nn = GraphTransformer().to(args.devices[1])
        self.trn_count = 1
    
    def forward(self, projectors, pck_nodes=None):
        embeds = projectors.to(args.devices[1])
        if pck_nodes is not None:
            embeds = embeds[pck_nodes]
        embeds = self.trainable_nn(embeds)
        return embeds

    def pred_norm(self, pos_preds, neg_preds):
        pos_preds_num = pos_preds.shape[0]
        neg_preds_shape = neg_preds.shape
        preds = t.concat([pos_preds, neg_preds.view(-1)])
        preds = preds - preds.max()
        pos_preds = preds[:pos_preds_num]
        neg_preds = preds[pos_preds_num:].view(neg_preds_shape)
        return pos_preds, neg_preds
    
    def cal_loss(self, batch_data, projectors):
        ancs, poss, negs = list(map(lambda x: x.to(args.devices[1]), batch_data))
        self.trn_count += ancs.shape[0]
        pck_nodes = t.concat([ancs, poss, negs])
        final_embeds = self.forward(projectors, pck_nodes)
        # anc_embeds, pos_embeds, neg_embeds = final_embeds[ancs], final_embeds[poss], final_embeds[negs]
        anc_embeds, pos_embeds, neg_embeds = t.split(final_embeds, [ancs.shape[0]] * 3)
        if final_embeds.isinf().any() or final_embeds.isnan().any():
            raise Exception('Final embedding fails')
        
        if args.loss == 'ce':
            pos_preds, neg_preds = self.pred_norm((anc_embeds * pos_embeds).sum(-1), anc_embeds @ neg_embeds.T)
            if pos_preds.isinf().any() or pos_preds.isnan().any() or neg_preds.isinf().any() or neg_preds.isnan().any():
                raise Exception('Preds fails')
            pos_loss = pos_preds
            neg_loss = (neg_preds.exp().sum(-1) + pos_preds.exp() + 1e-8).log()
            pre_loss = -(pos_loss - neg_loss).mean()
        elif args.loss == 'bpr':
            pos_preds = (anc_embeds * pos_embeds).sum(-1)
            neg_preds = (anc_embeds * neg_embeds).sum(-1)
            pos_loss, neg_loss = pos_preds, neg_preds
            pre_loss = -((pos_preds - neg_preds).sigmoid() + 1e-10).log().mean() 

        if t.isinf(pre_loss).any() or t.isnan(pre_loss).any():
            raise Exception('NaN or Inf')

        reg_loss = sum(list(map(lambda W: W.norm(2).square() * args.reg, self.parameters())))
        loss_dict = {'preloss': pre_loss, 'regloss': reg_loss, 'posloss': pos_loss.mean(), 'negloss': neg_loss.mean()}
        return pre_loss + reg_loss, loss_dict
    
    def pred_for_test(self, batch_data, cand_size, projectors, rerun_embed=True):
        ancs, trn_mask = list(map(lambda x: x.to(args.devices[1]), batch_data))
        if rerun_embed:
            try:
                final_embeds = self.forward(projectors)
            except Exception:
                final_embeds_list = []
                div = args.batch * 3
                temlen = projectors.shape[0] // div
                for i in range(temlen):
                    st, ed = div * i, div * (i + 1)
                    tem_projectors = projectors[st: ed, :]
                    final_embeds_list.append(self.forward(tem_projectors))
                if temlen * div < projectors.shape[0]:
                    tem_projectors = projectors[temlen*div:, :]
                    final_embeds_list.append(self.forward(tem_projectors))
                final_embeds = t.concat(final_embeds_list, dim=0)
            self.final_embeds = final_embeds
        final_embeds = self.final_embeds
        anc_embeds = final_embeds[ancs]
        cand_embeds = final_embeds[-cand_size:]

        mask_mat = t.sparse.FloatTensor(trn_mask, t.ones(trn_mask.shape[1]).to(args.devices[1]), t.Size([ancs.shape[0], cand_size]))
        dense_mat = mask_mat.to_dense()
        all_preds = anc_embeds @ cand_embeds.T * (1 - dense_mat) - dense_mat * 1e8
        return all_preds
    
    def pred_for_node_test(self, nodes, cand_size, feats, rerun_embed=True):
        if rerun_embed:
            final_embeds = self.forward(feats)
            self.final_embeds = final_embeds
        final_embeds = self.final_embeds
        anc_embeds = final_embeds[nodes]
        class_embeds = final_embeds[-cand_size:]
        preds = anc_embeds @ class_embeds.T
        return t.argmax(preds, dim=-1)

    def attempt(self, topo_embeds, dataset):
        final_embeds = self.trainable_nn(topo_embeds)
        rows, cols, negs = list(map(lambda x: t.from_numpy(x).long().to(args.devices[1]), [dataset.ancs, dataset.poss, dataset.negs]))
        if rows.shape[0] > args.attempt_cache:
            random_perm = t.randperm(rows.shape[0], device=args.devices[0])
            pck_perm = random_perm[:args.attempt_cache]
            rows = rows[pck_perm]
            cols = cols[pck_perm]
            negs = negs[pck_perm]
        while True:
            try:
                row_embeds = final_embeds[rows]
                col_embeds = final_embeds[cols]
                neg_embeds = final_embeds[negs]
                score = ((row_embeds * col_embeds).sum(-1) - (row_embeds * neg_embeds).sum(-1)).sigmoid().mean().item()
                break
            except Exception:
                args.attempt_cache = args.attempt_cache // 2
                random_perm = t.randperm(rows.shape[0], device=args.devices[0])
                pck_perm = random_perm[:args.attempt_cache]
                rows = rows[pck_perm]
                cols = cols[pck_perm]
                negs = negs[pck_perm]
        t.cuda.empty_cache()
        return score

class AnyGraph(nn.Module):
    def __init__(self):
        super(AnyGraph, self).__init__()
        self.experts = nn.ModuleList([Expert() for _ in range(args.expert_num)]).cuda()
        self.opts = list(map(lambda expert: t.optim.Adam(expert.parameters(), lr=args.lr, weight_decay=0), self.experts))

    def one_graph_one_expert(self, handlers, log_assignment=False):
        self.assignment = [i for i in range(len(handlers))]
        if log_assignment:
            print('\n----------\nAssignment')
            for dataset_id, handler in enumerate(handlers):
                print(handler.data_name, f'{self.assignment[dataset_id]}')
    
    def store_history_assignment(self, assignment):
        pass
        
    def assign_experts(self, handlers, reca=True, log_assignment=False):
        if args.expert_num == 1:
            self.assignment = [0] * len(handlers)
            return
        if args.assignment == 'one-graph-one-expert':
            self.one_graph_one_expert(handlers, log_assignment)
            return
        if args.assignment not in ['top1', 'top2']:
            raise Exception('Unrecognized assigning methods')
        try:
            expert_scores = np.array(list(map(lambda expert: expert.trn_count, self.experts)))
            expert_scores = (1.0 - expert_scores / np.sum(expert_scores)) * args.reca_range + 1.0 - args.reca_range / 2
        except Exception:
            expert_scores = np.ones(len(self.experts))
        with t.no_grad():
            assignment = [list() for i in range(len(handlers))]
            for dataset_id, handler in enumerate(handlers):
                topo_embeds = handler.projectors.to(args.devices[1])
                for expert_id, expert in enumerate(self.experts):
                    expert = expert.to(args.devices[1])
                    score = expert.attempt(topo_embeds, handler.trn_loader.dataset)
                    if reca:
                        score *= expert_scores[expert_id]
                    assignment[dataset_id].append((expert_id, score))
                assignment[dataset_id].sort(key=lambda x: x[1], reverse=True)
                if args.assignment == 'top2':
                    if assignment[dataset_id][0][1] - assignment[dataset_id][1][1] < 0.05 and np.random.uniform() > 0.5:
                        tem = assignment[dataset_id][0]
                        assignment[dataset_id][0] = assignment[dataset_id][1]
                        assignment[dataset_id][1] = tem
            if log_assignment:
                print('\n----------\nAssignment')
                for dataset_id, handler in enumerate(handlers):
                    out = ''
                    for exp_idx in range(min(4, len(self.experts))):
                        out += f'({assignment[dataset_id][exp_idx][0]}, {assignment[dataset_id][exp_idx][1]}) '
                    print(handler.data_name, out)
                print('----------\n')

            self.assignment = list(map(lambda x: x[0][0], assignment))
    
    def summon(self, dataset_id):
        return self.experts[self.assignment[dataset_id]]
    
    def summon_opt(self, dataset_id):
        return self.opts[self.assignment[dataset_id]]
