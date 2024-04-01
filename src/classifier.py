from typing import Tuple,Union,Optional
import torch
import torch.nn.init as init
import torch.nn.functional as F

from .utils import compute_wce, to_one_hot


class TransitionClassifier(object):
    """
    Class Similarity Transition: Decoupling Class Similarities and Imbalance 
    from Generalized Few-shot Segmentation
    """

    def __init__(self, args, base_weight, base_bias, n_tasks):
        self.num_base_classes_and_bg = base_weight.size(-1)
        self.num_novel_classes = args.num_classes_val
        self.num_classes = self.num_base_classes_and_bg + self.num_novel_classes
        self.n_tasks = n_tasks

        # Snapshot of the model right after training, frozen
        self.snapshot_weight: torch.Tensor = base_weight.squeeze(
            0).squeeze(0).clone()
        self.snapshot_bias: torch.Tensor = base_bias.clone()

        # [n_tasks, c, num_base_classes_and_bg]
        self.base_weight: torch.Tensor = base_weight.squeeze(
            0).repeat(self.n_tasks, 1, 1)
        # [n_tasks, num_base_classes_and_bg]
        self.base_bias: torch.Tensor = base_bias.unsqueeze(
            0).repeat(self.n_tasks, 1)

        self.novel_weight: torch.Tensor = None
        self.novel_bias: torch.Tensor = None

        self.row_size = self.num_base_classes_and_bg
        self.col_size = self.num_classes
        self.transition_column_weight = None
        self.transition_column_bias = None
        self.transition_row_weight = None
        self.transition_row_bias = None

        self.layer_scale = None
        self.transition_matrix = None
        self.epoch_LDAM=args.epoch_LDAM
        self.semantic_transports = None

        self.pi: torch.Tensor = None

        self.fine_tune_base_classifier = args.fine_tune_base_classifier
        self.lr = args.cls_lr
        self.adapt_iter = args.adapt_iter
        self.weights = args.weights
        self.pi_estimation_strategy = args.pi_estimation_strategy
        self.pi_update_at = args.pi_update_at

    def compute_optimal_transport(self, M: torch.Tensor, target: torch.Tensor, source: torch.Tensor, lam=0.45, epsilon=1e-7) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the optimal transport matrix and Slinkhorn distance using the
        Sinkhorn-Knopp algorithm
        Inputs:
            - M : cost matrix (n x m)
            - r : vector of marginals (n, )
            - c : vector of marginals (m, )
            - lam : strength of the entropic regularization
            - epsilon : convergence parameter
        Outputs:
            - P : optimal transport matrix (n x m)
            - dist : Sinkhorn distance
        """
        n, m = M.shape
        P = torch.exp(-lam * M)
        P /= P.sum()
        u = torch.zeros(n, device=M.device)
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(1))) > epsilon:
            u = P.sum(1)
            P *= (target / u).reshape((-1, 1))
            P *= (source / P.sum(0)).reshape((1, -1))
        return P, torch.sum(P * M)

    @staticmethod
    def _valid_mean(t: torch.Tensor, valid_pixels: torch.Tensor, dim: Union[int , Optional[int]]):
        s = (valid_pixels * t).sum(dim=dim)
        return s / (valid_pixels.sum(dim=dim) + 1e-10)

    def init_prototypes(self, features_s: torch.Tensor, gt_s: torch.Tensor, zero_prototype: bool = False) -> None:
        """
        inputs:
            features_s : shape [num_novel_classes, shot, c, h, w]
            gt_s : shape [num_novel_classes, shot, H, W]
        """
        # Downsample support masks
        ds_gt_s = F.interpolate(
            gt_s.float(), size=features_s.shape[-2:], mode="nearest"
        )
        # [n_novel_classes, shot, 1, h, w]
        ds_gt_s = ds_gt_s.long().unsqueeze(2)

        ds_gt_s_copy = ds_gt_s.flatten(0, 1).unsqueeze(0)
        features_s_copy = features_s.flatten(0, 1).unsqueeze(0)
        pseudo_base_preds = self.get_base_snapshot_probas(
            features_s_copy).argmax(dim=2)
        transportation_cost = torch.zeros(self.num_novel_classes, self.num_base_classes_and_bg, device=features_s.device)
        for base_cls in range(0, self.num_base_classes_and_bg):
            base_cls_mask = (pseudo_base_preds == base_cls).unsqueeze(dim=2)
            base_cls_mean_feature_vector = (
                features_s_copy * base_cls_mask).mean(dim=(1, 3, 4)).squeeze()
            for novel_cls in range(self.num_base_classes_and_bg, self.num_classes):
                novel_cls_mask = (ds_gt_s_copy == novel_cls)
                novel_cls_feature_vector = (
                    features_s_copy * novel_cls_mask).mean(dim=(1, 3, 4)).squeeze()
                transportation_cost[novel_cls - self.num_base_classes_and_bg, base_cls] = torch.square(
                    base_cls_mean_feature_vector - novel_cls_feature_vector).sum()
        source = torch.ones(self.num_base_classes_and_bg, device=gt_s.device) / self.num_base_classes_and_bg
        target = torch.ones(self.num_novel_classes, device=gt_s.device) / self.num_novel_classes

        self.semantic_transports, _ = self.compute_optimal_transport(
            transportation_cost, target, source, 0.1)

        self.novel_weight = self.base_weight.squeeze() @ self.semantic_transports.T

        if zero_prototype:
            self.novel_weight = torch.zeros_like(self.novel_weight)
        assert torch.isnan(self.novel_weight).sum() == 0, self.novel_weight
        self.novel_bias = torch.zeros(
            (self.num_novel_classes,), device=features_s.device
        )

        # Copy prototypes for each task
        self.novel_weight = self.novel_weight.unsqueeze(0).repeat(
            self.n_tasks, 1, 1
        )  # [1, c, self.num_novel_classes]
        self.novel_bias = self.novel_bias.unsqueeze(
            0).repeat(self.n_tasks, 1)  # [1, self.num_novel_classes]

        self.transition_row_weight = torch.zeros(
            (features_s.size(2), self.row_size), device=features_s.device)
        init.normal_(self.transition_row_weight, mean=0, std=0.01)
        self.transition_row_bias = torch.zeros(
            self.row_size, device=features_s.device)

        self.transition_column_weight = torch.zeros(
            (features_s.size(2), self.col_size), device=features_s.device)
        init.normal_(self.transition_column_weight, mean=0, std=0.01)
        self.transition_column_bias = torch.zeros(
            self.col_size, device=features_s.device)

        self.transition_row_weight = self.transition_row_weight.unsqueeze(
            0).repeat(self.n_tasks, 1, 1)
        self.transition_row_bias = self.transition_row_bias.unsqueeze(
            0).repeat(self.n_tasks, 1)
        self.transition_column_weight = self.transition_column_weight.unsqueeze(
            0).repeat(self.n_tasks, 1, 1)
        self.transition_column_bias = self.transition_column_bias.unsqueeze(
            0).repeat(self.n_tasks, 1)

        self.layer_scale = torch.zeros(self.col_size, device=features_s.device)

    def get_transition_matrix(self) -> torch.Tensor:
        transition_matrix = torch.einsum(
            "fC,fR->fCR", self.transition_column_weight.squeeze(), self.transition_row_weight.squeeze()
        ) + self.transition_row_bias.unsqueeze(1) + self.transition_column_bias.unsqueeze(2)
        # return transition_matrix
        return transition_matrix.detach().clone()

    def get_classification_logits(self, features: torch.Tensor) -> torch.Tensor:
        # sourcery skip: inline-immediately-returned-variable
        """
        Computes old logits for given features

        inputs:
            features : shape [1 or batch_size_val, num_novel_classes * shot or 1, c, h, w]

        returns :
            logits : shape [batch_size_val, num_novel_classes * shot or 1, num_classes, h, w]
        """
        equation = "bochw,bcC->boChw"  # 'o' is n_novel_classes * shot for support and is 1 for query

        novel_logits = torch.einsum(equation, features, self.novel_weight)
        base_logits = torch.einsum(equation, features, self.base_weight)
        novel_logits += self.novel_bias.unsqueeze(1).unsqueeze(3).unsqueeze(4)
        base_logits += self.base_bias.unsqueeze(1).unsqueeze(3).unsqueeze(4)

        old_logits = torch.concat([base_logits, novel_logits], dim=2)

        return old_logits

    def get_transition_logits(self, features: torch.Tensor) -> torch.Tensor:
        """
        Computes old logits for given features

        inputs:
            features : shape [1 or batch_size_val, num_novel_classes * shot or 1, c, h, w]

        returns :
            logits : shape [batch_size_val, num_novel_classes * shot or 1, num_classes, h, w]
        """
        snapshot_logits = torch.einsum(
            "bochw,cC->boChw", features, self.snapshot_weight)
        transition_row = torch.einsum(
            "bochw,bcC->boChw", features, self.transition_row_weight
            ) + self.transition_row_bias.unsqueeze(1).unsqueeze(3).unsqueeze(4)
        transition_col = torch.einsum(
            "bochw,bcC->boChw", features, self.transition_column_weight
            ) + self.transition_column_bias.unsqueeze(1).unsqueeze(3).unsqueeze(4)
        transition_matrix = torch.einsum(
            "bochw,borhw->bocrhw", transition_col, transition_row)
        transition_logits = torch.einsum(
            "borchw,bochw->borhw", transition_matrix, snapshot_logits)
        transition_logits = torch.einsum(
            "bochw,c->bochw", transition_logits, self.layer_scale)
        self.transition_matrix=transition_matrix
        return transition_logits

    def get_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.get_classification_logits(features) + self.get_transition_logits(features)

    @staticmethod
    def get_probas(logits: torch.Tensor) -> torch.Tensor:
        """
        inputs:
            logits : shape [batch_size_val, num_novel_classes * shot or 1, num_classes, h, w]

        returns :
            probas : shape [batch_size_val, num_novel_classes * shot or 1, num_classes, h, w]
        """
        return torch.softmax(logits, dim=2)

    def get_base_snapshot_probas(self, features: torch.Tensor) -> torch.Tensor:
        """
        Computes probability maps for given query features, using the snapshot of the base model right after the
        training. It only computes values for base classes.

        inputs:
            features : shape [batch_size_val, 1, c, h, w]

        returns :
            probas : shape [batch_size_val, 1, num_base_classes_and_bg, h, w]
        """
        logits = torch.einsum(
            "bochw,cC->boChw", features, self.snapshot_weight
        ) + self.snapshot_bias.view(1, 1, -1, 1, 1)
        return torch.softmax(logits, dim=2)

    def self_estimate_pi(
        self, features_q: torch.Tensor, unsqueezed_valid_pixels_q: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimates pi using model's prototypes

        inputs:
            features_q : shape [batch_size_val, 1, c, h, w]
            unsqueezed_valid_pixels_q : shape [batch_size_val, 1, 1, h, w]

        returns :
            pi : shape [batch_size_val, num_classes]
        """
        logits_q = self.get_logits(features_q)
        # [1, 1, self.num_classes, h, w]
        probas = torch.softmax(logits_q, dim=2).detach()
        return self._valid_mean(probas, unsqueezed_valid_pixels_q, (1, 3, 4))

    def compute_pi(
        self, features_q: torch.Tensor, valid_pixels_q: torch.Tensor
    ) -> torch.Tensor:
        """
        inputs:
            features_q : shape [batch_size_val, 1, c, h, w]
            valid_pixels_q : shape [batch_size_val, 1, h, w]
        """
        valid_pixels_q = F.interpolate(
            valid_pixels_q.float(), size=features_q.size()[-2:], mode="nearest"
        ).long()
        valid_pixels_q = valid_pixels_q.unsqueeze(2)

        if self.pi_estimation_strategy == "self":
            self.pi = self.self_estimate_pi(
                features_q, valid_pixels_q)  # [1, self.num_classes]
        else:
            raise ValueError("pi_estimation_strategy is not implemented")

    def distillation_loss(
        self,
        curr_p: torch.Tensor,
        snapshot_p: torch.Tensor,
        valid_pixels: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        inputs:
            curr_p : shape [batch_size_val, 1, num_classes, h, w]
            snapshot_p : shape [batch_size_val, 1, num_base_classes_and_bg, h, w]
            valid_pixels : shape [batch_size_val, 1, h, w]

        returns:
             kl : Distillation loss for the query
        """
        adjusted_curr_p = curr_p.clone(
        )[:, :, : self.num_base_classes_and_bg, ...]
        adjusted_curr_p[:, :, 0, ...] += curr_p[
            :, :, self.num_base_classes_and_bg:, ...
        ].sum(dim=2)
        kl = (
            adjusted_curr_p *
            torch.log(1e-10 + adjusted_curr_p / (1e-10 + snapshot_p))
        ).sum(dim=2)
        kl = self._valid_mean(kl, valid_pixels, (1, 2, 3))
        if reduction == "sum":
            kl = kl.sum(0)
        elif reduction == "mean":
            kl = kl.mean(0)
        return kl

    def get_entropies(
        self, valid_pixels: torch.Tensor, probas: torch.Tensor, reduction: str = "mean"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sourcery skip: switch
        """
        inputs:
            valid_pixels: shape [batch_size_val, 1, h, w]
            probas : shape [batch_size_val, 1, num_classes, h, w]

        returns:
            d_kl : Classes proportion kl
            entropy : Entropy of predictions
            marginal : Current marginal distribution over labels [batch_size_val, num_classes]
        """
        entropy = -(probas * torch.log(probas + 1e-10)).sum(2)
        entropy = self._valid_mean(entropy, valid_pixels, (1, 2, 3))
        marginal = self._valid_mean(
            probas, valid_pixels.unsqueeze(2), (1, 3, 4))
        # entropy: [1]
        # marginal: [1, self.num_classes]
        # pi: [1, self.num_classes]

        d_kl = (marginal * torch.log(1e-10 + marginal / (self.pi + 1e-10))).sum(1)

        if reduction == "sum":
            entropy = entropy.sum(0)
            d_kl = d_kl.sum(0)
            assert not torch.isnan(entropy), entropy
            assert not torch.isnan(d_kl), d_kl
        elif reduction == "mean":
            entropy = entropy.mean(0)
            d_kl = d_kl.mean(0)
        return d_kl, entropy, marginal

    def get_LDAM_loss(self, logits: torch.Tensor, valid_pixels: torch.Tensor, one_hot_gt: torch.Tensor) -> torch.Tensor:
        # Statistically derived from the training and support sets
        m_list = torch.tensor([62655914 / 14, 43308472 / 14, 52174524 / 14, 2086042 / 14,
                              32711370 / 14, 17654415 / 14, 4659696 / 14, 33496325 / 14, 
                              11860, 34330, 39707, 85607], device=logits.device).float()
        m_list[:self.num_base_classes_and_bg] /= 14
        m_list[0] -= m_list[self.num_base_classes_and_bg:].sum()
        m_list[0] += m_list[1:self.num_base_classes_and_bg].sum()
        m_list /= m_list.sum()

        # LDAM loss
        max_m = 6
        m_list = 1.0 / torch.sqrt(torch.sqrt(m_list))
        m_list = m_list * (max_m / torch.max(m_list))

        batch_m: torch.Tensor = torch.einsum(
            "bochw,c->bochw", one_hot_gt, m_list
        )
        logits_m: torch.Tensor = logits - batch_m

        output = torch.where(one_hot_gt.bool(), logits_m, logits)
        return self.get_ce(self.get_probas(output), valid_pixels, one_hot_gt)

    def get_ce(
        self,
        probas: torch.Tensor,
        valid_pixels: torch.Tensor,
        one_hot_gt: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        inputs:
            probas : shape [batch_size_val, num_novel_classes * shot, c, h, w]
            valid_pixels : shape [1, num_novel_classes * shot, h, w]
            one_hot_gt: shape [1, num_novel_classes * shot, num_classes, h, w]

        returns:
             ce : Cross-Entropy between one_hot_gt and probas
        """
        probas = probas.clone()
        probas[:, :, 0, ...] += probas[:, :, 1: self.num_base_classes_and_bg, ...].sum(
            dim=2
        )
        probas[:, :, 1: self.num_base_classes_and_bg, ...] = 0.0

        ce = -(one_hot_gt * torch.log(probas + 1e-10))
        ce = (ce * compute_wce(one_hot_gt, self.num_novel_classes)).sum(2)
        ce = self._valid_mean(ce, valid_pixels, (1, 2, 3))  # [batch_size_val,]

        if reduction == "sum":
            ce = ce.sum(0)
        elif reduction == "mean":
            ce = ce.mean(0)
        return ce

    def get_confusion_loss(self) -> torch.Tensor:
        # [1 or batch_size_val, num_novel_classes * shot or 1, num_class, num_base_class, h, w]
        if self.transition_matrix is not None:
            logged = torch.log(self.transition_matrix)
            dot_re = torch.einsum(
                "borchw,borchw->borchw", logged, self.transition_matrix
            ) / self.num_base_classes_and_bg
            return torch.mean(dot_re)

    def similarity_loss(self, probas: torch.Tensor, valid_pixels: torch.Tensor) -> torch.Tensor:
        # proba: [1, 20, self.num_classes, h, w]
        # transition_matrix: [1, c, self.num_classes, self.num_base_classes_and_bg]
        transition_matrix = self.get_transition_matrix().mean(dim=1).squeeze()
        sl = torch.einsum(
            "bcRhw,CR->bcChw", probas[:, :, :self.num_base_classes_and_bg, ...], transition_matrix
        )
        return torch.sum(sl)

    def optimize(
        self, features_s: torch.Tensor, features_q: torch.Tensor, gt_s: torch.Tensor
    ) -> torch.Tensor:
        """
        Input:
            features_s : [num_novel_classes, shot, c, h, w]
            features_q : [batch_size_val, 1, c, h, w]
            gt_s : [num_novel_classes, shot, h, w]
            valid_pixels_q : [batch_size_val, 1, h, w]
        """
        l1, l2, l3, l4 = self.weights
        params = [self.novel_weight, self.novel_bias]
        if self.fine_tune_base_classifier:
            params.extend([self.base_weight, self.base_bias])

        params.extend([
            self.transition_column_weight, self.transition_column_bias,
            self.transition_row_weight, self.transition_row_bias,
            self.layer_scale
        ])
        for m in params:
            m.requires_grad_()
        optimizer = torch.optim.SGD(
            params, lr=self.lr, momentum=0.9, weight_decay=5e-4)

        # Flatten the dimensions of different novel classes and shots
        features_s = features_s.flatten(0, 1).unsqueeze(0)
        # [1, 20, h, w]
        gt_s = gt_s.flatten(0, 1).unsqueeze(0)

        ds_gt_s = F.interpolate(
            gt_s.float(), size=features_s.size()[-2:], mode="nearest"
        ).long()
        one_hot_gt_s = to_one_hot(
            ds_gt_s, self.num_classes
        )  # [1, num_novel_classes * shot, num_classes, h, w]
        valid_pixels_s = (ds_gt_s != 255).float()

        # Calculate initial pi
        self.compute_pi(features_q, valid_pixels_s)

        for iteration in range(self.adapt_iter):

            logits_s = self.get_transition_logits(
                features_s) + self.get_classification_logits(features_s)
            proba_s = self.get_probas(logits_s)

            logits_q = self.get_classification_logits(features_q)
            proba_q = self.get_probas(logits_q)

            snapshot_proba_q = self.get_base_snapshot_probas(features_q)

            # proba_s: [1, 20, self.num_classes, h, w]
            # proba_q: [1, 1, self.num_classes, h, w]
            # snapshot_proba_q: [1, 1, self.num_base_classes_and_bg, h, w]

            # Distillation Loss on base class to prevent catastrophic forgetting
            distillation = self.distillation_loss(
                proba_q, snapshot_proba_q, valid_pixels_s, reduction="none"
            )

            # pi-estimation to prevent overfitting on support set
            d_kl, entropy, _ = self.get_entropies(
                valid_pixels_s, proba_q, reduction="none"
            )

            # cross-entropy to learning on novel class
            if iteration < self.epoch_LDAM:
                ce = self.get_ce(proba_s, valid_pixels_s,
                                 one_hot_gt_s, reduction="none")
            else:
                ce = self.get_LDAM_loss(logits_s, valid_pixels_s, one_hot_gt_s)
            loss = l1 * ce  + l2 * entropy + l3 *  d_kl + l4 * distillation

            optimizer.zero_grad()
            loss.sum(0).backward()
            optimizer.step()

            # Update pi
            if (
                (iteration + 1) in self.pi_update_at
                and (self.pi_estimation_strategy == "self")
                and (l2 != 0)
            ):
                self.compute_pi(features_q, valid_pixels_s)