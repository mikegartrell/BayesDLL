import os
import copy
import time
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim.lr_scheduler
import collections # <-- Import collections for deque

import calibration

from methods.sghmc import Runner;
from methods.sghmc import Model;


#========================================================================================
# NEW REPULSIVE MODEL
#========================================================================================
class ModelRepulsive(Model):
    """
    SGHMC sampler model with Stein Self-Repulsive Dynamics.
    """
    def __init__(self, ND, prior_sig=1.0, bias='informative', momentum_decay=0.05, temperature=1.0, 
                 repulsive_alpha=1.0, repulsive_sigma=1.0, repulsive_sigma_adaptive=False):
        super().__init__(ND, prior_sig, bias, momentum_decay, temperature)
        # --- New parameters for repulsion ---
        self.repulsive_alpha = repulsive_alpha
        self.repulsive_sigma = repulsive_sigma

        # --- New flag to control adaptive sigma ---
        self.repulsive_sigma_adaptive = repulsive_sigma_adaptive

    def _calculate_adaptive_sigma(self, history_samples, device):
        """
        Calculates the adaptive RBF kernel bandwidth sigma = med^2 / log(M).
        """
        M = len(history_samples)
        # Need at least 2 samples to compute a distance
        if M < 2:
            return self.repulsive_sigma # Fallback to fixed sigma

        # --- Vectorized calculation of pairwise distances ---
        # Stack history vectors into a single tensor
        history_tensor = torch.stack(list(history_samples)).to(device)

        # Compute squared norms of each vector
        sum_sq = torch.sum(history_tensor * history_tensor, dim=1)
        
        # Compute pairwise squared L2 distances: ||x-y||^2 = ||x||^2 - 2<x,y> + ||y||^2
        # This uses broadcasting to compute the distance matrix efficiently.
        pairwise_sq_dists = sum_sq.unsqueeze(1) - 2 * torch.matmul(history_tensor, history_tensor.t()) + sum_sq.unsqueeze(0)
        
        # We only need the upper triangular part of the matrix (excluding the diagonal)
        # to get the unique pairwise distances.
        triu_indices = torch.triu_indices(M, M, offset=1)
        unique_sq_dists = pairwise_sq_dists[triu_indices[0], triu_indices[1]]
        
        # Clamp to avoid sqrt(0) -> NaN gradients if we were ever to backprop
        unique_sq_dists = unique_sq_dists.clamp(min=1e-12)
        
        # Median of the actual distances (not squared)
        med_dist = torch.sqrt(unique_sq_dists).median()
        med_sq = med_dist * med_dist

        # log(M) can't be zero, handle M=1 case (already handled by M < 2 check)
        # Use a small epsilon for numerical stability
        log_M = torch.log(torch.tensor(M, dtype=torch.float32, device=device))
        
        sigma = med_sq / (log_M + 1e-8)
        return sigma.item()

    def forward(self, x, y, net, net0, criterion, lrs, Ninflate=1.0, nd=1.0,
                # --- New arguments for repulsion ---
                apply_repulsion=False,
                history_samples=None):
        
        # --- 1. Standard SGHMC Step ---
        loss, out = super().forward(x, y, net, net0, criterion, lrs, Ninflate, nd)

        # --- 2. Stein Repulsive Step ---
        if apply_repulsion and history_samples and len(history_samples) > 0:
            with torch.no_grad():
                # Get current parameter vector
                current_theta_vec = nn.utils.parameters_to_vector(net.parameters())
                
                # --- Determine which sigma to use ---
                if self.repulsive_sigma_adaptive:
                    sigma = self._calculate_adaptive_sigma(history_samples, current_theta_vec.device)
                else:
                    sigma = self.repulsive_sigma

                total_repulsive_force = torch.zeros_like(current_theta_vec)

                # Calculate repulsive force from history
                for past_theta_vec in history_samples:
                    past_theta_vec = past_theta_vec.to(current_theta_vec.device)
                    
                    diff_vec = current_theta_vec - past_theta_vec
                    sq_dist = torch.sum(diff_vec * diff_vec)
                    
                    # RBF Kernel and its gradient w.r.t. the first argument (current_theta_vec)
                    # K(x, y) = exp(-||x-y||^2 / sigma)
                    # grad_x K(x, y) = K(x, y) * (-2 / sigma) * (x - y)
                    k_val = torch.exp(-sq_dist / sigma)
                    grad_k = k_val * (-2.0 / sigma) * diff_vec
                    
                    total_repulsive_force += grad_k
                
                # Average the force over all history samples
                total_repulsive_force /= len(history_samples)
                
                # Get the learning rate (use the body lr as the global step size for the force)
                lr_body = lrs[0]
                
                # Calculate the final repulsive update vector
                # This corresponds to: η * α * g(θ_k; δ_k^M)
                repulsive_update = lr_body * self.repulsive_alpha * total_repulsive_force
                
                # Apply the update to the network parameters
                updated_params_vec = nn.utils.parameters_to_vector(net.parameters()) + repulsive_update
                nn.utils.vector_to_parameters(updated_params_vec, net.parameters())

        return loss, out


#========================================================================================
# NEW REPULSIVE RUNNER
#========================================================================================
class RunnerRepulsive(Runner):
    """
    Runner for Self-Repulsive SGHMC.
    Manages the training loop and the history of samples for repulsion.
    """
    def __init__(self, net, net0, args, logger):
        super().__init__(net, net0, args, logger)
        hparams = args.hparams

        # --- Override the model with the repulsive version ---
        self.model = ModelRepulsive(
            ND=args.ND, 
            prior_sig=float(hparams['prior_sig']), 
            bias=str(hparams['bias']), 
            momentum_decay=float(hparams['momentum_decay']),
            temperature=float(hparams.get('temperature', 1.0)),
            # --- Pass repulsive hyperparameters to the model ---
            repulsive_alpha=float(hparams.get('replusive_alpha', 10.0)),
            repulsive_sigma=float(hparams.get('replusive_sigma', 100.0)),
            repulsive_sigma_adaptive=hparams.get('repulsive_sigma_adaptive', True)
        ).to(args.device)

        # --- Attributes for managing repulsion history ---
        self.repulsive_M = int(hparams.get('replusive_M', 10))
        self.repulsive_thinning = int(hparams.get('replusive_thinning', 100))
        self.repulsive_burnin_steps = self.repulsive_M * self.repulsive_thinning
        
        # Use a deque to automatically manage the history size
        self.history_samples = collections.deque(maxlen=self.repulsive_M)
        logger.info(f"Initialized Runner for Self-Repulsive SGHMC.")
        logger.info(f"History size (M): {self.repulsive_M}, Thinning: {self.repulsive_thinning}")
        if self.model.repulsive_sigma_adaptive:
            logger.info("Using ADAPTIVE kernel bandwidth sigma.")
        else:
            logger.info(f"Using FIXED kernel bandwidth sigma: {self.model.repulsive_sigma}")
        logger.info(f"Repulsion will start after {self.repulsive_burnin_steps} batch iterations.")

    def train_one_epoch(self, train_loader, collect, bi):
        """
        Run Self-Repulsive SGHMC steps for one epoch.
        """
        args = self.args
        logger = self.logger

        self.net.train()
        
        loss, error, nb_samples = 0, 0, 0
        with tqdm(train_loader, unit="batch") as tepoch:
            for x, y in tepoch:
                x, y = x.to(args.device), y.to(args.device)

                # Determine if repulsion should be applied in this step
                apply_repulsion = (bi > self.repulsive_burnin_steps)

                # Evaluate SGHMC updates for a given a batch
                loss_, out = self.model(
                    x, y, self.net, self.net0, self.criterion, 
                    [pg['lr'] for pg in self.optimizer.param_groups],
                    self.Ninflate, self.nd,
                    # --- Pass repulsive-specific arguments to the model ---
                    apply_repulsion=apply_repulsion,
                    history_samples=self.history_samples
                )
                
                # Prediction on training
                pred = out.data.max(dim=1)[1]
                err = pred.ne(y.data).sum()

                loss += loss_ * len(y)
                error += err.item()
                nb_samples += len(y)

                bi += 1

                # --- Update the history of samples for repulsion ---
                if bi % self.repulsive_thinning == 0:
                    with torch.no_grad():
                        # Store a clone of the parameter vector on the CPU to save GPU memory
                        self.history_samples.append(
                            nn.utils.parameters_to_vector(self.net.parameters()).clone().cpu()
                        )

                # If post-burnin, collect posterior samples every thinning steps
                if collect and bi % self.thin == 0:
                    logger.info('(post-burnin) accumulate posterior samples')
                    with torch.no_grad():
                        theta_vec = nn.utils.parameters_to_vector(self.net.parameters())
                        self.post_theta_mom1 = (theta_vec + self.post_theta_cnt * self.post_theta_mom1) / (self.post_theta_cnt + 1)
                        if self.nst > 0:
                            self.post_theta_mom2 = (theta_vec**2 + self.post_theta_cnt * self.post_theta_mom2) / (self.post_theta_cnt + 1)
                    self.post_theta_cnt += 1

                tepoch.set_postfix(loss=loss/nb_samples, error=error/nb_samples)

                self.scheduler.step()

        return loss/nb_samples, error/nb_samples, bi
