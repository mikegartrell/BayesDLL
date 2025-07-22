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
                 repulsive_alpha=1.0, repulsive_sigma=1.0, repulsive_sigma_adaptive=False, 
                 repulsive_stale_steps=1):
        super().__init__(ND, prior_sig, bias, momentum_decay, temperature)
        # --- New parameters for repulsion ---
        self.repulsive_alpha = repulsive_alpha
        self.repulsive_sigma = repulsive_sigma

        # --- New flag to control adaptive sigma ---
        self.repulsive_sigma_adaptive = repulsive_sigma_adaptive

        # Stale force optimization parameter and state for calculation
        self.repulsive_stale_steps = repulsive_stale_steps
        self.stale_force_cache = None
        self.step_counter = 0

    def _calculate_adaptive_sigma(self, history_samples_tensor, M):
        """
        Calculates the adaptive RBF kernel bandwidth sigma = med^2 / log(M).
        - Assumes history_samples_tensor is already on the correct device.
        """
        if M < 2:
            return self.repulsive_sigma # Fallback to fixed sigma

        sum_sq = torch.sum(history_samples_tensor * history_samples_tensor, dim=1)
        pairwise_sq_dists = sum_sq.unsqueeze(1) - 2 * torch.matmul(history_samples_tensor, history_samples_tensor.t()) + sum_sq.unsqueeze(0)
        
        triu_indices = torch.triu_indices(M, M, offset=1)
        unique_sq_dists = pairwise_sq_dists[triu_indices[0], triu_indices[1]]
        unique_sq_dists = unique_sq_dists.clamp(min=1e-12)
        
        med_dist = torch.sqrt(unique_sq_dists).median()
        med_sq = med_dist * med_dist

        log_M = torch.log(torch.tensor(M, dtype=torch.float32, device=history_samples_tensor.device))
        
        sigma = med_sq / (log_M + 1e-8)
        return sigma.item()
    
    def _calculate_repulsive_force_vectorized(self, current_theta_vec, history_samples):
        """
        Calculates the Stein repulsive force in a fully vectorized manner.
        """
        M = len(history_samples)
        device = current_theta_vec.device

        # 1. Bulk Transfer: Move all history from CPU to GPU at once.
        history_tensor = torch.stack(list(history_samples)).to(device)

        # 2. Determine Sigma: Use adaptive or fixed sigma.
        if self.repulsive_sigma_adaptive:
            sigma = self._calculate_adaptive_sigma(history_tensor, M)
        else:
            sigma = self.repulsive_sigma
        
        # 3. Vectorized Computation of Repulsive Force
        # Compute all differences between current and historical points via broadcasting.
        # Shape: (1, D) - (M, D) -> (M, D)
        diff_matrix = current_theta_vec.unsqueeze(0) - history_tensor

        # Compute all squared L2 distances.
        # Shape: (M,)
        sq_dists = torch.sum(diff_matrix * diff_matrix, dim=1)

        # Compute all RBF kernel values.
        # Shape: (M,)
        k_vals = torch.exp(-sq_dists / sigma)

        # Compute the final force vector.
        # grad_k = k_val * (-2 / sigma) * diff_vec
        # We want sum(grad_k) / M
        # This is equivalent to: (-2 / sigma) * sum(k_val * diff_vec) / M
        
        # Reshape k_vals to (M, 1) for broadcasting with diff_matrix (M, D)
        # Resulting shape is (M, D), where each row is correctly scaled.
        scaled_diffs = k_vals.unsqueeze(1) * diff_matrix
        
        # Sum across all historical points (dimension 0) to get the total force vector.
        total_force = torch.sum(scaled_diffs, dim=0)
        
        # Apply the remaining scaling factors and average over M.
        # Note: The (-2 / sigma) is part of the gradient calculation.
        final_force = total_force * (-2.0 / sigma) / M
        
        return final_force

    def forward(self, x, y, net, net0, criterion, lrs, Ninflate=1.0, nd=1.0,
                apply_repulsion=False,
                history_samples=None):
        self.step_counter += 1

        # --- 1. Standard SGHMC Step (Unchanged) ---
        loss, out = super().forward(x, y, net, net0, criterion, lrs, Ninflate, nd)

        # --- 2. Stein Repulsive Step (Optimized) ---
        if apply_repulsion and history_samples and len(history_samples) > 0:
            with torch.no_grad():
                # Only compute the repulsive force if it's the first step or stale interval is reached.
                if self.step_counter % self.repulsive_stale_steps == 0 or self.stale_force_cache is None:
                    current_theta_vec = nn.utils.parameters_to_vector(net.parameters())
                    
                    ## --- OPTIMIZATION ---
                    # Call the new vectorized function instead of the slow loop.
                    repulsive_force_vec = self._calculate_repulsive_force_vectorized(
                        current_theta_vec, history_samples
                    )

                    # Cache the computed force
                    self.stale_force_cache = repulsive_force_vec                        
                else:
                    # Use the cached force
                    repulsive_force_vec = self.stale_force_cache.to(net.parameters().__next__().device)

                lr_body = lrs[0]
                repulsive_update = lr_body * self.repulsive_alpha * repulsive_force_vec
                
                # Apply the update to the network parameters
                # We add to `current_theta_vec` which is a *copy*, then load it back.
                # This is safer than in-place modification on `net.parameters()`.
                current_theta_vec = nn.utils.parameters_to_vector(net.parameters())
                updated_params_vec = current_theta_vec + repulsive_update
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
            repulsive_alpha=float(hparams.get('repulsive_alpha', 10.0)),
            repulsive_sigma=float(hparams.get('repulsive_sigma', 100.0)),
            repulsive_sigma_adaptive=hparams.get('repulsive_sigma_adaptive', True),
            repulsive_stale_steps=int(hparams.get('repulsive_stale_steps', 1))
        ).to(args.device)

        # --- Attributes for managing repulsion history ---
        self.repulsive_M = int(hparams.get('repulsive_M', 10))
        self.repulsive_thinning = int(hparams.get('repulsive_thinning', 100))
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
        logger.info(f"Stale Force Calculation (every K steps): {self.model.repulsive_stale_steps}")

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
