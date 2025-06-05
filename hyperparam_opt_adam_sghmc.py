import os
import sys
import logging
import argparse
import json
import numpy as np
import torch
import torch.multiprocessing as mp
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import tempfile
import shutil
from datetime import datetime
import flash.core.optimizers
import time
import threading
from typing import Optional, List
import psutil

# Import the necessary modules from your codebase
import networks
import datasets
import utils
from methods.adam_sghmc import Runner


class GPUManager:
    """Manages GPU allocation across processes."""
    
    def __init__(self):
        self.available_devices = self._get_available_devices()
        self.device_lock = threading.Lock()
        self.device_queue = list(self.available_devices)
        
    def _get_available_devices(self) -> List[torch.device]:
        """Get list of available devices (CUDA, MPS, CPU)."""
        devices = []
        
        # Add CUDA devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(torch.device(f'cuda:{i}'))
        
        # Add MPS device if available
        elif torch.backends.mps.is_available():
            devices.append(torch.device('mps'))
        
        # Fallback to CPU
        if not devices:
            devices.append(torch.device('cpu'))
            
        return devices
    
    def get_device(self) -> torch.device:
        """Get an available device for training."""
        with self.device_lock:
            if self.device_queue:
                return self.device_queue.pop(0)
            else:
                # If no devices available, cycle through them
                return self.available_devices[len(self.device_queue) % len(self.available_devices)]
    
    def return_device(self, device: torch.device):
        """Return a device to the available pool."""
        with self.device_lock:
            if device not in self.device_queue:
                self.device_queue.append(device)


# Global GPU manager instance
gpu_manager = None


def setup_process_device(base_args, trial_number: int) -> torch.device:
    """Setup device for a specific process/trial."""
    global gpu_manager
    
    if gpu_manager is None:
        gpu_manager = GPUManager()
    
    device = gpu_manager.get_device()
    
    # Set environment variables for this process
    if device.type == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device.index)
        # Set device to cuda:0 since we're using CUDA_VISIBLE_DEVICES
        device = torch.device('cuda:0')
    
    return device


def create_objective_function(base_args, use_multiprocessing: bool = True):
    """
    Creates an objective function for Optuna optimization.
    
    Args:
        base_args: Base arguments with fixed hyperparameters
        use_multiprocessing: Whether to use multiprocessing-safe practices
    
    Returns:
        objective function for Optuna
    """
    
    def objective(trial):
        try:
            # Setup device for this trial
            device = setup_process_device(base_args, trial.number)
            
            # Create a copy of base arguments
            args = argparse.Namespace(**vars(base_args))
            args.device = device
            args.use_cuda = device.type == 'cuda'
            
            # Set process-specific random seed
            process_seed = base_args.seed + trial.number
            torch.manual_seed(process_seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed(process_seed)
            torch.backends.cudnn.deterministic = True
            np.random.seed(process_seed)
            
            # Sample hyperparameters
            lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
            momentum_decay = trial.suggest_float('momentum_decay', 0.001, 0.1, log=True)
            temperature = trial.suggest_float('temperature', 0.01, 1.0)
            prior_sig = trial.suggest_float('prior_sigma', 0.01, 1.0)
            
            # Update learning rate
            args.lr = lr
            args.lr_head = lr
            
            # Update hparams string with new hyperparameters
            hparams_dict = {}
            if args.hparams:
                # Parse existing hparams
                opts = args.hparams.split(',')
                for opt in opts:
                    if '=' in opt:
                        key, val = opt.split('=')
                        hparams_dict[key] = val
            
            # Update with trial hyperparameters
            hparams_dict['momentum_decay'] = str(momentum_decay)
            hparams_dict['temperature'] = str(temperature)
            hparams_dict['prior_sig'] = str(prior_sig)
            
            # Reconstruct hparams string
            hparams_list = [f"{k}={v}" for k, v in hparams_dict.items()]
            args.hparams = ','.join(hparams_list)
            
            # Parse hparams for the runner
            hparams = {}
            for opt in hparams_list:
                if '=' in opt:
                    key, val = opt.split('=')
                    hparams[key] = val
            args.hparams = hparams
            
            # Create trial-specific directory
            trial_dir = os.path.join(base_args.results_dir, f'optuna_trial_{args.study_name}_{trial.number}')
            os.makedirs(trial_dir, exist_ok=True)
            args.log_dir = trial_dir
            
            # Setup logging for this trial
            log_file = os.path.join(trial_dir, 'trial_log.txt')
            logger = logging.getLogger(f'trial_{trial.number}')
            logger.handlers.clear()  # Clear any existing handlers
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('[%(asctime)s,%(msecs)03d %(levelname)s] %(message)s', datefmt='%H:%M:%S')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False  # Prevent propagation to avoid conflicts
            
            logger.info(f"Trial {trial.number} starting on device: {device}")
            logger.info(f"Trial parameters: lr={lr:.6f}, momentum_decay={momentum_decay:.6f}, temperature={temperature:.3f}, prior_sig={prior_sig:.3f}")
            
            # Prepare data
            train_loader, val_loader, test_loader, args.ND = datasets.prepare(
                args, train_data_aug=args.train_data_aug
            )
            
            # Create networks
            net = networks.create_backbone(args)
            if args.pretrained is not None:
                net0 = networks.load_pretrained_backbone(args)
                net = networks.load_pretrained_backbone(args, zero_head=False)
            else:
                net0 = None
            
            # Move networks to device
            net = net.to(device)
            if net0 is not None:
                net0 = net0.to(device)
            
            # Create runner and train
            runner = Runner(net, net0, args, logger)
            
            # Enhanced training with pruning and early stopping
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 10
            
            def train_with_pruning(train_loader, val_loader, test_loader):
                nonlocal best_val_loss, patience_counter
                
                num_epochs = args.epochs
                steps_per_epoch = len(train_loader)
                total_steps = num_epochs * steps_per_epoch
                logger.info('Total training steps: %d' % total_steps)

                num_warmup_epochs = args.warmup_epochs
                warmup_steps = num_warmup_epochs * steps_per_epoch
                logger.info('Total warmup steps: %d' % warmup_steps)

                runner.scheduler = flash.core.optimizers.LinearWarmupCosineAnnealingLR(
                    runner.optimizer, warmup_epochs=warmup_steps, max_epochs=total_steps
                )

                losses_train = np.zeros(args.epochs)
                errors_train = np.zeros(args.epochs)
                if val_loader is not None:
                    losses_val = np.zeros(args.epochs)
                    errors_val = np.zeros(args.epochs)
                
                bi = 0
                
                for ep in range(args.epochs):
                    tic = time.time()

                    # Train one epoch
                    collect = ep >= runner.burnin
                    losses_train[ep], errors_train[ep], bi = runner.train_one_epoch(
                        train_loader, collect=collect, bi=bi
                    )
                    
                    # Validation
                    if val_loader is not None:
                        losses_val[ep], errors_val[ep] = runner.compute_validation_metrics(val_loader)
                        current_val_loss = losses_val[ep]
                    else:
                        test_loss, test_error = runner.compute_validation_metrics(test_loader)
                        current_val_loss = test_loss

                    toc = time.time()

                    if ep % 5 == 0 or ep == args.epochs - 1:  # Log every 5 epochs to reduce output
                        prn_str = '[Epoch %d/%d] Training summary: ' % (ep, args.epochs)
                        prn_str += 'loss = %.4f, prediction error = %.4f, ' % (losses_train[ep], errors_train[ep])
                        if val_loader is not None:
                            prn_str += 'val loss = %.4f, val prediction error = %.4f, ' % (losses_val[ep], errors_val[ep])
                        prn_str += 'lr = %.7f ' % runner.scheduler.get_last_lr()[0]
                        prn_str += '(time: %.4f seconds)' % (toc-tic,)
                        logger.info(prn_str)
                    
                    # Report intermediate value for pruning
                    trial.report(current_val_loss, ep)
                    
                    # Check if trial should be pruned
                    if trial.should_prune():
                        logger.info(f"Trial {trial.number} pruned at epoch {ep}")
                        raise optuna.exceptions.TrialPruned()
                    
                    # Early stopping
                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(f"Early stopping at epoch {ep}")
                            break
                
                return best_val_loss
            
            # Replace train method
            runner.train = train_with_pruning
            
            # Train the model
            final_loss = runner.train(train_loader, val_loader, test_loader)
            
            logger.info(f"Trial {trial.number} completed with loss: {final_loss:.6f}")
            
            # Clean up GPU memory
            del net, runner
            if net0 is not None:
                del net0
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            
            return final_loss
            
        except optuna.exceptions.TrialPruned:
            logger.info(f"Trial {trial.number} was pruned")
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {str(e)}")
            print(f"Trial {trial.number} failed with error: {str(e)}")
            return float('inf')
        finally:
            # Return device to pool
            global gpu_manager
            if gpu_manager and 'device' in locals():
                gpu_manager.return_device(device)
    
    return objective


def get_optimal_n_jobs(args) -> int:
    """Determine optimal number of parallel jobs based on available resources."""
    
    # Get number of available devices
    n_devices = 0
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        n_devices = 1
    else:
        n_devices = 1  # CPU
    
    # Get number of CPU cores
    n_cpus = psutil.cpu_count(logical=False) or 1
    
    # If using GPUs, limit by GPU count, otherwise by CPU count
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        optimal_jobs = min(n_devices, args.n_trials)
    else:
        # For CPU-only, use fewer processes to avoid memory issues
        optimal_jobs = min(max(1, n_cpus // 2), args.n_trials)
    
    # Allow user override
    if hasattr(args, 'n_jobs') and args.n_jobs > 0:
        optimal_jobs = min(args.n_jobs, args.n_trials)
    
    return optimal_jobs


def main():
    # Set multiprocessing start method
    if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
        mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='Parallel Optuna hyperparameter optimization for Adam-SGHMC')
    
    # Add all the original arguments
    parser.add_argument('--method', type=str, default='adam_sghmc', help='method (should be adam_sghmc)')
    parser.add_argument('--hparams', type=str, default='', help='base hparams (will be extended with optimized params)')
    parser.add_argument('--pretrained', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset name')
    parser.add_argument('--backbone', type=str, default='mlp', help='backbone name')
    parser.add_argument('--val_heldout', type=float, default=0.1, help='validation set heldout proportion')
    parser.add_argument('--ece_num_bins', type=int, default=15, help='number of bins for error calibration')
    parser.add_argument('--epochs', type=int, default=100, help='max number of training epochs per trial')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='number of warmup epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--test_eval_freq', type=int, default=1, help='test evaluation frequency')
    parser.add_argument('--train_data_aug', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--momentum', type=float, default=0, help='SGD optimizer momentum')
    
    # Optuna-specific arguments
    parser.add_argument('--n_trials', type=int, default=100, help='number of optimization trials')
    parser.add_argument('--study_name', type=str, default='adam_sghmc_optimization', help='study name')
    parser.add_argument('--storage', type=str, default=None, help='database URL for study storage (optional)')
    parser.add_argument('--results_dir', type=str, default='optuna_results', help='directory to save results')
    
    # Parallelization arguments
    parser.add_argument('--n_jobs', type=int, default=-1, help='number of parallel jobs (-1 for auto, 1 for sequential)')
    parser.add_argument('--timeout', type=int, default=3600, help='timeout per trial in seconds')
    
    args = parser.parse_args()
    
    # Ensure method is adam_sghmc
    if args.method != 'adam_sghmc':
        raise ValueError("This optimization script is specifically for adam_sghmc method")
    
    # Determine optimal number of jobs
    if args.n_jobs == -1:
        args.n_jobs = get_optimal_n_jobs(args)
    elif args.n_jobs == 0:
        args.n_jobs = 1
    
    # Initialize global GPU manager
    global gpu_manager
    gpu_manager = GPUManager()
    
    print(f"Available devices: {gpu_manager.available_devices}")
    print(f"Running optimization with {args.n_jobs} parallel jobs")
    
    # Set initial device info (will be overridden per process)
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        args.device = torch.device('mps')
    else:
        args.device = torch.device('cpu')
    
    args.use_cuda = torch.cuda.is_available()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set up Optuna study with database for parallel execution
    if args.storage is None and args.n_jobs > 1:
        # Create SQLite database for parallel trials
        args.storage = f'sqlite:///{os.path.join(args.results_dir, "optuna_study.db")}'
    
    study_kwargs = {
        'direction': 'minimize',
        'sampler': TPESampler(seed=args.seed),
        'pruner': MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    }
    
    if args.storage:
        study_kwargs.update({
            'study_name': args.study_name,
            'storage': args.storage,
            'load_if_exists': True
        })
        study = optuna.create_study(**study_kwargs)
    else:
        study = optuna.create_study(**study_kwargs)
    
    # Create objective function
    objective = create_objective_function(args, use_multiprocessing=args.n_jobs > 1)
    
    # Setup logging
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(args.results_dir, 'optimization.log')),
            logging.StreamHandler()
        ],
        format='[%(asctime)s] %(levelname)s: %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger()
    
    logger.info(f"Starting Optuna optimization with {args.n_trials} trials using {args.n_jobs} parallel jobs")
    logger.info(f"Available devices: {[str(d) for d in gpu_manager.available_devices]}")
    logger.info(f"Base arguments: {args}")
    
    # Run optimization
    start_time = time.time()
    
    if args.n_jobs == 1:
        # Sequential execution
        study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=True)
    else:
        # Parallel execution
        study.optimize(
            objective, 
            n_trials=args.n_trials, 
            n_jobs=args.n_jobs,
            timeout=args.timeout,
            show_progress_bar=True
        )
    
    end_time = time.time()
    
    # Print results
    logger.info("Optimization completed!")
    logger.info(f"Total time: {end_time - start_time:.2f} seconds")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value:.6f}")
    logger.info(f"Best parameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    # Save results
    results = {
        'best_trial': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'optimization_time': end_time - start_time,
        'n_jobs': args.n_jobs,
        'devices_used': [str(d) for d in gpu_manager.available_devices],
        'all_trials': []
    }
    
    for trial in study.trials:
        trial_data = {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params,
            'state': trial.state.name
        }
        results['all_trials'].append(trial_data)
    
    results_file = os.path.join(args.results_dir, 'optimization_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Generate optimization plots if optuna visualization is available
    try:
        import optuna.visualization as vis
        
        # Plot optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(os.path.join(args.results_dir, 'optimization_history.html'))
        
        # Plot parameter importances
        if len(study.trials) > 10:  # Need sufficient trials for importance analysis
            fig = vis.plot_param_importances(study)
            fig.write_html(os.path.join(args.results_dir, 'param_importances.html'))
        
        # Plot parameter relationships
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(os.path.join(args.results_dir, 'parallel_coordinate.html'))
        
        logger.info("Visualization plots saved in results directory")
        
    except ImportError:
        logger.warning("optuna[visualization] not installed. Skipping plot generation.")
    
    # Print command to run with best parameters
    best_hparams = args.hparams
    if best_hparams:
        best_hparams += ','
    best_hparams += f"momentum_decay={study.best_params['momentum_decay']:.6f}"
    best_hparams += f",temperature={study.best_params['temperature']:.6f}"
    best_hparams += f",prior_sig={study.best_params['prior_sigma']:.6f}"
    
    logger.info("\nTo run with best parameters, use:")
    logger.info(f"python demo_vision.py --method adam_sghmc --lr {study.best_params['lr']:.6f} "
                f"--hparams \"{best_hparams}\" --dataset {args.dataset} --backbone {args.backbone}")


if __name__ == '__main__':
    main()