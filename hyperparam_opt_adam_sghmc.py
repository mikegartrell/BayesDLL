import os
import sys
import logging
import argparse
import json
import numpy as np
import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import tempfile
import shutil
from datetime import datetime
import flash.core.optimizers
import time

# Import the necessary modules from your codebase
import networks
import datasets
import utils
from methods.adam_sghmc import Runner


def create_objective_function(base_args):
    """
    Creates an objective function for Optuna optimization.
    
    Args:
        base_args: Base arguments with fixed hyperparameters
    
    Returns:
        objective function for Optuna
    """
    
    def objective(trial):
        # Create a copy of base arguments
        args = argparse.Namespace(**vars(base_args))
        
        # Sample hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        momentum_decay = trial.suggest_float('momentum_decay', 0.001, 0.1, log=True)  # This is momentum_decay in Adam-SGHMC
        temperature = trial.suggest_float('temperature', 0.01, 1.0)
        prior_sig = trial.suggest_float('prior_sigma', 0.01, 1.0)
        
        # Update learning rate
        args.lr = lr
        args.lr_head = lr  # Keep head lr same as body lr for simplicity
        
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
        
        # Create temporary directory for this trial
        # trial_dir = tempfile.mkdtemp(prefix=f'optuna_trial_{trial.number}_')
        trial_dir = os.path.join(base_args.results_dir, f'optuna_trial_{args.study_name}_{trial.number}')
        os.makedirs(trial_dir, exist_ok=True)
        args.log_dir = trial_dir
        
        try:
            # Setup logging for this trial (suppress most output)
            logging.basicConfig(
                handlers=[logging.FileHandler(os.path.join(trial_dir, 'trial_log.txt'))],
                format='[%(asctime)s,%(msecs)03d %(levelname)s] %(message)s',
                datefmt='%H:%M:%S',
                level=logging.INFO,  # Reduce logging verbosity
                force=True
            )
            logger = logging.getLogger()
            
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
            
            # Create runner and train
            runner = Runner(net, net0, args, logger)
            
            # Store original train method to add early stopping
            original_train = runner.train
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 10  # Early stopping patience
            
            def train_with_pruning(train_loader, val_loader, test_loader):
                nonlocal best_val_loss, patience_counter
                
                # Reduce number of epochs for hyperparameter search
                original_epochs = args.epochs
                # args.epochs = min(args.epochs, 75)  # Cap at 75 epochs for faster search
                
                num_epochs = args.epochs
                steps_per_epoch = len(train_loader)
                total_steps = num_epochs * steps_per_epoch
                logger.info('Total training steps: %d' % total_steps)

                num_warmup_epochs = args.warmup_epochs
                warmup_steps = num_warmup_epochs * steps_per_epoch
                logger.info('Total warmup steps: %d' % warmup_steps)

                runner.scheduler = flash.core.optimizers.LinearWarmupCosineAnnealingLR(runner.optimizer, warmup_epochs=warmup_steps, max_epochs=total_steps)

                # Call original training loop with modifications for early stopping
                losses_train = np.zeros(args.epochs)
                errors_train = np.zeros(args.epochs)
                if val_loader is not None:
                    losses_val = np.zeros(args.epochs)
                    errors_val = np.zeros(args.epochs)
                
                # runner.net.train()
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
                        # Use test loss if no validation set
                        test_loss, test_error = runner.compute_validation_metrics(test_loader)
                        current_val_loss = test_loss

                    toc = time.time()

                    prn_str = '[Epoch %d/%d] Training summary: ' % (ep, args.epochs)
                    prn_str += 'loss = %.4f, prediction error = %.4f, ' % (losses_train[ep], errors_train[ep])
                    prn_str += 'val loss = %.4f, val prediction error = %.4f, ' % (losses_val[ep], errors_val[ep])
                    prn_str += 'lr = %.7f ' % runner.scheduler.get_last_lr()[0]
                    prn_str += '(time: %.4f seconds)' % (toc-tic,)
                    logger.info(prn_str)
                    
                    # Report intermediate value for pruning
                    trial.report(current_val_loss, ep)
                    
                    # Check if trial should be pruned
                    if trial.should_prune():
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
            
            return final_loss
            
        except optuna.exceptions.TrialPruned:
            raise  # Re-raise pruned exception
        except Exception as e:
            # Log the error and return a high loss
            print(f"Trial {trial.number} failed with error: {str(e)}")
            return float('inf')
        # finally:
        #     # Clean up temporary directory
        #     try:
        #         shutil.rmtree(trial_dir)
        #     except:
        #         pass
    
    return objective


def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter optimization for Adam-SGHMC')
    
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
    
    args = parser.parse_args()
    
    # Ensure method is adam_sghmc
    if args.method != 'adam_sghmc':
        raise ValueError("This optimization script is specifically for adam_sghmc method")
    
    # Set device
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        args.device = torch.device('mps')
    else:
        args.device = torch.device('cpu')
    
    args.use_cuda = torch.cuda.is_available()
    
    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set up Optuna study
    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction='minimize',
            sampler=TPESampler(seed=args.seed),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            load_if_exists=True
        )
    else:
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=args.seed),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
    
    # Create objective function
    objective = create_objective_function(args)
    
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
    
    logger.info(f"Starting Optuna optimization with {args.n_trials} trials")
    logger.info(f"Base arguments: {args}")
    
    # Run optimization
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    
    # Print results
    logger.info("Optimization completed!")
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
    best_hparams += f"momentum_decay={study.best_params['momentum']}"
    best_hparams += f",temperature={study.best_params['temperature']}"
    
    logger.info("\nTo run with best parameters, use:")
    logger.info(f"python demo_vision.py --method adam_sghmc --lr {study.best_params['lr']:.6f} "
                f"--hparams \"{best_hparams}\" --dataset {args.dataset} --backbone {args.backbone}")


if __name__ == '__main__':
    main()
