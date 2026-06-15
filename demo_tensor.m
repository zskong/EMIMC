% =========================================================================
% ===  Association and Consolidation: Evolutionary Memory-Enhanced Incremental Multi-View Clustering (Fixed version 2) ===
% =========================================================================
clear; clc; close all;

% --- 1. Load datasets ---
dataname = "GRAZ02"; 
load([strcat(dataname)]); 
cls_num = length(unique(Y));
gt = Y;
nV = length(X);

% nV_experiment = 5; 
% X = X(1:nV_experiment);
%
% nV = length(X);
% rng(10);
% shuffle_order = randperm(nV);
% X_shuffled = X(shuffle_order);
% X=X_shuffled;

% NormalizeData
for v=1:length(X)
    X{v} = NormalizeData(X{v}');
    X{v} = full(X{v});
end

% --- 2. Grid Search ---
latent_dims = [50];      % latent dimension
lambda_exps = [-3, -2];      % lambda = 10^x
gamma_exps  = [-3, -2];      % gamma = 10^y
% delta_fixed = 0.1;                    % fix delta=10^0=1
%ebbinghaus_lambdas = [0.5, 1, 1.5, 2]; % λ
% latent_dims = [50];     
% lambda_exps = [0];      % lambda = 10^x
% gamma_exps  = [-2];      % gamma = 10^y
delta_fixed = 0.1;                  
ebbinghaus_lambdas = [1]; % λ

% result record
results_log = [];
max_acc = 0;
best_params = struct();

fprintf('====== Grid search start! ======\n', dataname);

% --- main loop ---
for ld_idx = 1:length(latent_dims)
    latent_dim = latent_dims(ld_idx);
    
    for l_exp = lambda_exps
        lambda = 10^(l_exp);
        
        for g_exp = gamma_exps
            gamma = 10^(g_exp);
            
            % λ
            for eb_idx = 1:length(ebbinghaus_lambdas)
                eb_lambda = ebbinghaus_lambdas(eb_idx);
                
                fprintf('\n--- test: dim=%d, λ=%.1e, γ=%.1e, δ=%d, ebλ=%.1f ---\n',...
                        latent_dim, lambda, gamma, delta_fixed, eb_lambda);
                
                tic;
                % main function
                [Z_final, labels, times] = Incremental_Train4(X, cls_num, latent_dim,...
                                          lambda, gamma, delta_fixed, eb_lambda);
                time_cost = toc;
                
                % Evluation metrics
                [current_metrics] = Clustering8Measure(gt, labels);
                
                % Record the result
                results_log = [results_log; 
                             current_metrics, latent_dim, lambda, gamma,...
                             delta_fixed, eb_lambda, time_cost];
                
                fprintf('Result -> ACC: %.4f, NMI: %.4f, ARI: %.4f, Time: %.2fs\n',...
                        current_metrics(1), current_metrics(2), current_metrics(7), time_cost);
                
                % Update the best result
                if current_metrics(1) > max_acc
                    max_acc = current_metrics(1);
                    best_params = struct(...
                        'latent_dim', latent_dim,...
                        'lambda', lambda,...
                        'gamma', gamma,...
                        'delta', delta_fixed,...
                        'eb_lambda', eb_lambda);
                    best_result_metrics = current_metrics;
                    fprintf('!!! Find the new result ACC: %.4f !!!\n', max_acc);
                end
            end
        end
    end
end

% --- 3. save result ---
fprintf('\n====== dataset %s grid reserch done ! ======\n', dataname);
fprintf('Best ACC: %.4f\n', max_acc);
disp('Best parms:');
disp(best_params);

save_filename = sprintf('Result_Sequential_%s_WithEbbinghaus.mat', dataname);
save(save_filename, 'results_log', 'best_params', 'best_result_metrics');
fprintf('The results are saved in %s\n', save_filename);
