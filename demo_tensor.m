% =========================================================================
% ===  序列化多视图学习模型测试脚本 (修正版) ===
% =========================================================================
clear; clc; close all;

% --- 1. 加载数据集 ---
dataname = "GRAZ02"; % 修改为有效的字符串，去掉方括号
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
% 数据归一化
for v=1:length(X)
    X{v} = NormalizeData(X{v}');
    X{v} = full(X{v});
end

% --- 2. 参数网格搜索设置 ---
latent_dims = [50];      % 潜在维度
lambda_exps = [-3, -2];      % lambda = 10^x
gamma_exps  = [-3, -2];      % gamma = 10^y
% delta_fixed = 0.1;                    % 固定delta=10^0=1
%ebbinghaus_lambdas = [0.5, 1, 1.5, 2]; % 遗忘系数λ
% latent_dims = [50];      % 潜在维度
% lambda_exps = [0];      % lambda = 10^x
% gamma_exps  = [-2];      % gamma = 10^y
delta_fixed = 0.1;                    % 固定delta=10^0=1
ebbinghaus_lambdas = [1]; % 遗忘系数λ

% 结果记录
results_log = [];
max_acc = 0;
best_params = struct();

fprintf('====== 开始在数据集 %s 上进行网格搜索 ======\n', dataname);

% --- 网格搜索主循环 ---
for ld_idx = 1:length(latent_dims)
    latent_dim = latent_dims(ld_idx);
    
    for l_exp = lambda_exps
        lambda = 10^(l_exp);
        
        for g_exp = gamma_exps
            gamma = 10^(g_exp);
            
            % 艾宾浩斯λ循环
            for eb_idx = 1:length(ebbinghaus_lambdas)
                eb_lambda = ebbinghaus_lambdas(eb_idx);
                
                fprintf('\n--- 测试参数: dim=%d, λ=%.1e, γ=%.1e, δ=%d, ebλ=%.1f ---\n',...
                        latent_dim, lambda, gamma, delta_fixed, eb_lambda);
                
                tic;
                % 调用增量学习函数
                [Z_final, labels, times] = Incremental_Train4(X, cls_num, latent_dim,...
                                          lambda, gamma, delta_fixed, eb_lambda);
                time_cost = toc;
                
                % 评估聚类结果
                [current_metrics] = Clustering8Measure(gt, labels);
                
                % 记录结果（新增eb_lambda列）
                results_log = [results_log; 
                             current_metrics, latent_dim, lambda, gamma,...
                             delta_fixed, eb_lambda, time_cost];
                
                fprintf('结果 -> ACC: %.4f, NMI: %.4f, ARI: %.4f, Time: %.2fs\n',...
                        current_metrics(1), current_metrics(2), current_metrics(7), time_cost);
                
                % 更新最佳结果
                if current_metrics(1) > max_acc
                    max_acc = current_metrics(1);
                    best_params = struct(...
                        'latent_dim', latent_dim,...
                        'lambda', lambda,...
                        'gamma', gamma,...
                        'delta', delta_fixed,...
                        'eb_lambda', eb_lambda);
                    best_result_metrics = current_metrics;
                    fprintf('!!! 发现新的最佳ACC: %.4f !!!\n', max_acc);
                end
            end
        end
    end
end

% --- 3. 保存结果 ---
fprintf('\n====== 数据集 %s 搜索完毕 ======\n', dataname);
fprintf('最佳ACC: %.4f\n', max_acc);
disp('最佳参数组合:');
disp(best_params);

save_filename = sprintf('Result_Sequential_%s_WithEbbinghaus.mat', dataname);
save(save_filename, 'results_log', 'best_params', 'best_result_metrics');
fprintf('结果已保存至 %s\n', save_filename);