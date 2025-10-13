function [tree, splits, is_leaf, clusters, timings, Ws, priorities] = hiernmf_unified(X, k, params, method)
%%HIERNMF_UNIFIED - Hierarchical clustering based on rank-2 NMF with different splitting strategies
% [tree, splits, is_leaf, clusters, timings, Ws, priorities] = hiernmf_unified(X, k, params, method)
%
% Input parameters --
% X: m*n data matrix (m features x n data points)
% k: The max number of leaf nodes to be generated
% params (optional): Parameter structure (same as original functions)
% method (optional): Splitting strategy. Options:
%   'neat' (default) - Original neat splitting using max(H) assignment
%   'balanced' - Uses max(H) assignment with enhanced priority computation
%   'even' - Forces even split using median of H difference
%
% Output parameters --
% Same as original functions

    % Set default method if not provided
    if nargin < 4
        method = 'neat';
    end
    
    % Validate method parameter
    valid_methods = {'neat', 'balanced', 'even'};
    if ~ismember(method, valid_methods)
        error('Invalid method. Must be one of: neat, balanced, even');
    end

    % Parameter initialization (same for all methods)
    if ~exist('params', 'var')
        trial_allowance = 3;
        unbalanced = 0.1;
        vec_norm = 2.0;
        normW = true;
        anls_alg = @anls_entry_rank2_precompute;
        tol = 1e-4;
        maxiter = 10000;
    else
        if isfield(params, 'trial_allowance')
            trial_allowance = params.trial_allowance;
        else
            trial_allowance = 3;
        end
        if isfield(params, 'unbalanced')
            unbalanced = params.unbalanced;
        else
            unbalanced = 0.1;
        end
        if isfield(params, 'vec_norm')
            vec_norm = params.vec_norm;
        else
            vec_norm = 2.0;
        end
        if isfield(params, 'normW')
            normW = params.normW;
        else
            normW = true;
        end
        if isfield(params, 'anls_alg')
            anls_alg = params.anls_alg;
        else
            anls_alg = @anls_entry_rank2_precompute;
        end
        if isfield(params, 'tol')
            tol = params.tol;
        else
            tol = 1e-4;
        end
        if isfield(params, 'maxiter')
            maxiter = params.maxiter;
        else
            maxiter = 10000;
        end
    end

    params = [];
    params.vec_norm = vec_norm;
    params.normW = normW;
    params.anls_alg = anls_alg;
    params.tol = tol;
    params.maxiter = maxiter;

    t0 = tic;
    [m, n] = size(X);

    % Initialize data structures (same for all methods)
    timings = zeros(1, k-1);
    clusters = cell(1, 2*(k-1));
    Ws = cell(1, 2*(k-1));
    W_buffer = cell(1, 2*(k-1));
    H_buffer = cell(1, 2*(k-1));
    priorities = zeros(1, 2*(k-1));
    is_leaf = -1 * ones(1, 2*(k-1));
    tree = zeros(2, 2*(k-1));
    splits = -1 * ones(1, k-1);

    % Initial NMF computation (same for all methods)
    term_subset = find(sum(X, 2) ~= 0);
    W = rand(length(term_subset), 2);
    H = rand(2, n);
    if length(term_subset) == m
        [W, H] = nmfsh_comb_rank2(X, W, H, params);
    else
        [W_tmp, H] = nmfsh_comb_rank2(X(term_subset, :), W, H, params);
        W = zeros(m, 2);
        W(term_subset, :) = W_tmp;
        clear W_tmp;
    end

    result_used = 0;
    for i = 1 : k-1
        timings(i) = toc(t0);

        if i == 1
            split_node = 0;
            new_nodes = [1 2];
            min_priority = 1e308;
            split_subset = 1:n;
        else
            leaves = find(is_leaf == 1);
            temp_priority = priorities(leaves);
            min_priority = min(temp_priority(temp_priority > 0));
            [max_priority, split_node] = max(temp_priority);
            if max_priority < 0
                fprintf('Cannot generate all %d leaf clusters\n', k);
                return;
            end
            split_node = leaves(split_node);
            is_leaf(split_node) = 0;
            W = W_buffer{split_node};
            H = H_buffer{split_node};
            split_subset = clusters{split_node};
            new_nodes = [result_used+1 result_used+2];
            tree(1, split_node) = new_nodes(1);
            tree(2, split_node) = new_nodes(2);
        end

        result_used = result_used + 2;
        
        % METHOD-SPECIFIC CLUSTERING ASSIGNMENT
        if strcmp(method, 'even')
            % Even splitting: use median of H difference
            H_sub = H(1, :) - H(2, :);
            med = median(H_sub);
            clusters{new_nodes(1)} = split_subset(find(H_sub >= med));
            clusters{new_nodes(2)} = split_subset(find(H_sub < med));
        else
            % Neat and balanced: use max(H) assignment
            [max_val, cluster_subset] = max(H);
            clusters{new_nodes(1)} = split_subset(find(cluster_subset == 1));
            clusters{new_nodes(2)} = split_subset(find(cluster_subset == 2));
        end
        
        Ws{new_nodes(1)} = W(:, 1);
        Ws{new_nodes(2)} = W(:, 2);
        splits(i) = split_node;
        is_leaf(new_nodes) = 1;

        % Process child nodes with method-specific actual_split function
        subset = clusters{new_nodes(1)};
        [subset, W_buffer_one, H_buffer_one, priority_one] = trial_split(trial_allowance, unbalanced, min_priority, X, subset, W(:, 1), params, method);
        clusters{new_nodes(1)} = subset;
        W_buffer{new_nodes(1)} = W_buffer_one;
        H_buffer{new_nodes(1)} = H_buffer_one;
        priorities(new_nodes(1)) = priority_one;

        subset = clusters{new_nodes(2)};
        [subset, W_buffer_one, H_buffer_one, priority_one] = trial_split(trial_allowance, unbalanced, min_priority, X, subset, W(:, 2), params, method);
        clusters{new_nodes(2)} = subset;
        W_buffer{new_nodes(2)} = W_buffer_one;
        H_buffer{new_nodes(2)} = H_buffer_one;
        priorities(new_nodes(2)) = priority_one;
    end
end

%--------------------------------------

function [subset, W_buffer_one, H_buffer_one, priority_one] = trial_split(trial_allowance, unbalanced, min_priority, X, subset, W_parent, params, method)
    [m, n] = size(X);

    trial = 0;
    subset_backup = subset;
    while trial < trial_allowance
        [cluster_subset, W_buffer_one, H_buffer_one, priority_one] = actual_split(X, subset, W_parent, params, method);
        if priority_one < 0
            break;
        end
        unique_cluster_subset = unique(cluster_subset);
        if length(unique_cluster_subset) ~= 2
            error('Invalid number of unique sub-clusters!');
        end
        length_cluster1 = length(find(cluster_subset == unique_cluster_subset(1)));
        length_cluster2 = length(find(cluster_subset == unique_cluster_subset(2)));
        if min(length_cluster1, length_cluster2) < unbalanced * length(cluster_subset)
            [min_val, idx_small] = min([length_cluster1, length_cluster2]);
            subset_small = find(cluster_subset == unique_cluster_subset(idx_small));
            subset_small = subset(subset_small);
            [cluster_subset_small, W_buffer_one_small, H_buffer_one_small, priority_one_small] = actual_split(X, subset_small, W_buffer_one(:, idx_small), params, method);
            if priority_one_small < min_priority
                trial = trial + 1;
                if trial < trial_allowance
                    disp(['Drop ', num2str(length(subset_small)), ' documents ...']);
                    subset = setdiff(subset, subset_small);
                end
            else
                break;
            end
        else
            break;
        end
    end

    if trial == trial_allowance
        disp(['Recycle ', num2str(length(subset_backup) - length(subset)), ' documents ...']);
        subset = subset_backup;
        W_buffer_one = zeros(m, 2);
        H_buffer_one = zeros(2, length(subset));
        priority_one = -2;
    end
end

%--------------------------------------

function [cluster_subset, W_buffer_one, H_buffer_one, priority_one] = actual_split(X, subset, W_parent, params, method)
    [m, n] = size(X);
    if length(subset) <= 3
        cluster_subset = ones(1, length(subset));
        W_buffer_one = zeros(m, 2);
        H_buffer_one = zeros(2, length(subset));
        priority_one = -1;
    else
        term_subset = find(sum(X(:, subset), 2) ~= 0);
        X_subset = X(term_subset, subset);
        W = rand(length(term_subset), 2);
        H = rand(2, length(subset));
        [W, H] = nmfsh_comb_rank2(X_subset, W, H, params);
        [max_val, cluster_subset] = max(H);
        W_buffer_one = zeros(m, 2);
        W_buffer_one(term_subset, :) = W;
        H_buffer_one = H;
        
        if length(unique(cluster_subset)) > 1
            % METHOD-SPECIFIC PRIORITY COMPUTATION
            if strcmp(method, 'balanced')
                num_elements = size(X_subset, 2);
                priority_one = compute_priority(W_parent, W_buffer_one) * num_elements;
            elseif strcmp(method, 'even')
                priority_one = 1; % Simplified priority for even method
            else % neat method
                priority_one = compute_priority(W_parent, W_buffer_one);
            end
        else
            priority_one = -1;
        end
    end
end

%--------------------------------------

function priority = compute_priority(W_parent, W_child)
    n = length(W_parent);
    [sorted_parent, idx_parent] = sort(W_parent, 'descend');
    [sorted_child1, idx_child1] = sort(W_child(:, 1), 'descend');
    [sorted_child2, idx_child2] = sort(W_child(:, 2), 'descend');

    n_part = length(find(W_parent ~= 0));
    if n_part <= 1
        priority = -3;
    else
        weight = log(n:-1:1)';
        first_zero = find(sorted_parent == 0, 1); 
        if length(first_zero) > 0 
            weight(first_zero:end) = 1;
        end
        weight_part = zeros(n, 1);
        weight_part(1:n_part) = log(n_part:-1:1)';
        [sorted, idx1] = sort(idx_child1);
        [sorted, idx2] = sort(idx_child2);
        max_pos = max(idx1, idx2);
        discount = log(n-max_pos(idx_parent)+1);
        discount(discount == 0) = log(2);
        weight = weight ./ discount;
        weight_part = weight_part ./ discount;
        % NOTE: in hiernmf_balanced, we multiplied by n here as well
        % however, n is the same for all nodes so this does not make 
        % a difference in the split result...
        priority = NDCG_part(idx_parent, idx_child1, weight, weight_part) * NDCG_part(idx_parent, idx_child2, weight, weight_part);
    end
end

%--------------------------------------

function score = NDCG_part(ground, test, weight, weight_part)
    [sorted, seq_idx] = sort(ground);
    weight_part = weight_part(seq_idx);

    n = length(test);
    uncum_score = weight_part(test);
    uncum_score(2:n) = uncum_score(2:n) ./ log2(2:n)';
    cum_score = cumsum(uncum_score);

    ideal_score = sort(weight, 'descend');
    ideal_score(2:n) = ideal_score(2:n) ./ log2(2:n)';
    cum_ideal_score = cumsum(ideal_score);

    score = cum_score ./ cum_ideal_score;
    score = score(end);
end