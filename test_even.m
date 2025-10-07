load reuters;
% X = matrix_ncw(X);
params = struct();
params.trial_allowance = 1;
[tree, splits, is_leaf, clusters, timings, Ws, priorities] = hiernmf_even(X, 20, params);
