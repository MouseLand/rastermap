function [isort1, isort2, Sm] = mapTmap(S, ops)
% run the activity map along the second dimension, then use this
% information to run it across the first dimension
% ops.nCall contains two numbers: number of clusters along dimension 1, and
% along dimension 2

if nargin<2
   ops = []; 
end

ops.nCall      = getOr(ops, 'nCall', [30 100]);


tic
ops.nC = ops.nCall(2);
[~, isort2, ~] = activityMap(S', ops);
toc

Sm = my_conv2(S(:, isort2), 3, 2);
ops.nC = ops.nCall(1);
[~, isort1, ~] = activityMap(Sm, ops);

Sm = my_conv2(S(isort1, :), [10 1], [1 2]);
Sm = zscore(Sm, 1, 2);

toc

end

