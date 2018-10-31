function v = normc(v, eps)
% add epsilon to normalizer, and choose what dimension ndim to normalize

if nargin<2
    eps = 1e-20;
end

norms = sum(v.^2, 1).^.5;
norms = norms + eps;

v = bsxfun(@rdivide, v, norms);
