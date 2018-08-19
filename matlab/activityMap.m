function [iclustup, isort, Vout] = activityMap(S, ops)
% sorts the matrix S (neurons by time) along the first axis
% ops.nC = 30, number of clusters to use 
% ops.iPC = 1:100, number of PCs to use 
% ops.isort = [], initial sorting, otherwise will be the top PC sort
% ops.useGPU = 0, whether to use the GPU
% ops.upsamp = 100, upsampling factor for the embedding position
% ops.sigUp = 1, % standard deviation for upsampling

if nargin<2
   ops = []; 
end
%%
ops.nC = getOr(ops, 'nC', 30);
ops.iPC = getOr(ops, 'iPC', 1:200);
ops.isort = getOr(ops, 'isort', []);
ops.useGPU = getOr(ops, 'useGPU', 0);
ops.upsamp = getOr(ops, 'upsamp', 100);
ops.sigUp = getOr(ops, 'sigUp', 1);

nC = ops.nC;
if ops.useGPU
    S = gpuArray(single(S));
end
S = S - mean(S,2);

% initialize sortings by top PC
[U Sv V1] = svdecon(S);
%%
if isempty(ops.isort)
    [~, isort] = sort(U(:,1), 'descend');
end
iPC = ops.iPC;
S = U(:, iPC) * Sv(iPC, iPC);


[NN, nPC] = size(S);

nn = floor(NN/nC);
iclust = zeros(NN, 1);
iclust(isort) = ceil([1:NN]/nn);
iclust(iclust>nC) = nC;

%% annealing schedule for embedding smoothness
sig = [linspace(nC/10,1,100) 1*ones(1,50)];
for t = 1:numel(sig)
    if ops.useGPU
        V = gpuArray.zeros(nPC, nC, 'single');
    else
        V = zeros(nPC, nC, 'single');
    end
    for j = 1:nC
        ix = iclust== j;
        V(:, j) = sum(S(ix, :),1);
    end
    
    V = my_conv2(V, sig(t), 2);
    V = normc(V);
    
    cv = S * V;
    [cmax, iclust] = max(cv, [], 2);
end
%%
% create kernels for upsampling
Km = getUpsamplingKernel(nC, ops.sigUp, ops.upsamp);
[cmaxup, iclustup] = max(cv * Km', [], 2);
iclustup = gather_try(iclustup);
[~, isort] = sort(iclustup);
iclustup = iclustup/ops.upsamp;

Vout = V1(:,iPC) * V;

end

function Km = getUpsamplingKernel(nC, sig, upsamp)

xs = 1:nC;
xn = linspace(1, nC, nC * upsamp);
d0 = (xs' - xs).^2;
d1 = (xn' - xs).^2;
K0 = exp(-d0/sig);
K1 = exp(-d1/sig);
Km = K1 / (K0 + 0.001 * eye(nC));

end