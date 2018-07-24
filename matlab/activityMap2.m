function [iclustup, isort] = activityMap2(S, ops)

% nC, iPCs, isort, useGPU
ops.nC = getOr(ops, 'nC', [30 100]);
ops.iPC = getOr(ops, 'iPC', 1:30);
ops.isort1 = getOr(ops, 'isort1', []);
ops.isort2 = getOr(ops, 'isort2', []);
ops.useGPU = getOr(ops, 'useGPU', 0);
ops.upsamp = getOr(ops, 'upsamp', 100);
ops.sigUp = getOr(ops, 'sigUp', 1);
ops.axisZeroMean = getOr(ops, 'axisZeroMean', 2);

if numel(ops.nC)==1
   error('nC is a scalar. You must provide the number of clusters for both axes') 
end

% S is neurons by time
S = S - mean(S,ops.axisZeroMean);

if isempty(ops.isort1) || isempty(ops.isort2)
   % initialize sortings by top PC 
    [U Sv V] = svdecon(S);   
    
    if isempty(ops.isort1)
        isort1 = sort(U(:,1), 'descend');
    end
    if isempty(ops.isort2)
        isort2 = sort(V(:,1), 'descend');
    end
end

[NN, NT] = size(S);

% 
nn = floor(NN/nC);
iclust1 = zeros(NN, 1);
iclust1(isort1) = ceil([1:NN]/nn);
iclust1(iclust1>nC) = nC;

% annealing schedule for embedding smoothness
sig = [linspace(nC/10,1,100) 1*ones(1,50)];

for t = 1:numel(sig)
    if useGPU
        V = gpuArray.zeros(NT, nC, 'single');
    else
        V = zeros(NT, nC, 'single');
    end
    for j = 1:nC
        ix = iclust== j;
        V(:, j) = sum(S(ix, :),1);
    end
    
    V = my_conv2(V, sig(t), 2);
    V = normc(V);
    
    cv = S * V;
    [cmax, iclust] = max(cv, [], 2);
    
    %disp(mean(100 * cmax.^2));
    
end

% create kernels for upsampling
Km = getUpsamplingKernel(nC, ops.sigUp);
[cmaxup, iclustup] = max(cv * Km', [], 2);
iclustup = gather_try(iclustup);
[~, isort] = sort(iclustup);
iclustup = iclustup/ops.upsamp;


end

function Km = getUpsamplingKernel(nC, sig)

xs = 1:nC;
xn = linspace(1, nC, nC*ops.upsamp);
d0 = (xs' - xs).^2;
d1 = (xn' - xs).^2;
K0 = exp(-d0/sig);
K1 = exp(-d1/sig);
Km = K1 / (K0 + 0.001 * eye(nC));

end