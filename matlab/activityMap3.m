function [isort1, isort2, Sm] = activityMap3(S, ops)
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
ops.nC      = getOr(ops, 'nC', [30 100]);
ops.iPC     = getOr(ops, 'iPC', 1:200);
ops.isort1   = getOr(ops, 'isort1', []);
ops.isort2   = getOr(ops, 'isort2', []);
ops.upsamp  = getOr(ops, 'upsamp', 100);
ops.sigUp   = getOr(ops, 'sigUp', 1);
ops.axisZero = getOr(ops, 'axisZero', 2);

nC = ops.nC;
iPC = ops.iPC;

S = gpuArray(single(S));
S = S - mean(S, ops.axisZero);

% initialize sortings by top PC
[U Sv V] = svdecon(S);
S = U(:, iPC) * Sv(iPC, iPC) * V(:, iPC)';

if isempty(ops.isort1)
    [~, isort1] = sort(U(:,1), 'descend');
    [~, isort2] = sort(V(:,1), 'descend');
end

[NN, NT] = size(S);



iclust1        = zeros(NN, 1);
iclust1(isort1) = ceil(linspace(1e-5, nC(1), NN));

S = S(isort1, isort2);
ss = [NN NT]./nC;


Sm = my_conv2(S, ss/5);           
sig1 = [linspace(nC(1)/10, 1,100) ones(1,50)];
Km1 = getUpsamplingKernel(nC(1), ops.sigUp, ops.upsamp);
for t = 1:numel(sig1)
    iresort1 = recluster(Sm, nC(1), sig1(t), iclust1, Km1, ops);
    iclust1(iresort1) = ceil(linspace(1e-5, nC(1), NN));    
    isort1 = isort1(iresort1);
end

iclust2        = zeros(NT, 1);
iclust2(isort2) = ceil(linspace(1e-5, nC(2), NT));
sig2 = [linspace(nC(2)/10, 1,100) ones(1,50)];
Km2 = getUpsamplingKernel(nC(2), ops.sigUp, ops.upsamp);
for t = 1:numel(sig1)
    iresort2 = recluster(Sm', nC(2), sig2(t), iclust2, Km2, ops);
    iclust2(iresort2) = ceil(linspace(1e-5, nC(2), NT));    
    isort2 = isort2(iresort2);    
    
    S = S(iresort1, iresort2);    
    Sm = my_conv2(S, ss/5, [1 2]);   
    toc
end

end


function isort = recluster(S, nC, sig, iclust, Km, ops)

V = gpuArray.zeros(size(S,2), nC(1), 'single');
for j = 1:nC
    ix = iclust== j;
    V(:, j) = sum(S(ix, :),1);
end
V = my_conv2(V, sig, 2);
V = normc(V);
cv = S * V;

[~, iclust] = max(cv * Km', [], 2);
iclust = iclust / ops.upsamp;
[~, isort] = sort(iclust);

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