mname = 'TX28';
datexp = '2018_10_02';
blk = '1';

spks = readNPY(sprintf('Z:/data/PROC/%s/%s/%s/suite2p/combined/spks.npy', mname, datexp, blk));

iscell = readNPY(sprintf('Z:/data/PROC/%s/%s/%s/suite2p/combined/iscell.npy', mname, datexp, blk));
spks = spks(logical(iscell(:,1)), :);
[NN, Nframes] = size(spks);
%%
X = readNPY('D:\Github\rastercode\imgcov.npy');
X = zscore(X, [], 2);

[U, S, V] = svdecon(X);
U = U .* diag(S)';
U = gpuArray(U(:, 1:256));
% U = zscore(U, 1, 2);
U = zscore(U, 1, 2)/size(U,2)^.5;

%%
X = readNPY('D:/Github/data/allen-visp-alm/X.npy');
X = zscore(X(1:3:end, :), 1, 2)/size(X,2)^.5;
U = gpuArray(single(X));
%%
X = gpuArray(single(spks));
X = zscore(X, [], 2);
%X = X - mean(X, 1) ;

Lblock = 128;
inds = 1:size(X,2);
iblock = ceil(inds/Lblock);
nblocks = max(iblock);
iperm = randperm(nblocks);

Ntrain = ceil(3/4 * nblocks);
itrain = ismember(iblock, iperm(1:Ntrain));
itest  = ismember(iblock, iperm(1+Ntrain:nblocks));

X1 = X(:, itrain);
Xz = X(:, itest);

[U, S, V] = svdecon(X1);
U = U .* diag(S)';

U = gpuArray(U(:, 1:256));
U = zscore(U, 1, 2)/size(U,2)^.5;
% U = 3.4 * U;
%%
ndims = 2;
niter = 4000;
eta0 = .001;
pW = 0.9;

my_metric = 'cauchy';

NN = size(U,1);
ys = U(:, 1:ndims);
ys = .5 * ys./std(ys,1,1);

ys = gpuArray(single(ys));
dy = gpuArray.zeros(NN, ndims, 'single');
eta = linspace(eta0, eta0, niter);
lam = ones(NN,1);
oy = zeros(NN, ndims);

UtU = U * U';
% UtU = UtU  - mean(UtU, 1);
% UtU = UtU  - mean(UtU, 2);
% UtU = zscore(UtU, 1, 1);
% UtU = zscore(UtU, 1, 2);
err0 = mean(UtU(:).^2);

cold = Inf;
tic
for k = 1:niter
    ds = gpuArray.zeros(NN, NN, 'single');
    for j = 1:ndims
        ds = ds + (ys(:,j) -ys(:,j)').^2;
    end
    if strcmp(my_metric, 'cauchy')
        W = 1./(1 + ds);
    else
       W = exp(-ds); 
    end    
    W = W - diag(diag(W));
    
    U0 = U + 0 * gpuArray.randn(size(U), 'single');
    
    ypred = W;

%      lam = mean(ypred .* UtU, 2) ./ (1e-3 + mean(ypred.^2, 2));        
%      err = lam .* ypred - UtU;
    err = ypred - UtU;
    err = err - diag(diag(err));
    
    if rem(k,25)==1
        cnew = mean(err(:).^2)/err0;
        if cold < cnew
            eta = eta/2;
        else
            cold = cnew;
        end
        fprintf('iter %d, eta %2.4f, time %2.2f, err %2.6f \n', k, eta(k),  toc, cnew)        
        plot(ys(:,1), ys(:,2), '.')
%         xlim([-50 50])
%         ylim([-50 50])
        drawnow
    end
    
%     err = (lam.*err);
    if strcmp(my_metric, 'cauchy')
        err = err .* W.^2;
    else        
        err = err .*  W;
    end
    
    for i = 1:ndims
       D = mean(err .* (ys(:,i)  - ys(:,i)'), 2);       
       E = mean(err .* (ys(:,i)' - ys(:,i) ), 1);       
       dy(:,i) = -D - E'; % + D1 + D2;
    end
    dy = dy./sum(dy.^2,2).^.5;
    
    oy = pW * oy + (1-pW) * dy;
    ys = ys - eta(k) * oy;
    
end
toc


drawnow
%%
ds = reshape(ys, [NN, 1, ndims]) - reshape(ys, [1, NN, ndims]);
W = exp(-sum(ds.^2, 3));
W = W - diag(diag(W));
W = gather(W);
[~, isort] = sort(W, 2, 'descend');
Xz = zscore(Xz, [], 2)/size(Xz,2)^.5;
X0 = gpuArray.zeros(size(Xz), 'single');
cb = zeros(128,1);
for j = 1:128
    X0 = X0 + Xz(isort(:, j),:);
    X0z = zscore(X0, [], 2)/size(X0,2)^.5;
    cc = sum(Xz .* X0z, 2);
    cb(j) = gather(mean(cc));
end

cb(1)
semilogx(cb)
