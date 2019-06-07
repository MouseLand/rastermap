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
U = zscore(U, 1, 2);
%%
X = readNPY('D:/Github/data/allen-visp-alm/X.npy');
% X = zscore(X(1:3:end, :), [], 2);
U = gpuArray(single(X));

nraster = 256;
irand = randperm(size(U,1), nraster);
u0 = U(irand, :);
u0 = u0./sum(u0.^2,2).^.5;
tic
N = zeros(nraster, 1);
for k = 1:10
    cv = max(0, U * u0');    
    [lam, imax] = max(cv, [], 2);
    
    for j = 1:nraster
        N(j)=gather(sum(imax==j));
        u0(j, :) = lam(imax==j)' * U(imax==j, :);
    end
    u0 = u0./sum(u0.^2,2).^.5;
end
toc

U = u0;
UtU = max(0, U * u0');
dmax = max(UtU, [], 2).^.5;
UtU = UtU ./(dmax * dmax');
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
U = zscore(U, 1, 2);
% U = 3.4 * U;
%%
CC = corrcoef(U');
%%
ndims = 1;
niter = 4000;
eta0 = .1;
pW = 0.9;

my_metric = 'gaussian';

NN = size(U,1);
ys = U(:, 1:ndims);
ys = .5 * ys./std(ys,1,1);
% ys = ys0;

ys = gpuArray(single(ys));
dy = gpuArray.zeros(NN, ndims, 'single');
err0 = mean(U(:).^2);
eta = linspace(eta0, eta0, niter);
lam = ones(NN,1);
oy = zeros(NN, ndims);

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
    
    ypred = W * U;
    lam = mean(ypred .* U, 2) ./ (1e-6 + mean(ypred.^2, 2));        
    err = lam .* ypred - U;
    if rem(k,100)==1
        cnew = mean(err(:).^2)/err0;
        if cold < cnew
%             eta = eta/2;
        else
            cold = cnew;
        end
        fprintf('iter %d, eta %2.4f, time %2.2f, err %2.6f \n', k, eta(k),  toc, cnew)        
        if ndims>1
            plot(ys(:,1), ys(:,2), '.', 'markersize', 4)
            hold on
            plot(zs(:,1), zs(:,2), 'or', 'Linewidth', 2)
            hold off
        else
%             [~, isort] = sort(ys);
%             imagesc(CC(isort, isort))
        end
        drawnow
    end
    
    err = (lam.*err) * U';
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
ikr = randperm(NN, min(NN, 1e3));
ds = reshape(ys(ikr, :), [numel(ikr), 1, ndims]) - reshape(ys, [1, NN, ndims]);
% ds = mod(ds + pi, 2*pi)-pi;

ds = sum(ds.^2,3);
ds(ds<1e-5) = Inf;
ds = gather(ds);

[~, isort] = sort(ds, 2);
Xz = zscore(Xz, [], 2)/size(Xz,2)^.5;
X0 = gpuArray.zeros(numel(ikr), size(Xz,2), 'single');
cb = zeros(256,1);
for j = 1:256
    X0 = X0 + Xz(isort(:, j),:);
    X0z = zscore(X0, [], 2)/size(X0,2)^.5;
    cc = sum(Xz(ikr, :) .* X0z, 2);
    cb(j) = gather(mean(cc));
end

cb(1)
semilogx(cb)
