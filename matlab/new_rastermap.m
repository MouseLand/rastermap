mname = 'TX28';
datexp = '2018_10_02';
blk = '1';

spks = readNPY(sprintf('Z:/data/PROC/%s/%s/%s/suite2p/combined/spks.npy', mname, datexp, blk));

iscell = readNPY(sprintf('Z:/data/PROC/%s/%s/%s/suite2p/combined/iscell.npy', mname, datexp, blk));
spks = spks(logical(iscell(:,1)), :);
[NN, Nframes] = size(spks);
%%
spks = readNPY('H:/s2p_paper/subj17_spks.npy');

%%
X = readNPY('D:\Github\rastercode\imgcov.npy');
X = zscore(X, [], 2);

%%
X = readNPY('D:/Github/data/allen-visp-alm/X.npy');
X = zscore(X(1:3:end, :), [], 2);
U = gpuArray(single(X));
%%
X = gpuArray(single(spks));
X = zscore(X, [], 2);
% X = X - mean(X, 1) ;

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
%%
[U, S, V] = svdecon(X1);
U = U .* diag(S)';

U = gpuArray(U(:, 1:200));
% U = 3.4 * zscore(U, [], 2);
U = zscore(U, 1, 2)./size(U,2).^.5;

%%
ndims = 2;
niter = 4200;
eta0 = .001;
pW = 0.9;

NN = size(U,1);
ys = .01 * U(:, 1:ndims);
% [~, ys] = sort(U(:, 1:ndims), 1);
% ys = 2 * pi * ys/size(ys,1) - pi;


ys = gpuArray(single(ys));
dy = gpuArray.zeros(NN, ndims, 'single');
err0 = mean(U(:).^2);

lam = ones(NN,1);
oy = zeros(NN, ndims);

nbase = 21;
ks = zeros(nbase, nbase, 2);
ks(:,:, 1) = repmat(0:nbase-1, nbase, 1);
ks(:,:, 2) = repmat(0:nbase-1, nbase, 1)';

ks = reshape(ks, [], 2);
fxx = max(ks,[], 2) + min(ks,[], 2)/1000;
[~, isort]  = sort(fxx);
ks          = ks(isort, :);

% ks = ks(2:end, :); % discard DC mode


alpha = 1;
nA = 1:size(ks,1);
nA = 2 * ceil(nA/2);
vscale = 1./nA.^(alpha/2);
vscale = vscale / sum(vscale.^2).^.5;

cold = Inf;

% nc = (3:nbase).^2 - 1;
% nc = repmat(nc, 50, 1);
% nc = cat(1, nc(:), nbase.^2 * ones(500,1));
nc = size(ks,1) * ones(niter,1);

niter = numel(nc);
eta = linspace(eta0, eta0, niter);
%%
% eta = eta/10;
tic
for k = 1:niter
    S = gpuArray.ones(NN,nc(k), 'single');
    dS = gpuArray.ones(NN,nc(k), ndims,'single');
    for j = 1:ndims
       S = S .* sin(pi + mod(ks(1:nc(k),j)'+1, 2) * pi/2 + ys(:,j) * floor((1+ks(1:nc(k),j)')/2));        
    end
    for j = 1:ndims
        dS(:,:,j) = S  ./ sin(pi + mod(ks(1:nc(k),j)'+1, 2) * pi/2 + ys(:,j) * floor((1+ks(1:nc(k),j)')/2));        
        dS(:,:,j) = dS(:,:,j) .* cos(pi + mod(ks(1:nc(k),j)'+1, 2) * pi/2 + ys(:,j) * floor((1+ks(1:nc(k),j)')/2)) .* floor((1+ks(1:nc(k),j)')/2);
    end    
    
    S2 = sum(S.^2,2).^.5;
    S = S./S2;
    
    xts = S' * U;
    if 1
        A = xts;
        nA = ones(nc(k), 1);
    else
        VV = mean(xts.^2, 2);
        nV = sum(VV);
        nA = nV^.5 * vscale' ./ VV.^.5;
        A =  xts .* nA ;
    end
    ypred = S * A;
    
    lam = mean(ypred .* U, 2) ./ (1e-4 + mean(ypred.^2, 2));
    err = lam .* ypred - U;
    if rem(k,100)==1
        cnew = mean(err(:).^2)/err0;
        if cold < cnew
            %             eta = eta/2;
        else
            cold = cnew;
        end
        fprintf('iter %d, eta %2.4f, time %2.2f, err %2.6f \n', k, eta(k),  toc, cnew)
        plot(ys(:,1), ys(:,2), '.')
        drawnow
    end
    
    err = lam.*err;
    err = (err) * A' + U * (err' * (S.*nA'));
    
    dy = squeeze(mean(dS .* err, 2));
    dy = dy./sum(dy.^2,2).^.5;
    
    oy = pW * oy + (1-pW) * dy;
    ys = ys - eta(k) * oy;
    
    ys = mod(ys+pi, 2*pi)-pi;
end
toc

plot(ys(:,1), ys(:,2), '.')
drawnow
%%
ds = reshape(ys, [NN, 1, ndims]) - reshape(ys, [1, NN, ndims]);
ds = mod(ds + pi, 2*pi)-pi;

ds = sum(ds.^2,3) + diag(Inf * ones(NN,1));
ds = gather(ds);

[~, isort] = sort(ds, 2);
Xz = zscore(Xz, [], 2)/size(Xz,2)^.5;
X0 = gpuArray.zeros(size(Xz), 'single');
cb = zeros(256,1);
for j = 1:256
    X0 = X0 + Xz(isort(:, j),:);
    X0z = zscore(X0, [], 2)/size(X0,2)^.5;
    cc = sum(Xz .* X0z, 2);
    cb(j) = gather(mean(cc));
end

cb(1)
semilogx(cb)
