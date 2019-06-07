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
X = X(1:3:end, :);
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
% U = 3.4 * U;
%%
U = zscore(U, 1, 2)/size(U,2)^.5;

ndims = 2;
niter = 4000;
eta0 = .01;
pW = 0.9;

my_metric = 'cauchy';

NN = size(U,1);
ys = U(:, 1:ndims);
ys = .5 * ys./std(ys,1,1);

nbase = 41;
ks = zeros(nbase, nbase, 2);
ks(:,:, 1) = repmat(0:nbase-1, nbase, 1);
ks(:,:, 2) = repmat(0:nbase-1, nbase, 1)';
ks = reshape(ks, [], 2);
zs = ks - mean(ks(:));
ngrid = nbase^2;

ys = gpuArray(single(ys));
dy = gpuArray.zeros(NN, ndims, 'single');
eta = linspace(eta0, eta0, niter);
lam = ones(NN,1);
oy = zeros(NN, ndims);

nbase = 21;
ks = zeros(nbase, nbase, 2);
ks(:,:, 1) = repmat(0:nbase-1, nbase, 1);
ks(:,:, 2) = repmat(0:nbase-1, nbase, 1)';
ks = reshape(ks, [], 2);


UtU = U * U';
UtU = UtU - diag(diag(UtU));
err0 = mean(mean(UtU.^2));
nc = size(ks,1) * ones(niter,1);

cold = Inf;
tic
for k = 1:niter
    ds = gpuArray.zeros(NN, ngrid, 'single');
    for j = 1:ndims
        ds = ds + (ys(:,j) - zs(:,j)').^2;
    end    
    S = exp(-ds);   
    
    y2 = sum(S.^2, 2).^.5;
    S = S ./y2;
       
    if rem(k,100)==1
        ypred = S * S';
        ypred = ypred -diag(diag(ypred));
        err = ypred - UtU;
        err = err - diag(diag(err));
        
        cnew = mean(err(:).^2)/err0;
        if cold < cnew
%             eta = eta/2;
        else
            cold = cnew;
        end
        fprintf('iter %d, eta %2.4f, time %2.2f, err %2.6f \n', k, eta(k),  toc, cnew)        
        plot(ys(:,1), ys(:,2), '.')
%         xlim([-50 50])
%         ylim([-50 50])
        drawnow
    end
    
    err = S * (S'*S) - U * (U' * S);    
    err = err .* S;
    for j = 1:ndims
       err2 = err .* (ys(:,j) - zs(:,j)');       
       dy(:,j) = - mean(err2, 2);                
    end
    dy = dy./sum(dy.^2,2).^.5;
    oy = pW * oy + (1-pW) * dy;    
    ys = ys - eta(k) * oy;
    
end
toc


drawnow
%%
ds = reshape(ys, [NN, 1, ndims]) - reshape(ys, [1, NN, ndims]);
% ds = mod(ds + pi, 2*pi) - pi;

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

semilogx(cb)
