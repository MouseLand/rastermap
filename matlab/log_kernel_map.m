mname = 'TX28';
datexp = '2018_10_02';
blk = '1';

spks = readNPY(sprintf('Z:/data/PROC/%s/%s/%s/suite2p/combined/spks.npy', mname, datexp, blk));

iscell = readNPY(sprintf('Z:/data/PROC/%s/%s/%s/suite2p/combined/iscell.npy', mname, datexp, blk));
spks = spks(logical(iscell(:,1)), :);
[NN, Nframes] = size(spks);
%%
spks = readNPY('H:/s2p_paper/subj17_spks.npy');
spks = spks(1:5:end, :);
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
X = readNPY('D:/Github/data/allen-visp-alm/logCPM.npy');
X = gpuArray(single(X(1:2:end, :)));
X = X - mean(X,1);
[U, S, V] = svdecon(X);
U = X * V;
U = U(:, 1:256);
U = zscore(U, 1, 2)/size(U,2)^.5;
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

[U, S, V] = svdecon(X1);
U = U .* diag(S)';

U = gpuArray(U(:, 1:1024));
U = zscore(U, 1, 2)/size(U,2)^.5;

%%
u0 = U;
u0 = u0./sum(u0.^2,2).^.5;

ndims = 2;
niter = 10000;
eta0 = .01;
pW = 0.9;

my_metric = 'neglog';

NN = size(U,1);
ys = U(:, 1:ndims);
ys = .5 * ys./std(ys,1,1);

ys = gpuArray(single(ys));
dy = gpuArray.zeros(NN, ndims, 'single');
eta = linspace(eta0, eta0, niter);
lam = - gpuArray.ones(NN,1, 'single');
olam = gpuArray.zeros(NN,1, 'single');
oy = zeros(NN, ndims);


K  = u0 * u0';
% K = readNPY('D:\Github\rastercode\imgcov.npy');
% LAM = gpuArray.ones(size(K), 'single');
LAM = abs(K).^2;
err0 = mean(mean(LAM .* K.^2));

Ld = 10;

cold = Inf;
tic
for k = 1:niter    
    ds = gpuArray.zeros(NN, NN, 'single');
    for j = 1:ndims
        ds = ds + (ys(:,j) -ys(:,j)').^2;
    end
    switch my_metric
        case 'cauchy'
            W  = 1./(1 + ds);
        case 'gaussian'    
            W = exp(-ds); 
        case 'exp'
            W = exp(-ds.^.5); 
        case 'neglog'
            W = 1  - log(.25 + ds)/log(Ld^2); 
    end
        
    Wlam = W .* exp(lam)';
    err = (exp(lam) .* Wlam - K);
    err = err - diag(diag(err));
    
    
    if rem(k,100)==1
         cnew = mean(mean(LAM .* err.^2)) / err0;
        if cold < cnew
%             eta = eta/2;
        else
            cold = cnew;
        end
        fprintf('iter %d, eta %2.4f, time %2.2f, err %2.6f \n', k, eta(k),  toc, cnew)        
        if ndims>1
            hold off
            plot(ys(:,1), ys(:,2), '.')
            drawnow
        end
    end
    err = err .* LAM;
    dlam = mean(err.*Wlam,2).*exp(lam);    
    dlam = dlam./mean(dlam.^2).^.5;
    olam = pW * olam + (1-pW) * dlam;
    
    err = exp(lam).*err.* exp(lam)';
    
     switch my_metric
        case 'cauchy'
            err = err .* W.^2;
        case 'gaussian'    
            err = err .*  W;        
        case 'exp'
           err = err .*  W;
        case 'neglog'
            err = err ./(.25 + +ds);
    end

    for i = 1:ndims
         switch my_metric
            case 'exp'
                err2 = err2 .* (1e-2 + ds).^-.5;
        end
        err2 = err .* (ys(:,i)  - ys(:,i)');
        D = mean(err2, 2);
        E = mean(err2, 1);
        dy(:,i) = -D + E'; % + D1 + D2;
    end
    dy = dy./sum(dy.^2,2).^.5;
    
    oy = pW * oy + (1-pW) * dy;
    ys = ys - eta(k) * oy;
    lam = lam - eta(k) * olam;
end
toc


drawnow
%%



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
