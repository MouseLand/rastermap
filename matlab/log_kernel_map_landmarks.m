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
spks = readNPY('H:/s2p_paper/recGAD.npy');
spks = squeeze(mean(spks, 1))';

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
U = gpuArray(single(X));
U = zscore(U, 1, 2);
%%
X = readNPY('D:/Github/data/allen-visp-alm/logCPM.npy');
% X = gpuArray(single(X(1:3:end, :)));
X = gpuArray(single(X));
X = X - mean(X,1);
[U, S, V] = svdecon(X);
U = X * V;
U = U(:, 1:256);
U = zscore(U, 1, 2)/size(U,2)^.5;

%%
X = gpuArray(single(spks));
X = zscore(X, 1, 2);
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

[U, S, V] = svdecon(X1(1:end, :));

U = X1 * V;
U = gpuArray(U(:, 1:512));
U = zscore(U, 1, 2)/size(U,2)^.5;
%%

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


%%

ndims = 2;
niter = 4000;
eta0 = .1;
pW = 0.9;

my_metric = 'neglog';

NN = size(U,1);
ys = U(:, 1:ndims);
nA = .5 ./std(ys,1,1);
zs = nA .* u0(:, 1:ndims);
ys = zs(imax, :);
 
ys = gpuArray(single(ys));
dy = gpuArray.zeros(NN, ndims, 'single');
dz = gpuArray.zeros(nraster, ndims, 'single');

eta = linspace(eta0, eta0, niter);
lam = gpuArray.ones(NN,1, 'single');

oy = zeros(NN, ndims);
oz = zeros(nraster, ndims);

% define kernel here
K = U * u0';
% K = K./max(K, [], 1);

% LAM = gpuArray.ones(size(K), 'single');
LAM = .1 + abs(K);
err0 = mean(mean(LAM .* K.^2));

tic
for k = 1:niter
    ds = gpuArray.zeros(NN, nraster, 'single');
    for j = 1:ndims
        ds = ds + (ys(:,j) -zs(:,j)').^2;
    end
    switch my_metric
        case 'cauchy'
            W  = 1./(1 + ds);
        case 'gaussian'    
            W = exp(-ds); 
        case 'exp'
            W = exp(-ds.^.5); 
        case 'neglog'
            W = (log(20^2) - log(1 + ds))/log(20^2); 
    end
    
    lam = mean(LAM .* W .* K, 2) ./ (1e-3 + mean(LAM .* W.^2, 2));
    err = lam .* W - K;
    
    if rem(k,100)==1 || k==niter
        cnew = mean(mean(LAM .* err.^2)) / err0;
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
            [~, isort] = sort(zs);
            imagesc(CC(isort, isort))
        end
        drawnow        
    end
    
     err = (err .* lam) .* LAM;
    switch my_metric
        case 'cauchy'
            err = err .* W.^2;
        case 'gaussian'    
            err = err .*  W;        
        case 'exp'
           err = err .*  W;
        case 'neglog'
            err = err ./(1+ds);
    end
    
    for i = 1:ndims        
        err2 = err .* (ys(:,i)  - zs(:,i)');
        switch my_metric
            case 'exp'
                err2 = err2 .* (1e-2 + ds).^-.5;
        end
        
        D = mean(err2, 2);
        E = mean(err2, 1);
        dy(:,i) = -D;
        dz(:,i) = E'; 
    end
    dy = dy./sum(dy.^2,2).^.5;
    dz = dz./sum(dz.^2,2).^.5;
    
    oy = pW * oy + (1-pW) * dy;
    oz = pW * oz + (1-pW) * dz;
    
    ys = ys - eta(k) * oy;
    zs = zs - eta(k) * oz;    
end
toc


drawnow


%%
ikr = randperm(NN, min(NN, 1e3));
ds = reshape(ys(ikr, :), [numel(ikr), 1, ndims]) - reshape(ys, [1, NN, ndims]);
ds = mod(ds + pi, 2*pi)-pi;

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
