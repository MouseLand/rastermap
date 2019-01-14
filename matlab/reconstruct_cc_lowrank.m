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
X = gpuArray(single(spks));
X = zscore(X, 1, 2);
X = X - mean(X, 1) ;

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

% U = u0;
UtU = max(0, U * u0');
% dmax1 = max(UtU, [], 2).^.5;
% dmax2 = max(UtU, [], 1).^.5;
% UtU = UtU ./(dmax1 * dmax2);

dmax = max(UtU, [], 2);
UtU = UtU ./dmax;

%%

ndims = 2;
niter = 4000;
eta0 = .01;
pW = 0.9;

my_metric = 'cauchy';

NN = size(U,1);
ys = U(:, 1:ndims);
ys = .05 * ys./std(ys,1,1);
zs = ys(irand, :);
% zs = ys;

ys = gpuArray(single(ys));
dy = gpuArray.zeros(NN, ndims, 'single');
dz = gpuArray.zeros(nraster, ndims, 'single');

eta = linspace(eta0, eta0, niter);
lam = ones(NN,1);

oy = zeros(NN, ndims);
oz = zeros(nraster, ndims);

% t = 0;
% sig = 1./linspace(1, 1, niter/100);
% UtUx = exp((UtU - 1)/sig(1));
UtUx = UtU;
err0 = mean(UtUx(:).^2);

% zs = 2 * zs;
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
    end
    
    lam = mean(W .* UtUx, 2) ./ (1e-3 + mean(W.^2, 2));
    err = lam .* W - UtUx;
%     err = W - UtU;  
    
    if rem(k,100)==1
        t = t+1;
%         UtUx = exp((UtU - 1) / sig(t));
        
        cnew = mean(err(:).^2) / err0;
        if cold < cnew
%             eta = eta/2;
        else
            cold = cnew;
        end
        fprintf('iter %d, eta %2.4f, time %2.2f, err %2.6f \n', k, eta(k),  toc, cnew)        
        plot(ys(:,1), ys(:,2), '.')
        hold on
        plot(zs(:,1), zs(:,2), 'or', 'Linewidth', 2)
        hold off
        drawnow        
    end
    
    err = err .* lam;
    switch my_metric
        case 'cauchy'
            err = err .* W.^2;
        case 'gaussian'    
            err = err .*  W;        
        case 'exp'
           err = err .*  W;
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
