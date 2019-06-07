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
[U, S, V] = svdecon(X);
U = U * diag(diag(S).^.5);
U = gpuArray(U(:, 1:256));
u0 = U / mean(sum(U.^2,2).^.5);
% U = zscore(U, 1, 2)/size(U,2)^.5;

%%
X = readNPY('D:/Github/data/allen-visp-alm/X.npy');
X = zscore(X(1:3:end, :), 1, 2)/size(X,2)^.5;
U = gpuArray(single(X));
%%
X = readNPY('D:/Github/data/allen-visp-alm/logCPM.npy');
X = gpuArray(single(X));
X = X - mean(X,1);
[U, S, V] = svdecon(X);
U = X * V;
U = U(:, 1:256);
U = zscore(U, 1, 2)/size(U,2)^.5;
%%
X = gpuArray(single(spks));
X = zscore(X, [], 2);
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
U = X1 * V;

U = gpuArray(U(:, 1:256));
U = zscore(U, 1, 2)/size(U,2)^.5;
%%
nraster = 512;
irand = randperm(size(U,1), nraster);
u0 = U(irand, :);
tic
N = zeros(nraster, 1);
L = gpuArray.zeros(nraster, NN, 'single');
for k = 1:10
    u0 = u0./sum(u0.^2,2).^.5;
    cv = max(0, U * u0');    
    [lam, imax] = max(cv, [], 2);
    
    L = gpuArray.zeros(nraster, NN, 'single');    
    for j = 1:nraster
        N(j)=gather(sum(imax==j));
        ll = lam(imax==j);
        ll = ll/sum(ll.^2).^.5;
        u0(j, :) = ll' * U(imax==j, :);
        L(j, imax==j) = ll;        
    end       
    disp(mean(lam.^2))
end

NN = size(U,1);
%%
Ld = 10;

% u0 = u0./sum(u0.^2,2).^.5;
K = u0 * u0';

ndims = 2;
niter = 4000;
eta0 = .01;
pW = 0.9;

my_metric = 'neglog';

NN = size(u0,1);
zs = u0(:, 1:ndims);
zs = .5 * zs./std(zs,1,1);

zs = gpuArray(single(zs));
dy = gpuArray.zeros(NN, ndims, 'single');
eta = linspace(eta0, eta0, niter);
oy = zeros(NN, ndims);

% LAM = gpuArray.ones(size(K), 'single');
LAM = abs(K);
% LAM = exp(K);
err0 = mean(mean(LAM .* K.^2));


lam = gpuArray.zeros(NN,1, 'single');
olam = gpuArray.zeros(NN,1, 'single');

cold = Inf;
tic
for k = 1:niter    
    ds = gpuArray.zeros(NN, NN, 'single');
    for j = 1:ndims
        ds = ds + (zs(:,j) -zs(:,j)').^2;
    end
    switch my_metric
        case 'cauchy'
            W  = 1./(1 + ds);
        case 'gaussian'    
            W = exp(-ds); 
        case 'exp'
            W = exp(-ds.^.5); 
        case 'neglog'
             W = 1 - log(1 + ds)/log(Ld^2); 
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
            
            plot(zs(:,1), zs(:,2), '.')
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
            err = err ./(1+ds);
    end
    
    for i = 1:ndims
        switch my_metric
            case 'exp'
                err2 = err2 .* (1e-2 + ds).^-.5;
        end
        err2 = err .* (zs(:,i)  - zs(:,i)');
        D = mean(err2, 2);
        E = mean(err2, 1);
        dy(:,i) = -D + E'; % + D1 + D2;
    end
    dy = dy./mean(dy.^2,2).^.5;
    
    oy = pW * oy + (1-pW) * dy;
    
    zs = zs - eta(k) * oy;
    lam = lam - eta(k) * olam;
end
toc

%% UPSAMPLING
NN = size(U,1);
K  = U * u0';
% K([1:NN] +(imax'-1)*NN) = K([1:NN] +(imax'-1)*NN) - L(imax' + ([1:NN]-1)*nraster) .* u2';

ndims = 2;
niter = 4000;
eta0 = .01;
pW = 0.9;

my_metric = 'neglog';

ys = zs(imax, :);

ys = gpuArray(single(ys));
eta = linspace(eta0, eta0, niter);

% LAM = gpuArray.ones(size(K), 'single');
LAM = .1 + abs(K);
% LAM = exp(K);
err0 = mean(mean(LAM .* K.^2));

dy = gpuArray.zeros(NN, ndims, 'single');
oy = zeros(NN, ndims);

lamY = gpuArray.ones(NN,1, 'single');
olam = gpuArray.zeros(NN,1, 'single');


cold = Inf;
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
            W = exp(-ds/Ld^2); 
        case 'exp'
            W = exp(-ds.^.5); 
        case 'neglog'
            W = 1 - log(1 + ds)/log(Ld^2); 
    end

    Wlam = W.*exp(lam)';
    
    err = (exp(lamY) .* Wlam - K);
    
    if rem(k,100)==1
         cnew = mean(mean(LAM .* err.^2)) / err0;
        if cold < cnew
%             eta = eta/2;
        else
            cold = cnew;
        end
        fprintf('iter %d, eta %2.4f, time %2.2f, err %2.6f \n', k, eta(k),  toc, cnew)        
        if ndims>1
            plot(ys(:,1), ys(:,2), '.')
            hold on
            plot(zs(:,1), zs(:,2), 'or')
            hold off
            drawnow
        end
    end
    
    err = err .* LAM;
    
    dlam = mean(err.*Wlam,2).*exp(lamY);    
    dlam = dlam./mean(dlam.^2).^.5;
    olam = pW * olam + (1-pW) * dlam;    
    
    err = exp(lamY).* err .* exp(lam)';
    
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
        dy(:,i) = -D; % + D1 + D2;
    end
    dy = dy./sum(dy.^2,2).^.5;
    
    oy = pW * oy + (1-pW) * dy;
    ys = ys - eta(k) * oy;    
    
    lamY = lamY - 10*eta(k) * olam;
end
toc


drawnow
%%
ikr = randperm(NN, min(NN, 1e3));
ds = reshape(ys(ikr, :), [numel(ikr), 1, ndims]) - reshape(ys, [1, NN, ndims]);

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






