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
X = readNPY('D:/Github/data/allen-visp-alm/logCPM.npy');
X = gpuArray(single(X(1:3:end, :)));
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
% U = zscore(U, 1, 2)/size(U,2)^.5;
%%
nraster = 300;
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
    
end

L0  = L;

NN = size(U,1);
%%
ndims = 2;
niter = 101;
eta0 = .1;
pW = 0.9;
eta = linspace(eta0, eta0, niter);
Ld = 20;

% zs = .1 * u0(:, 1:ndims);
dz = gpuArray.zeros(nraster, ndims, 'single');
oz = gpuArray.zeros(nraster, ndims, 'single');
vtv = u0 * u0';
err0 = mean(mean(vtv.^2));

%%

for k = 1:niter         
    ds = sum((reshape(zs, [nraster, 1, ndims]) - reshape(zs, [1, nraster, ndims])).^2,3);
    
    K = 1 - log(1 + ds)/log(Ld^2);
    aK = mean(K(:).*vtv(:)) / mean(K(:).^2);
    K = aK * K;
    
    err = K - vtv;    
    
    if rem(k,100)==1
        cnew = mean(err(:).^2)/err0;
        fprintf('iter %d, eta %2.4f, time %2.2f, err %2.6f \n', k, eta(k),  toc, cnew)
        if ndims>1
            plot(zs(:,1), zs(:,2), 'or')
            drawnow
        end        
    end
    
    if rem(k,100)==0
        %% update neuron allocations
        upred = K * L;
        cv = (upred * U) * U';
        vnorms = 1e-3 + sum(upred.^2,2).^.5;
        [lam, imax] = max(max(0,cv)./vnorms, [],1);
        fprintf('Kerr %2.6f \n',mean(lam.^2))
        
        lam = lam(:)./vnorms(imax);
        
        L = gpuArray.zeros(nraster, NN, 'single');
        for j = 1:nraster
            if sum(imax==j)>0
                ll = lam(imax==j);
                ll = ll/(1e-4 + sum(ll.^2).^.5);
                L(j, imax==j) = ll;
            end
        end
        u0 = L * U;
        vtv = u0 * u0';        
        
    else        
        err = err ./(1+ds);
        for i = 1:ndims
            err2 = err .* (zs(:,i)  - zs(:,i)');
            D = mean(err2, 2);
            E = mean(err2, 1);
            dz(:,i) = -D + E'; % + D1 + D2;
        end
        
        dz = dz./sum(dz.^2,2).^.5;
        
        oz = pW * oz + (1-pW) * dz;
        zs = zs - eta(k) * oz;
    end
end

%%
[xs, ys] = meshgrid(1:21, 1:21);
ds = (xs(:) - xs(:)').^2 + (ys(:) - ys(:)').^2;
K = 1 - log(1+ds)/log(10^2);
K = gpuArray(K);
nraster = size(K,1);

L = gpuArray.randn(size(K,1), NN);
% K = gpuArray.eye(nraster, 'single');

for k = 1:20    
    upred = K * L;
    cv = (upred * U) * U';
    vnorms = 1e-3 + sum(upred.^2,2);
    [lam, imax] = max(max(0,cv).^2./vnorms, [],1);
    fprintf('Kerr %2.6f \n',mean(lam))
    
    lam = sqrt(lam(:)./vnorms(imax));
    
    L = gpuArray.zeros(nraster, NN, 'single');
    for j = 1:nraster
        if sum(imax==j)>0
            ll = lam(imax==j);
            ll = ll/(1e-4 + sum(ll.^2).^.5);
            L(j, imax==j) = ll;
        end
    end
%     u0 = L * U;
%     vtv = u0 * u0';
end
        
        %%