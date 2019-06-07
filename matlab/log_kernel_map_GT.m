NN = 10000;
yGT = 20*randn(NN, 2);
% [x,y] = meshgrid(1:41, 1:41);
% yGT = [x(:) y(:)];
NN = size(yGT,1);

ds = gpuArray.zeros(NN, NN, 'single');
for j = 1:ndims
    ds = ds + (yGT(:,j) -yGT(:,j)').^2;
end
Ld = 20;
K =  1  - log(.01 + ds)/log(Ld^2);

[U, S, V] = svdecon(K);

%%
% Ld = 5;

ndims = 2;
niter = 4000;
eta0 = .1;
pW = 0.9;

my_metric = 'neglog';

NN = size(u0,1);
zs = u0(:, 1:ndims);
zs = .1 * zs./std(zs,1,1);

zs = gpuArray(single(zs));
dy = gpuArray.zeros(NN, ndims, 'single');
eta = linspace(eta0, eta0, niter);
oy = zeros(NN, ndims);

LAM = gpuArray.ones(size(K), 'single');
% LAM = .1 + abs(K);
err0 = mean(mean(LAM .* K.^2));


lam = gpuArray.ones(NN,1, 'single');
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
        
    Wlam = W .* lam';
    err = (lam .* Wlam - K);
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
    
    dlam = mean(err.*Wlam,2);    
    dlam = dlam./mean(dlam.^2).^.5;
    olam = pW * olam + (1-pW) * dlam;
    
    err = lam.*err.* lam';
    
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
