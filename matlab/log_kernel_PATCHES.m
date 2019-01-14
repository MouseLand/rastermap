%%
X = readNPY('D:/Github/data/patches11.npy');
X = gpuArray(single(X));

X = zscore(X, [], 2);
X = X - mean(X, 1) ;
[U, S, V] = svdecon(X);
U = zscore(U, 1, 2)/size(U,2)^.5;
U = U./sum(U.^2,2).^.5;
%%
nraster = 256;
irand = randperm(size(U,1), nraster);
u0 = U(irand, :);
tic
N = zeros(nraster, 1);
L = gpuArray.zeros(nraster, NN, 'single');
for k = 1:15
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
% u0(N<5, :) = [];

nraster = size(u0,1);
NN = size(U,1);

%%
Ld = 10;

% u0 = u0./sum(u0.^2,2).^.5;
K = u0 * u0';

ndims = 2;
niter = 4000;
eta0 = .1;
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

cW = .25 * gpuArray.ones(NN,1, 'single');

lam = gpuArray.zeros(NN,1, 'single');
olam = gpuArray.zeros(NN,1, 'single');
ocW = gpuArray.zeros(NN,1, 'single');

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
             W = 1 - log(cW + cW' + ds)/log(Ld^2); 
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
            err = err ./(cW+ cW'+ds);
    end
    
    dcW = -mean(err,2) - mean(err,1)';
    dcW = dcW./mean(dcW.^2).^.5;
    ocW = pW * ocW + (1-pW) * dcW;
    
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
    cW = max(0.1, cW - eta(k) * ocW);
end
toc

%% UPSAMPLING
NN = size(U,1);
K  = U * u0';

%%
% K([1:NN] +(imax'-1)*NN) = K([1:NN] +(imax'-1)*NN) - L(imax' + ([1:NN]-1)*nraster) .* u2';

ndims = 2;
niter = 4000;
eta0 = .01;
pW = 0.9;

my_metric = 'neglog';

ys = zs(imax, :);
cWY = cW(imax);
lamY = lam(imax);

ys = gpuArray(single(ys));
eta = linspace(eta0, eta0, niter);

% LAM = gpuArray.ones(size(K), 'single');
LAM = abs(K);
% LAM = exp(K);
err0 = mean(mean(LAM .* K.^2));

dy = gpuArray.zeros(NN, ndims, 'single');
oy = zeros(NN, ndims);
olam = gpuArray.zeros(NN,1, 'single');
ocW = gpuArray.zeros(NN,1, 'single');

% lamY = gpuArray.ones(NN,1, 'single');


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
            W = 1 - log(cWY + cW' + ds)/log(Ld^2); 
    end

    Wlam = W.*exp(lam)';
    
    err = (exp(lamY) .* Wlam - K);
%     lamY = sum(Wlam .* K, 2) ./ (1e-3 + sum(Wlam.^2,2));
%     err = (lamY .* Wlam - K);
    
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
%     err = lamY.* err .* exp(lam)';
    
     switch my_metric
        case 'cauchy'
            err = err .* W.^2;
        case 'gaussian'    
            err = err .*  W;        
        case 'exp'
           err = err .*  W;
        case 'neglog'
            err = err ./(cWY + cW'+ds);
     end

    dcW = -mean(err,2);
    dcW = dcW./mean(dcW.^2).^.5;
    ocW = pW * ocW + (1-pW) * dcW;
    
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
    
    lamY = lamY - eta(k) * olam;
    cWY = max(0.1, cWY - eta(k) * ocW);
end
toc
%%
irand = ceil(rand * size(X,1));
ds = reshape(ys(irand, :), [1, 1, ndims]) - reshape(ys, [1, NN, ndims]);
ds = sq(sum(ds.^2,3));

[~, isort] = sort(ds);

A = gather(X(isort(1:400), :))';
A = reshape(A, 11, 11, 20, 20);
A = permute(A, [1 3 2 4]);
A = reshape(A, 11*20, 11*20);

figure(1)
imagesc(A, [-2, 2])
figure(2)
imagesc(reshape(mean(X(isort(1:400), :), 1), 11, 11))
colormap('gray')
%%