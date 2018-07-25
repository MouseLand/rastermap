load('H:\spont_TX8_2018_03_14')
statall = stat;
%%
ix = [statall.skew]>.5 & ([statall.npix]<120 & [statall.npix]>20);
sum(ix)

S0 = S(ix,:);

S0 = S0 - mean(S0,2);

ops.useGPU = 1;

xs = [statall.xglobal];
ys = [statall.yglobal];
xs = xs(ix);
ys = ys(ix);

%%

[NN, NT] = size(S0);

rng(100);
i0 = ceil(rand*NN);
ds = ((xs(i0) - xs).^2 + (ys(i0) - ys).^2).^.5;

iY = ds<300;
iX = ds>400;

Strain = S0(iX, :);
Stest  = S0(iY, :);

% use this map to predict new neurons
nt0 = 60 * 3;
nBatches = ceil(NT/nt0);

ibatch = ceil(linspace(1e-10, nBatches, NT));

Ntrain = ceil(nBatches * 7/10);
rtrain = randperm(nBatches);

itrain = ismember(ibatch, rtrain(1:Ntrain));
itest  = ismember(ibatch, rtrain((1+Ntrain):end));

%%
addpath('D:\Github\embeddings\matlab')

ops.nCall = [60 200] ;
ops.iPC   = 1:250;

[isort1, isort2, amap] = mapTmap(Strain, ops);

Sm = Strain(isort1, isort2);
Smap = my_conv2(Sm, [5 10], [1 2]);
Sm(isort1, isort2) = Smap;

%%
err = Stest;
for i = 1:10
    b = Sm(:,itrain) * err(:,itrain)';
    a = sum(Sm(:,itrain) .* Sm(:,itrain), 2);
    
    cmax  = b.^2./a;
    
    [~, imax] = max(cmax, [],1);
    
    b = sum(Sm(imax, itrain) .* err(:, itrain),2);
    a = sum(Sm(imax, itrain).^2, 2);
    
    coefs = b./a;
    
    Spred = Sm(imax,:) .* coefs;
    err = err - Spred;
end

1 - mean(mean(err(:, itest).^2))/mean(mean(Stest(:, itest).^2))
1 - mean(mean(err(:, itrain).^2))/mean(mean(Stest(:, itrain).^2))

%%
[U Sv V] = svdecon(gpuArray(Strain));

nPC = 200;
V = U(:, 1:nPC)'*Strain;
V = V/1000;

lam = 200;

c = (V(:,itrain)*V(:,itrain)' + lam * eye(nPC)) \ (V(:,itrain) * Stest(:, itrain)');

Spred = c' * V;

err = Spred - Stest;

1 - mean(mean(err(:, itest).^2))/mean(mean(Stest(:, itest).^2))
1 - mean(mean(err(:, itrain).^2))/mean(mean(Stest(:, itest).^2))







