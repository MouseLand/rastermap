
ix = [statall.skew]>.5 & ([statall.npix]<120 & [statall.npix]>20);
sum(ix)

S0 = S(ix,:);

S0 = S0 - mean(S0,2);

ops.useGPU = 1;

[isort1, isort2, amap] = mapTmap(S0);

% [iclustup, isort, Vout] = activityMap(S0);

%%
figure;
Sm = S0(isort1, isort2);
Sm = my_conv2(Sm, [10 10], [1 2]);

NT = size(S0,2);

% Sm(:, isort2) = Sm;

imagesc(zscore(Sm, 1, 2), [-3 10])

% imagesc(zscore(Sm(:, isortback), 1, 2), [-3 10])

%%