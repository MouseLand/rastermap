S = gpuArray.randn(5000, 20000, 'single');

tic 
svdecon(S);
toc

%%
addpath('D:\Github\embeddings\matlab')

ops.NC = [30 60];

[isort1, isort2, Sm] = doubleMap(S, ops);
%%

imagesc(Sm)