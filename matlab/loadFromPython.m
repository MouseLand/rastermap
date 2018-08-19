% load from python spks and check out results

S = readNPY('../../bigdat/spks.npy');
iscell = readNPY('../../bigdat/iscell.npy');
S = S(logical(iscell(:,1)),:);

%% map in neurons
[iclustup, isort, Vout] = activityMap(S);

%%
Sm = my_conv2(S(isort,:)-mean(S(isort,:),2), 10, 1);

%%
imagesc(zscore(Sm, 1, 2), [0 3])