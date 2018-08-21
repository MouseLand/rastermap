% load from python spks and check out results

S = readNPY('/media/carsen/DATA1/BootCamp/mesoscope_cortex/spks.npy');
iscell = readNPY('/media/carsen/DATA1/BootCamp/mesoscope_cortex/iscell.npy');
S = S(logical(iscell(:,1)),:);

%% full algorithm
[isort1, isort2, Sm] = mapTmap(S);
imagesc(Sm(:,1000:3000),[0,5])

%% run map in neurons without smoothing across time sorting
[iclustup, isort, Vout] = activityMap(S);
%%
imagesc(zscore(S(isort,:), 1, 2), [0 3])
