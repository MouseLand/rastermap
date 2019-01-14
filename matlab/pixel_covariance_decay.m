subplot(1,2,1)
imagesc(UtUx, [0 1])

subplot(1,2,2)
imagesc(lam.*W, [0 1])
%%
X = readNPY('D:\Github\rastercode\imgcovDC.npy');

[xs, ys] = meshgrid(1:41, 1:41);
ds = ((xs(:) - xs(:)').^2 + (ys(:) - ys(:)').^2).^.5;

X = reshape(X, 41, 41, 41*41);
ds = reshape(ds, 41, 41, 41*41);
%%
subplot(1,2,1)
semilogx(sq(ds(1,1,:)), sq(X(1,1,:)), '.')

subplot(1,2,2)
x = 1:100;
fx = 1 - log(1+x)/log(100);
semilogx(fx)
