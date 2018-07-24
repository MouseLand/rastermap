
% function [ccb0, theta] = embed1Dcirc(ccb0, theta0) 

nBatches = size(ccb0,1);
% % 
% [u, s, v] = svdecon(ccb0);
% 
% u = u - mean(u,1);
% theta0 = angle(u(:,1) + 1i*u(:,2));


theta = (theta0);

A0 = ones(nBatches, 'single');
eta = .001;

niB = 100;
Cost = zeros(1, niB);

oTheta = zeros(nBatches,1, 'single');
p = .9;

y = ccb0(:);

tic
for k = 1:100
    A1 = sin(theta) * sin(theta');
    A2 = cos(theta) * cos(theta');
    
    x = [ A1(:) A2(:) A0(:)];    
    
    B = (x'*x)\(x'*y);
    ypred = x * B;
    
    err = ypred - y;
    
    Cost(k) = gather(mean(err(:).^2));
    err = reshape(err, nBatches, nBatches);
    
    d1 = B(1) * (err * sin(theta)) .* cos(theta);
    d2 = - B(2) * (err * cos(theta)) .* sin(theta);
        
    
    dTheta = d1 + d2;
    
    oTheta = p * oTheta + (1-p) * dTheta;
    theta = theta - eta * oTheta;

    if rem(k, 100)>=0
        figure(1)
        plot(Cost(1:k))
        drawnow
        axis tight
    end
end
%%

theta = mod(theta, 2*pi);
thsort = sort(theta);
thsort = [thsort(end)-2*pi; thsort];
[~, imax] = max(diff(thsort));
t0 = thsort(imax); 

ix = theta < t0;
theta(ix) = theta(ix) + 2*pi;

[~, isort] = sort(theta);

ccbsort = ccb0(isort, isort);