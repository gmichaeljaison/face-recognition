addpath('./lib');

%%
testim = imread(test(6).ImageLocation{1});

A = double(trainingFeatures);
y = double(imresize(testim, [12 12]));
y = y(:);
epsilon = 0.05;

x0 = A * y;

%%
% xp = l1qc_logbarrier(x0, A, At, b, epsilon, lbtol, mu, cgtol, cgmaxiter)

xp = l1qc_logbarrier(x0, A', [], y, epsilon, 1e-3);

%%
residuals = zeros(40,1);
for i = 1 : 40
    delta_xp = zeros(size(xp));
    range = ((i-1)*8) + 1 : i*8;
    delta_xp(range) = xp(range);
    
    err = y - (A' * delta_xp);
    residuals(i) = norm(err,2);
end

[~,minInd] = min(residuals);