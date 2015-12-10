%%
addpath('../lib');

%%

imgdb = imageSet('../data/CroppedYale', 'recursive');
[training, test] = partition(imgdb, [0.5 0.5]);

% featureExtractor = ResizeExtractFeature();
% featureExtractor = EigenFaceFeature();
featureExtractor = LaplacianFace();
% featureExtractor = HoGExtractFeature();

[images, yTrain] = readImageSet(training);
[testImgs, yTest] = readImageSet(test);

%%

% dims = 15 : 10 : 110;
% dims = [15 : 10 : 110]';
dims = 35;
accuracy = zeros(numel(dims), 1);

for i = 1 : numel(dims)
    featureExtractor.dimension = dims(i);
    featureExtractor.eigenF.dimension = dims(i);
    
    featureExtractor.init(images);
    xTrain = featureExtractor.extract(images);
    xTest = featureExtractor.extract(testImgs);

%     Model = fitcknn(xTrain, yTrain, 'NumNeighbors', 1, ...
%         'Distance', 'euclidean', 'DistanceWeight', 'inverse');
%     Model = fitcknn(xTrain, yTrain, 'NumNeighbors', 1, ...
%         'Distance', 'mahalanobis', 'DistanceWeight', 'inverse');
   Model = L1MinFaceRecognition(xTrain, yTrain, 0.05);
    
   yPredict = Model.predict(xTest);
    
    c = confusionmat(yPredict, yTest');
    accuracy(i) = sum(diag(c)) / sum(c(:));
end

clear c i;

%%
% 
% [labels,~,ic] = unique(yTrain);
% 
% avgXTrain = zeros(numel(labels), size(xTrain,2));
% for i = 1 : numel(labels)
%     xi = xTrain(ic == i, :);
%     avgXTrain(i,:) = mean(xi);
% end
% 
% 
% Model = fitcknn(avgXTrain, labels, 'NumNeighbors', 1, ...
%     'Distance', 'euclidean', 'DistanceWeight', 'inverse');
% yPredict = Model.predict(xTest);
% 
% c = confusionmat(yPredict, yTest');
% accuracy = sum(diag(c)) / sum(c(:));

%{
nTest = size(xTest,1);
yPredict = cell(nTest,1);
for i = 1 : nTest
    xi = xTest(i,:);
    dist = pdist2(avgXTrain, xi);
%     dist = pdist2(avgXTrain, xi, 'mahalanobis', nancov(avgXTrain));
    [D,mI] = min(dist);
    yPredict{i} = labels{mI};
end

c = confusionmat(yPredict, yTest');
accuracy = sum(diag(c)) / sum(c(:));
%}


