%%
addpath('../lib');

%%

imgdb = imageSet('../data/small', 'recursive');
[training, test] = partition(imgdb, [0.8 0.2]);

% featureExtractor = RandomExtractFeature();
% featureExtractor = EigenFaceFeature();
featureExtractor = LaplacianFace();
% featureExtractor = HoGExtractFeature();

[images, yTrain] = readImageSet(training);
[testImgs, yTest] = readImageSet(test);

featureExtractor.init(images);
xTrain = featureExtractor.extract(images);
xTest = featureExtractor.extract(testImgs);

% faceRec = L1MinFaceRecognition(xTrain, yTrain, 0.05);

% Model = fitcecoc(xTrain, yTrain);


%%

% yPredict = faceRec.predict(xTest);

Model = fitcknn(xTrain, yTrain, 'NumNeighbors', 1);
yPredict = Model.predict(xTest);

c = confusionmat(yPredict, yTest');
accuracy = sum(diag(c)) / sum(c(:));

%%

[labels,~,ic] = unique(yTrain);

avgXTrain = zeros(numel(labels), size(xTrain,2));
for i = 1 : numel(labels)
    xi = xTrain(ic == i, :);
    avgXTrain(i,:) = mean(xi);
end

nTest = size(xTest,1);
yPredict = cell(nTest,1);
for i = 1 : nTest
    xi = xTest(i,:);
    dist = pdist2(avgXTrain, xi);
%     dist = pdist2(avgXTrain, xi, 'mahalanobis', nancov(avgXTrain));
    [~,mI] = min(dist);
    yPredict{i} = labels{mI};
end

c = confusionmat(yPredict, yTest');
accuracy = sum(diag(c)) / sum(c(:));

%%

Model = fitcknn(avgXTrain, labels, 'NumNeighbors', 1);
yPredict = Model.predict(xTest);

c = confusionmat(yPredict, yTest');
accuracy = sum(diag(c)) / sum(c(:));

