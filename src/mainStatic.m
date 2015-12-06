%%
addpath('../lib');

%%

imgdb = imageSet('../data/small', 'recursive');
[training, test] = partition(imgdb, [0.8 0.2]);

% featureExtractor = RandomExtractFeature();
featureExtractor = EigenFaceFeature();
% featureExtractor = LaplacianFace();
% featureExtractor = HoGExtractFeature();

[images, yTrain] = readImageSet(training);
[testImgs, yTest] = readImageSet(test);

featureExtractor.init(images);
xTrain = featureExtractor.extract(images);
xTest = featureExtractor.extract(testImgs);

% faceRec = L1MinFaceRecognition(xTrain, yTrain, 0.05);

% Model = fitcecoc(xTrain, yTrain);

Model = fitcknn(xTrain, yTrain);


%%

% yPredict = faceRec.predict(xTest);

yPredict = Model.predict(xTest);

c = confusionmat(yPredict, yTest');
accuracy = sum(diag(c)) / sum(c(:));