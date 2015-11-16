
%%
imgdb = imageSet('./data/owndata', 'recursive');
[training, test] = partition(imgdb, [0.8 0.2]);

%%
trainingFeatures = [];
trainingLabels = [];

for i = 1 : size(training,2)
    imgset = training(i);
    for j = 1 : imgset.Count
        image = read(imgset, j);
        image = imresize(image, [110 110]);
        trainingFeatures = [trainingFeatures; extractHOGFeatures(image)];
        trainingLabels = [trainingLabels; i];
    end
end

%%
Model = fitcecoc(trainingFeatures, trainingLabels);

%%
testFeatures = [];
testLabels = [];
predictedLabels = [];
for i = 1 : size(test,2)
    imgset = test(i);
    for j = 1 : imgset.Count
        image = read(imgset, j);
        image = imresize(image, [110 110]);
        testFeatures = [testFeatures; extractHOGFeatures(image)];
        testLabels = [testLabels; i];
    end
end

%%
predictedLabels = predict(Model, testFeatures);

%%
cm = confusionmat(testLabels, predictedLabels);
accuracy = sum(diag(cm)) / sum(cm(:));
