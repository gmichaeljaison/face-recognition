function [ Model ] = learnRecognition( training )

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
    
    Model = fitcecoc(trainingFeatures, trainingLabels);

end
