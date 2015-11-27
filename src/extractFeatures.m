function [trainingFeatures, trainingLabels] = extractFeatures ...
    (training, featureExtractor)

    trainingLabels = {};
    images = {};

    n = 1;
    for i = 1 : size(training,2)
        imgset = training(i);
        for j = 1 : imgset.Count
            image = read(imgset, j);
            images{n} = image;
            trainingLabels{n} = imgset.Description;
            n = n + 1;
        end
    end
    
    trainingFeatures = extract(featureExtractor, images);
    
%     Model = fitcecoc(trainingFeatures, trainingLabels);

end
